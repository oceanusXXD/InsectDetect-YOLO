"""
train.py — YOLOv11 训练脚本（符合 COMP9517 项目要求）
- 支持 detect / classify
- AutoBatch：batch=-1 自动探测最大 batch
- 每个 epoch 保存权重（save_period=1）
- 训练结束自动在 val 集上跑官方 evaluator（mAP 或分类指标）
- 导出 metrics 到 JSON/CSV，并记录训练/验证耗时 timings.json
- 自动启动 TensorBoard（端口自动避让）
"""

import os
import json
import csv
import time
import threading
import subprocess
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO


# ------------------------- TensorBoard -------------------------
def _start_tb_when_ready(log_dir: Path, port: int = 6006):
    import socket
    log_dir = Path(log_dir)
    # 端口避让
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                s.close()
                break
            except OSError:
                print(f"[WARN] TensorBoard 端口 {port} 占用，尝试 {port+1}")
                port += 1

    print(f"[INFO] 等待 tfevents 出现于: {log_dir} ...")
    while not any(log_dir.glob('**/events.out.tfevents*')):
        time.sleep(2)

    print(f"[INFO] 启动 TensorBoard (port={port})")
    try:
        subprocess.Popen(
            ["tensorboard", "--logdir", str(log_dir),
             "--port", str(port), "--reload_multifile", "true"],
            shell=True
        )
        print(f"[INFO] ✅ TensorBoard: http://localhost:{port}")
    except Exception as e:
        print(f"[ERROR] 启动 TensorBoard 失败: {e}")


# ------------------------- 指标/耗时写盘 -------------------------
def _dump_metrics(save_dir: Path, metrics_obj, task: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    d = {}

    # 检测
    if task == 'detect' and hasattr(metrics_obj, 'box') and metrics_obj.box is not None:
        d['map50'] = float(getattr(metrics_obj.box, 'map50', float('nan')))
        d['map50_95'] = float(getattr(metrics_obj.box, 'map', float('nan')))
        d['mp'] = float(getattr(metrics_obj.box, 'mp', float('nan')))   # mean precision
        d['mr'] = float(getattr(metrics_obj.box, 'mr', float('nan')))   # mean recall

    # 分类（不同版本字段略有差异，这里尽量兼容）
    if task == 'classify':
        for k in ['top1', 'top5', 'accuracy', 'precision', 'recall', 'f1', 'auc']:
            v = getattr(metrics_obj, k, None)
            if v is not None:
                d[k] = float(v)

    with open(save_dir / "metrics_summary.json", "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

    with open(save_dir / "metrics_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(d.keys()))
        w.writeheader()
        if d:
            w.writerow(d)

    print(f"[INFO] 指标已导出: {save_dir}/metrics_summary.(json|csv)")


def _dump_timings(save_dir: Path, timings: dict):
    with open(save_dir / "timings.json", "w") as f:
        json.dump(timings, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 耗时已导出: {save_dir}/timings.json")


# ------------------------- 训练入口 -------------------------
def train_yolo(
    # 基础
    task='detect',                 # 'detect' or 'classify'
    model_size='x',                # n/s/m/l/x
    data_yaml='data/data.yaml',    # 检测: 标注 YAML；分类: 分类数据 YAML
    epochs=50,
    batch=-1,                      # ✅ AutoBatch
    imgsz=None,                    # None -> detect:896 / classify:224
    device='0',
    workers=16,
    project='runs/train',
    name=None,
    exist_ok=False,
    pretrained=True,
    patience=100,
    save_period=1,                 # ✅ 每轮保存

    # 优化器 & 调度
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3.0,
    cos_lr=True,

    # 数据增强（检测）
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0,

    # 其他
    amp=True, deterministic=True, seed=42, close_mosaic=10
):
    # 设备
    if not device:
        device = '0' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu' and not torch.cuda.is_available():
        print(f"[WARN] 指定设备 {device} 不可用，回退 CPU")
        device = 'cpu'
    amp_flag = (device != 'cpu') and amp

    # 输入尺寸
    if imgsz is None:
        imgsz = 896 if task == 'detect' else 224

    # 实验名
    if name is None:
        name = f"crop_pests_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[INFO] 实验名: {name}")
    if torch.cuda.is_available() and device != 'cpu':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # 模型
    model_name = f'yolo11{model_size}-cls.pt' if task == 'classify' else f'yolo11{model_size}.pt'
    print(f"[INFO] 加载模型: {model_name}")
    model = YOLO(model_name)

    # TensorBoard
    log_dir = Path(project) / name
    tb_thread = threading.Thread(target=_start_tb_when_ready, args=(log_dir,))
    tb_thread.daemon = True
    tb_thread.start()

    # 训练参数
    train_args = dict(
        task=task,
        data=data_yaml,
        epochs=epochs,
        batch=batch,                # ✅ -1 自动探测
        imgsz=imgsz,
        device=device,
        workers=workers,
        project=project,
        name=name,
        exist_ok=exist_ok,
        pretrained=pretrained,
        patience=patience,
        save=True,
        save_period=save_period,    # ✅
        verbose=True,
        seed=seed,
        deterministic=deterministic,
        resume=False,
        amp=amp_flag,
        optimizer='auto',
        lr0=lr0, lrf=lrf, momentum=momentum, weight_decay=weight_decay,
        warmup_epochs=warmup_epochs, warmup_momentum=0.8, warmup_bias_lr=0.1,
        plots=True,
        val=True,
    )
    if task == 'detect':
        train_args.update(
            dict(
                rect=False, cos_lr=cos_lr, close_mosaic=close_mosaic,
                hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v,
                degrees=degrees, translate=translate, scale=scale, shear=shear, perspective=perspective,
                flipud=flipud, fliplr=fliplr, mosaic=mosaic, mixup=mixup, copy_paste=copy_paste,
                box=7.5, cls=0.5, dfl=1.5,
            )
        )

    timings = {}

    # 训练
    print(f"[INFO] 开始训练：task={task}, epochs={epochs}, batch={batch}, imgsz={imgsz}")
    t0 = time.time()
    results = model.train(**train_args)
    timings['train_seconds'] = round(time.time() - t0, 3)

    # best/last
    try:
        best_path = getattr(model.trainer, 'best', '')
        last_path = getattr(model.trainer, 'last', '')
        print(f"[INFO] 最佳模型: {best_path}")
        print(f"[INFO] 最后模型: {last_path}")
    except Exception:
        pass

    # 官方 evaluator 验证
    print("[INFO] 评测验证集（官方 evaluator）...")
    t1 = time.time()
    metrics = model.val(data=data_yaml, imgsz=imgsz, device=device, plots=True, save_json=False)
    timings['val_seconds'] = round(time.time() - t1, 3)

    # 写指标/耗时
    save_dir = Path(model.trainer.save_dir)
    _dump_metrics(save_dir, metrics, task)
    _dump_timings(save_dir, timings)

    print("[INFO] ✅ 训练+评测完成")
    return results


if __name__ == '__main__':
    # A100 推荐：detect 任务
    train_yolo(
        task='detect',
        model_size='x',
        data_yaml='data/data.yaml',
        epochs=50,
        batch=-1,          # ✅ AutoBatch
        imgsz=896,
        device='0',
        workers=24,
        project='runs/train',
        exist_ok=False,
        pretrained=True,
        patience=100,
        save_period=1
    )
