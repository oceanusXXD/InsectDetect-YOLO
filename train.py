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
    # Port conflict resolution
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                s.close()
                break
            except OSError:
                print(f"[WARN] TensorBoard port {port} occupied, trying {port+1}")
                port += 1

    print(f"[INFO] Waiting for tfevents in: {log_dir} ...")
    while not any(log_dir.glob('**/events.out.tfevents*')):
        time.sleep(2)

    print(f"[INFO] Starting TensorBoard (port={port})")
    try:
        subprocess.Popen(
            ["tensorboard", "--logdir", str(log_dir),
             "--port", str(port), "--reload_multifile", "true"],
            shell=True
        )
        print(f"[INFO] ✅ TensorBoard: http://localhost:{port}")
    except Exception as e:
        print(f"[ERROR] Failed to start TensorBoard: {e}")


# ------------------------- Metrics/Timing Export -------------------------
def _dump_metrics(save_dir: Path, metrics_obj, task: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    d = {}

    if task == 'detect' and hasattr(metrics_obj, 'box') and metrics_obj.box is not None:
        d['map50'] = float(getattr(metrics_obj.box, 'map50', float('nan')))
        d['map50_95'] = float(getattr(metrics_obj.box, 'map', float('nan')))
        d['mp'] = float(getattr(metrics_obj.box, 'mp', float('nan')))
        d['mr'] = float(getattr(metrics_obj.box, 'mr', float('nan')))

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

    print(f"[INFO] Metrics exported to: {save_dir}/metrics_summary.(json|csv)")


def _dump_timings(save_dir: Path, timings: dict):
    with open(save_dir / "timings.json", "w") as f:
        json.dump(timings, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Timings exported to: {save_dir}/timings.json")


# ------------------------- Training Entry Point -------------------------
def train_yolo(
    task='detect',
    model_size='x',
    data_yaml='data/data.yaml',
    epochs=50,
    batch=-1,
    imgsz=None,
    device='0',
    workers=16,
    project='runs/train',
    name=None,
    exist_ok=False,
    pretrained=True,
    patience=100,
    save_period=1,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3.0,
    cos_lr=True,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0,
    amp=True, deterministic=True, seed=42, close_mosaic=10
):
    if not device:
        device = '0' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu' and not torch.cuda.is_available():
        print(f"[WARN] Device {device} unavailable, falling back to CPU")
        device = 'cpu'
    amp_flag = (device != 'cpu') and amp

    if imgsz is None:
        imgsz = 896 if task == 'detect' else 224

    if name is None:
        name = f"crop_pests_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[INFO] Experiment Name: {name}")
    if torch.cuda.is_available() and device != 'cpu':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    model_name = f'yolo11{model_size}-cls.pt' if task == 'classify' else f'yolo11{model_size}.pt'
    print(f"[INFO] Loading Model: {model_name}")
    model = YOLO(model_name)

    log_dir = Path(project) / name
    tb_thread = threading.Thread(target=_start_tb_when_ready, args=(log_dir,))
    tb_thread.daemon = True
    tb_thread.start()

    train_args = dict(
        task=task,
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project=project,
        name=name,
        exist_ok=exist_ok,
        pretrained=pretrained,
        patience=patience,
        save=True,
        save_period=save_period,
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

    print(f"[INFO] Starting Training: task={task}, epochs={epochs}, batch={batch}, imgsz={imgsz}")
    t0 = time.time()
    results = model.train(**train_args)
    timings['train_seconds'] = round(time.time() - t0, 3)

    try:
        best_path = getattr(model.trainer, 'best', '')
        last_path = getattr(model.trainer, 'last', '')
        print(f"[INFO] Best Model: {best_path}")
        print(f"[INFO] Last Model: {last_path}")
    except Exception:
        pass

    print("[INFO] Running validation (official evaluator)...")
    t1 = time.time()
    metrics = model.val(data=data_yaml, imgsz=imgsz, device=device, plots=True, save_json=False)
    timings['val_seconds'] = round(time.time() - t1, 3)

    save_dir = Path(model.trainer.save_dir)
    _dump_metrics(save_dir, metrics, task)
    _dump_timings(save_dir, timings)

    print("[INFO]  Training & Evaluation Complete")
    return results


if __name__ == '__main__':
    train_yolo(
        task='detect',
        model_size='x',
        data_yaml='data/data.yaml',
        epochs=50,
        batch=-1,
        imgsz=896,
        device='0',
        workers=24,
        project='runs/train',
        exist_ok=False,
        pretrained=True,
        patience=100,
        save_period=1
    )