"""
YOLOv11 训练脚本（支持 TensorBoard 实时可视化）
说明：
- 所有超参数和数据路径在 train_yolo 默认参数中声明，直接修改即可生效
- 自动生成实验名（带时间戳）避免覆盖
- 自动启动 TensorBoard（线程方式），训练过程中可实时查看
"""

from ultralytics import YOLO
import torch
import time
from pathlib import Path
from datetime import datetime
import threading
import subprocess
import sys


def start_tensorboard(log_dir, port=6006):
    import socket
    log_dir = Path(log_dir)
    # 自动寻找可用端口
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                s.close()
                break
            except OSError:
                print(f"[WARN] 端口 {port} 已被占用，尝试 {port+1}")
                port += 1
    
    print(f"[INFO] 等待日志生成: {log_dir} ...")
    while not any(log_dir.glob('**/events.out.tfevents*')):
        time.sleep(2)
    
    print(f"[INFO] 启动 TensorBoard 服务 (端口 {port}) ...")
    try:
        subprocess.Popen(
            ["tensorboard", "--logdir", str(log_dir),
             "--port", str(port), "--reload_multifile", "true"],
             shell=True
        )
        print(f"[INFO] ✅ TensorBoard 可访问: http://localhost:{port}")
    except Exception as e:
        print(f"[ERROR] 启动 TensorBoard 失败: {e}")

def train_yolo(
    # ================= 基础配置 =================
    task='detect',        # 任务类型: 'detect'(目标检测) 或 'classify'(图像分类)
    model_size='x',         # 模型尺寸: n(nano), s(small), m(medium), l(large), x(xlarge)
    data_yaml='data/data.yaml',  # 数据集配置文件路径(YAML格式)
    epochs=100,             # 训练总轮数(epoch)
    batch=1,                # 每批次处理的图像数量(batch size)
    imgsz=224,              # 输入图像尺寸(正方形边长)
    device='',              # 训练设备: 空字符串自动选择, '0'表示GPU0, 'cpu'表示CPU
    workers=8,              # 数据加载的线程数(建议设为CPU核心数的50-75%)
    project='runs/train',   # 训练结果保存根目录
    name=None,              # 实验名称(自动生成时间戳若为None)
    exist_ok=False,         # 是否允许覆盖同名实验目录
    pretrained=True,        # 是否使用预训练权重
    patience=50,            # 早停机制等待轮数(验证指标无改善时)
    save_period=10,         # 每隔多少轮保存一次模型

    # ================= 优化器参数 =================
    lr0=0.01,               # 初始学习率(典型值0.01-0.001)
    lrf=0.01,               # 最终学习率(lr0 * lrf)
    momentum=0.937,         # 优化器动量参数
    weight_decay=0.0005,    # 权重衰减(L2正则化系数)
    warmup_epochs=3.0,      # 学习率预热轮数(初始阶段逐步提高学习率)

    # ================= 数据增强 =================
    hsv_h=0.015,            # 色调增强幅度(0-0.5)
    hsv_s=0.7,              # 饱和度增强幅度(0-1)
    hsv_v=0.4,              # 亮度增强幅度(0-1)
    degrees=0.0,            # 图像旋转角度范围(度)
    translate=0.1,          # 图像平移比例(相对于图像尺寸)
    scale=0.5,              # 图像缩放比例范围(0-1)
    shear=0.0,              # 图像剪切幅度(度)
    perspective=0.0,        # 透视变换幅度(0-0.001)
    flipud=0.0,             # 上下翻转概率(0-1)
    fliplr=0.5,             # 左右翻转概率(0-1)
    mosaic=1.0,             # 使用马赛克数据增强的概率(0-1)
    mixup=0.0,              # 使用MixUp数据增强的概率(0-1)
    copy_paste=0.0,         # 使用复制粘贴增强的概率(0-1)

    # ================= 高级设置 =================
    amp=True,               # 是否启用自动混合精度训练(加速训练)
    
):
    """YOLOv11训练函数"""
    
    # ------------------------- 设备选择 -------------------------
    if device == '':
        device = '0' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu' and not torch.cuda.is_available():
        print(f"[WARN] 指定设备 '{device}' 无 GPU，已回退为 CPU")
        device = 'cpu'
    amp_flag = amp if device != 'cpu' else False
    if device == 'cpu' and amp:
        print("[INFO] 当前使用 CPU，自动禁用 AMP（amp=False）")
    
    print(f"[INFO] 使用设备: {device}")
    if torch.cuda.is_available() and device != 'cpu':
        try:
            print(f"[INFO] GPU 名称: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    
    # ------------------------- 实验名 -------------------------
    if name is None:
        name = f"crop_pests_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[INFO] 实验名称: {name}")
    
    # ------------------------- 模型加载 -------------------------
    model_name = f'yolo11{model_size}-cls.pt' if task=='classify' else f'yolo11{model_size}.pt'
    print(f"[INFO] 加载模型: {model_name}")
    print(f"[INFO] 数据集配置: {data_yaml}")
    print(f"[INFO] imgsz={imgsz}, batch={batch}, epochs={epochs}")
    
    # ------------------------- 模型加载 -------------------------
    model = YOLO(model_name)

    # ✅ 启动 TensorBoard 手动记录
    from torch.utils.tensorboard import SummaryWriter
    log_dir = Path(project) / name
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_scalar("init/ready", 0, 0)
    writer.flush()
    print(f"[INFO] TensorBoard 日志路径: {log_dir}")

    def tb_callback(trainer):
        if hasattr(trainer, 'metrics') and trainer.metrics is not None:
            metrics = trainer.metrics.results_dict
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(k, v, trainer.epoch)
            writer.flush()

    model.add_callback("on_train_epoch_end", tb_callback)

    # ✅ 启动 TensorBoard 服务线程
    tb_thread = threading.Thread(target=start_tensorboard, args=(log_dir,))
    tb_thread.daemon = True
    tb_thread.start()
    
    # ------------------------- 训练参数 -------------------------
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch,
        'imgsz': imgsz,
        'device': device,
        'workers': workers,
        'project': project,
        'name': name,
        'exist_ok': exist_ok,
        'pretrained': pretrained,
        'patience': patience,
        'save': True,
        'save_period': save_period,
        'cache': False,
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': False,
        'amp': amp_flag,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'optimizer': 'auto',
        'lr0': lr0,
        'lrf': lrf,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': hsv_h,
        'hsv_s': hsv_s,
        'hsv_v': hsv_v,
        'degrees': degrees,
        'translate': translate,
        'scale': scale,
        'shear': shear,
        'perspective': perspective,
        'flipud': flipud,
        'fliplr': fliplr,
        'mosaic': mosaic,
        'mixup': mixup,
        'copy_paste': copy_paste,
        'plots': True,
        'val': True,
    }
    
    # ------------------------- 开始训练 -------------------------
    print("\n[INFO] 开始训练...\n")
    results = model.train(**train_args)
    
    print("\n[INFO] 训练完成！")
    try:
        best_path = getattr(model.trainer, 'best', None)
        last_path = getattr(model.trainer, 'last', None)
        print(f"[INFO] 最佳模型路径: {best_path}")
        print(f"[INFO] 最后模型路径: {last_path}")
    except Exception:
        pass
    
    # ------------------------- 验证最佳模型 -------------------------
    print("\n[INFO] 验证最佳模型...")
    metrics = model.val(data=data_yaml, imgsz=imgsz)
    if task == 'classify':
        top1 = getattr(metrics, 'top1', None)
        top5 = getattr(metrics, 'top5', None)
        if top1 is not None and top5 is not None:
            print(f"\n[RESULT] Top-1: {top1:.4f}, Top-5: {top5:.4f}")
        else:
            print("\n[RESULT] 分类验证完成（未提供 top1/top5 指标）")
    else:
        box = getattr(metrics, 'box', None)
        if box is not None:
            map50 = getattr(box, 'map50', None)
            map_all = getattr(box, 'map', None)
            if map50 is not None:
                print(f"\n[RESULT] mAP50: {map50:.4f}")
            if map_all is not None:
                print(f"[RESULT] mAP50-95: {map_all:.4f}")
        else:
            print("\n[RESULT] 检测验证完成（无 box 指标信息）")
    
    return results

if __name__ == '__main__':
    # 'main' 中不传参，使用 def 中的默认值
    train_yolo()