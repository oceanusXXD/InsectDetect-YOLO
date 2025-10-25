"""
YOLOv11 预测/检测脚本（带性能指标和绘图）
支持单张图片、文件夹、视频预测
"""
from ultralytics import YOLO
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def load_gt_labels(gt_folder):
    """
    加载真实标签
    假设每张图片对应一个 .txt 文件，YOLO 格式：class x_center y_center w h (归一化)
    """
    gt_dict = {}
    gt_path = Path(gt_folder)
    if not gt_path.exists():
        print(f"[警告] GT 文件夹不存在: {gt_folder}")
        return gt_dict
    for txt_file in gt_path.glob("*.txt"):
        img_name = txt_file.stem
        labels = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x, y, w, h = map(float, parts)
                    labels.append([cls, x, y, w, h])
        gt_dict[img_name] = np.array(labels)
    return gt_dict

def compute_metrics(results, gt_folder=None, iou_thresh=0.5):
    """
    计算 Precision / Recall / F1-score
    gt_folder: 真实标签文件夹 (可选)
    """
    precisions, recalls, f1_scores = [], [], []
    class_names = results[0].names if len(results) > 0 else []

    # 加载真实标签
    gt_dict = load_gt_labels(gt_folder) if gt_folder else {}

    for result in results:
        # 预测
        pred_boxes = np.array([box.xyxy.cpu().numpy().flatten() for box in result.boxes])
        pred_classes = np.array([int(box.cls[0]) for box in result.boxes])
        pred_scores = np.array([float(box.conf[0]) for box in result.boxes])

        # GT
        img_stem = Path(result.path).stem
        gt_labels = gt_dict.get(img_stem, np.zeros((0,5)))
        gt_boxes = gt_labels[:,1:] if len(gt_labels) > 0 else np.zeros((0,4))
        gt_classes = gt_labels[:,0] if len(gt_labels) > 0 else np.zeros((0,))

        # 简单匹配 TP/FP/FN
        # 这里只做每类统计，实际可用 pycocotools 更精确
        for cls_idx, cls_name in enumerate(class_names):
            pred_mask = pred_classes == cls_idx
            gt_mask = gt_classes == cls_idx
            n_pred = pred_mask.sum()
            n_gt = gt_mask.sum()

            # 粗略 TP 计算：min(n_pred, n_gt)
            tp = min(n_pred, n_gt)
            fp = n_pred - tp
            fn = n_gt - tp

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

    # 转换为 numpy
    precisions = np.array(precisions).reshape(-1, len(class_names))
    recalls = np.array(recalls).reshape(-1, len(class_names))
    f1_scores = np.array(f1_scores).reshape(-1, len(class_names))

    # 取每类平均
    precision_mean = precisions.mean(axis=0)
    recall_mean = recalls.mean(axis=0)
    f1_mean = f1_scores.mean(axis=0)

    return precision_mean, recall_mean, f1_mean, class_names

def plot_metrics(precision, recall, f1, class_names):
    """
    绘制柱状图 + PR曲线
    """
    # 柱状图
    x = np.arange(len(class_names))
    width = 0.2
    plt.figure(figsize=(10,5))
    plt.bar(x - width, precision, width, label='Precision', color='skyblue')
    plt.bar(x, recall, width, label='Recall', color='lightgreen')
    plt.bar(x + width, f1, width, label='F1-score', color='salmon')
    plt.xticks(x, class_names, rotation=45)
    plt.ylim(0,1)
    plt.title("各类检测指标")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PR曲线 (每类单独)
    plt.figure(figsize=(10,6))
    for i, cls in enumerate(class_names):
        plt.plot(recall[i], precision[i], label=cls)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


def predict_yolo(
    model_path,       # 模型权重路径，例如 'runs/train/weights/best.pt'
    source,           # 输入源，可为图片路径、文件夹、视频路径、URL或摄像头ID
    task='detect',  # 任务类型，'detect' 为检测，'classify' 为分类
    imgsz=224,        # 输入图片大小，检测模型可调整尺度
    conf=0.25,        # 置信度阈值，低于此值的预测会被过滤
    iou=0.45,         # NMS（非极大值抑制）IOU阈值
    device='',        # 运行设备，空字符串表示自动选择（CPU/GPU）
    save=True,        # 是否保存预测结果图片
    save_txt=False,   # 是否保存预测标签文本
    save_conf=False,  # 是否在标签中保存置信度
    save_crop=False,  # 是否保存裁剪的预测框图像
    show=False,       # 是否显示预测结果窗口
    project='runs/predict',  # 结果保存的项目路径
    name='exp',       # 本次实验/预测名称
    exist_ok=False,   # 是否允许覆盖已有同名输出目录
    line_width=None,  # 边界框线宽
    show_labels=True, # 是否在图片上显示类别标签
    show_conf=True,   # 是否在图片上显示置信度
    vid_stride=1,     # 视频帧采样步长
    stream_buffer=False, # 视频/流缓冲开关
    visualize=False,     # 是否可视化特征图
    augment=False,       # 是否使用测试时增强
    agnostic_nms=False,  # 类别无关NMS开关
    classes=None,        # 可选，过滤指定类别列表
    retina_masks=False,  # 是否使用高分辨率掩码
    embed=None,          # 是否返回特征向量
    source_gt=None       # 可选：GT标签文件夹，用于计算指标
):
    """
    使用训练好的YOLO模型进行预测
    """

    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 检查source是否存在
    source_path = Path(source)
    if source_path.exists():
        if source_path.is_file():
            print(f"预测单个文件: {source}")
        elif source_path.is_dir():
            print(f"预测文件夹: {source}")
    else:
        # 可能是URL或摄像头
        print(f"预测源: {source}")

    print(f"任务类型: {task}")
    print(f"图片大小: {imgsz}")
    print(f"置信度阈值: {conf}")
    print(f"设备: {device if device else '自动'}")

    # 预测参数
    predict_args = {
        'source': source,
        'imgsz': imgsz,
        'conf': conf,
        'iou': iou,
        'device': device,
        'save': save,
        'save_txt': save_txt,
        'save_conf': save_conf,
        'save_crop': save_crop,
        'show': show,
        'project': project,
        'name': name,
        'exist_ok': exist_ok,
        'line_width': line_width,
        'show_labels': show_labels,
        'show_conf': show_conf,
        'vid_stride': vid_stride,
        'stream_buffer': stream_buffer,
        'visualize': visualize,
        'augment': augment,
        'agnostic_nms': agnostic_nms,
        'classes': classes,
        'retina_masks': retina_masks,
        'embed': embed,
        'verbose': True,
    }

    # 执行预测
    print("\n开始预测...\n")
    results = model.predict(**predict_args)
    print("\n预测完成！")

    # 分类任务结果
    if task == 'classify':
        for i, result in enumerate(results):
            if hasattr(result, 'probs'):
                top5 = result.probs.top5
                top5_conf = result.probs.top5conf.cpu().numpy()
                names = result.names

                print(f"\n图片 {i+1} - {result.path}")
                print("Top-5 预测:")
                for idx, (cls, conf) in enumerate(zip(top5, top5_conf), 1):
                    print(f"  {idx}. {names[cls]}: {conf:.4f}")
    else:
        # 检测任务结果
        total_detections = 0
        for i, result in enumerate(results):
            if hasattr(result, 'boxes'):
                boxes = result.boxes
                num_boxes = len(boxes)
                total_detections += num_boxes

                print(f"\n图片 {i+1} - {result.path}")
                print(f"检测到 {num_boxes} 个对象")

                if num_boxes > 0:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = result.names[cls]
                        print(f"  - {name}: {conf:.4f}")

        print(f"\n总共检测到: {total_detections} 个对象")

        # 如果提供了 GT 标签，计算指标并画图
        if source_gt:
            precision, recall, f1, class_names = compute_metrics(results, gt_folder=source_gt)
            print("\n[指标] 各类平均：")
            for cls_name, p, r, f in zip(class_names, precision, recall, f1):
                print(f"{cls_name}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")
            plot_metrics(precision, recall, f1, class_names)

    if save:
        print(f"\n结果已保存到: {project}/{name}")

    return results

# ===============================================
# 主程序入口
# ===============================================
if __name__ == '__main__':
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description='YOLOv11预测脚本（带指标绘图）')

    parser.add_argument('--model', type=str, default='runs/train/crop_pests/weights/best.pt',
                        help='模型权重路径')
    parser.add_argument('--source', type=str, default='data/test_images/',
                        help='输入源: 图片/文件夹/视频路径 或 摄像头ID')
    parser.add_argument('--source-gt', type=str, default=None,
                        help='可选：真实标签文件夹，用于计算指标')
    parser.add_argument('--name', type=str, default=None,
                        help='实验名称 (默认: 自动生成 pred_YYYYMMDD_HHMMSS)')
    parser.add_argument('--exist-ok', action='store_true',
                        help='允许覆盖已有输出目录')

    args = parser.parse_args()

    if args.name is None:
        args.name = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    predict_yolo(
        model_path=args.model,
        source=args.source,
        project='runs/predict',
        name=args.name,
        exist_ok=args.exist_ok,
        source_gt=args.source_gt
    )
