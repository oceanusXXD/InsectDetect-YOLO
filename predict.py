"""
predict.py — YOLOv11 推理 / 评测脚本
- 单图 / 文件夹 / 视频推理（保存渲染结果）
- --eval-data data.yaml ：在数据集上跑官方 evaluator（检测 mAP）
- --cls-model <cls.pt> ：二阶段（检→裁→分类器），统计 accuracy / precision / recall / F1 / AUC(OVR)
- 记录预测/评测耗时，写入 timings.json
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, roc_auc_score
)

from ultralytics import YOLO


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: dict):
    _ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _crop_by_boxes(img_path: str, boxes_xyxy):
    im = cv2.imread(img_path)
    crops = []
    if im is None:
        return crops
    H, W = im.shape[:2]
    for xyxy in boxes_xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        x1 = max(x1, 0); y1 = max(y1, 0); x2 = min(x2, W - 1); y2 = min(y2, H - 1)
        if x2 > x1 and y2 > y1:
            crops.append(im[y1:y2, x1:x2].copy())
    return crops


def run_predict(
    model_path: str,
    source: str,
    imgsz=896,
    conf=0.25,
    iou=0.45,
    device='',
    save=True,
    project='runs/predict',
    name=None,
    exist_ok=False,
    show_labels=True,
    show_conf=True,
    line_width=None,
    eval_data_yaml=None,   # 数据集评测（检测）
    cls_model_path=None,   # 二阶段（检→裁→分类器）
):
    if name is None:
        name = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = Path(project) / name

    print(f"[INFO] 加载检测模型: {model_path}")
    det = YOLO(model_path)

    # ---------- 普通推理 ----------
    print(f"[INFO] 开始推理: source={source}")
    t0 = time.time()
    results = det.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=save,
        project=str(project),
        name=name,
        exist_ok=exist_ok,
        show_labels=show_labels,
        show_conf=show_conf,
        line_width=line_width,
        verbose=True
    )
    pred_seconds = round(time.time() - t0, 3)
    print(f"[INFO] 推理完成，输出目录: {save_dir}")

    timings = {"predict_seconds": pred_seconds}

    # ---------- 数据集评测（检测） ----------
    if eval_data_yaml:
        print(f"[INFO] 评测数据集（官方 evaluator）: {eval_data_yaml}")
        t1 = time.time()
        m = det.val(data=eval_data_yaml, imgsz=imgsz, device=device, plots=True)
        timings["dataset_eval_seconds"] = round(time.time() - t1, 3)
        summary = dict(
            map50=float(getattr(m.box, 'map50', np.nan)),
            map50_95=float(getattr(m.box, 'map', np.nan)),
            mp=float(getattr(m.box, 'mp', np.nan)) if hasattr(m, 'box') else np.nan,
            mr=float(getattr(m.box, 'mr', np.nan)) if hasattr(m, 'box') else np.nan,
        )
        _write_json(save_dir / "dataset_eval_metrics.json", summary)
        print(f"[INFO] ✅ 数据集评测完成: {summary}")

    # ---------- 二阶段（检→裁→分类器，多分类 AUC=OvR） ----------
    if cls_model_path is not None:
        print(f"[INFO] 启用二阶段：分类器 = {cls_model_path}")
        cls_model = YOLO(cls_model_path)

        y_true = []   # 检测出的类别，作为对分类器的一致性参考（方法对比）
        y_proba = []  # 分类器输出的全类概率

        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None or len(r.boxes) == 0:
                continue
            img_path = r.path
            boxes = r.boxes.xyxy.cpu().numpy()
            det_classes = r.boxes.cls.cpu().numpy().astype(int)
            crops = _crop_by_boxes(img_path, boxes)
            if not crops:
                continue

            for (crop, det_cls) in zip(crops, det_classes):
                ok, enc = cv2.imencode(".jpg", crop)
                if not ok:
                    continue
                img_bytes = enc.tobytes()
                pred_cls = cls_model.predict(source=img_bytes, imgsz=224, verbose=False)[0]
                probs = pred_cls.probs  # [C]
                if probs is None:
                    continue
                y_true.append(det_cls)
                y_proba.append(probs.detach().cpu().numpy())

        if len(y_true) > 0:
            y_true = np.array(y_true)
            y_proba = np.vstack(y_proba)      # [N, C]
            y_pred = y_proba.argmax(axis=1)

            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            # 多分类 AUC（One-vs-Rest）
            n_classes = y_proba.shape[1]
            y_true_ovr = np.eye(n_classes)[y_true]
            try:
                auc = roc_auc_score(y_true_ovr, y_proba, average='macro', multi_class='ovr')
            except Exception:
                auc = float('nan')

            metrics = dict(
                accuracy=float(acc),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                auc=float(auc)
            )
            _write_json(save_dir / "twostage_metrics.json", metrics)
            print(f"[INFO] 二阶段分类指标: {metrics}")
        else:
            print("[WARN] 二阶段没有产生裁剪或预测，无法统计分类指标。")

    # ---------- 写入耗时 ----------
    _write_json(save_dir / "timings.json", timings)
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv11 推理 / 评测脚本")
    parser.add_argument('--model', type=str, default='runs/train/crop_pests/weights/best.pt')
    parser.add_argument('--source', type=str, default='data/test_images')
    parser.add_argument('--imgsz', type=int, default=896)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--project', type=str, default='runs/predict')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--eval-data', type=str, default=None, help='在数据集上评测（检测 mAP）')
    parser.add_argument('--cls-model', type=str, default=None, help='二阶段分类器（YOLO-CLS 权重）')
    args = parser.parse_args()

    run_predict(
        model_path=args.model,
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        eval_data_yaml=args.eval_data,
        cls_model_path=args.cls_model
    )
