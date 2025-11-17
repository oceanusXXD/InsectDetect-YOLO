"""
prepare_data.py — Crop Pests 分类数据准备（默认无需运行）
用途：
- 当且仅当你们做“整图分类”子任务时，把“按类文件夹”的数据转换为 YOLO-CLS 目录结构：
  {train,val,test}/{images,labels}，labels 中每个样本 1 行（class_id）
- AgroPest-12（检测任务）自带 train/val/test 与 bbox 标注，检测任务请直接用其 data.yaml
"""

import shutil
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split


def prepare_classification_dataset(source_dir, output_dir):
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    classes = []
    image_label_pairs = []

    if (source_path / 'train').exists():
        _process_split_dataset(source_path, output_path, classes)
    else:
        _process_class_folders(source_path, output_path, classes, image_label_pairs)

    _save_cls_yaml(output_path, classes)
    print(f"[INFO] 分类数据准备完成：{output_path} | 类别数={len(classes)}")


def _process_class_folders(source_path: Path, output_path: Path, classes, image_label_pairs):
    print("[INFO] 发现“按类文件夹结构”，开始处理...")
    for class_folder in sorted(source_path.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name
        if class_name not in classes:
            classes.append(class_name)
        class_id = classes.index(class_name)
        for img_file in class_folder.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                image_label_pairs.append((str(img_file), class_id, class_name))

    # 70/15/15
    train_val, test = train_test_split(image_label_pairs, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.176, random_state=42)  # 0.176*0.85≈0.15

    _copy_and_label_images(train, output_path / 'train', 'train')
    _copy_and_label_images(val, output_path / 'val', 'val')
    _copy_and_label_images(test, output_path / 'test', 'test')


def _process_split_dataset(source_path: Path, output_path: Path, classes):
    print("[INFO] 发现“已分好 train/val/test”结构，开始处理（整图分类标签）...")
    for split in ['train', 'val', 'test']:
        sp = source_path / split
        if not sp.exists():
            continue
        # 收集类名
        for cf in sorted(sp.iterdir()):
            if cf.is_dir():
                cname = cf.name
                if cname not in classes:
                    classes.append(cname)
        # 复制
        for cf in sp.iterdir():
            if not cf.is_dir():
                continue
            cid = classes.index(cf.name)
            for img_file in cf.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    dst = output_path / split / 'images' / img_file.name
                    shutil.copy2(img_file, dst)
                    with open(output_path / split / 'labels' / f"{img_file.stem}.txt", "w") as f:
                        f.write(f"{cid}\n")


def _copy_and_label_images(pairs, dest_path: Path, split_name: str):
    print(f"[INFO] 处理 {split_name}: {len(pairs)} 张")
    for img_path, class_id, class_name in pairs:
        p = Path(img_path)
        dst = dest_path / 'images' / f"{class_name}_{p.name}"
        shutil.copy2(p, dst)
        with open(dest_path / 'labels' / f"{class_name}_{p.stem}.txt", "w") as f:
            f.write(f"{class_id}\n")


def _save_cls_yaml(output_path: Path, classes):
    cfg = {
        'path': str(output_path.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(classes)},
        'nc': len(classes)
    }
    with open(output_path / 'data.yaml', "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"[INFO] data.yaml 写入：{output_path / 'data.yaml'}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="按类文件夹 → YOLO-CLS 目录结构（仅整图分类任务需要）")
    ap.add_argument('--source', type=str, required=True)
    ap.add_argument('--output', type=str, default='data/cls_dataset')
    args = ap.parse_args()
    prepare_classification_dataset(args.source, args.output)
