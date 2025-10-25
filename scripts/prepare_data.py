"""
Crop Pests数据集准备脚本
将Kaggle数据集转换为YOLO格式
kaggle下载的是直接处理完的, 所以不用运行
"""
import os
import shutil
from pathlib import Path
import yaml
import json
from sklearn.model_selection import train_test_split

def prepare_crop_pests_dataset(source_dir, output_dir):
    """
    准备Crop Pests数据集为YOLO格式
    
    Args:
        source_dir: Kaggle下载的数据集路径
        output_dir: 输出YOLO格式数据集的路径
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 创建YOLO数据集结构
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 收集所有类别
    classes = []
    image_label_pairs = []
    
    # 遍历source_dir，找到所有图片和对应的类别
    # 假设数据集结构是: source_dir/class_name/image.jpg
    if (source_path / 'train').exists():
        # 如果数据集已经有train/test分割
        process_split_dataset(source_path, output_path, classes)
    else:
        # 如果数据集是按类别文件夹组织的
        process_class_folders(source_path, output_path, classes, image_label_pairs)
    
    # 保存类别信息
    save_dataset_yaml(output_path, classes)
    
    print(f"数据集准备完成！")
    print(f"类别数量: {len(classes)}")
    print(f"类别列表: {classes}")

def process_class_folders(source_path, output_path, classes, image_label_pairs):
    """处理按类别文件夹组织的数据集"""
    print("检测到类别文件夹结构，开始处理...")
    
    # 遍历所有类别文件夹
    for class_folder in sorted(source_path.iterdir()):
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        if class_name not in classes:
            classes.append(class_name)
        class_id = classes.index(class_name)
        
        # 收集该类别的所有图片
        for img_file in class_folder.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_label_pairs.append((str(img_file), class_id, class_name))
    
    # 划分训练集、验证集、测试集 (70%, 15%, 15%)
    train_val, test = train_test_split(image_label_pairs, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.176, random_state=42)  # 0.176 * 0.85 ≈ 0.15
    
    # 复制文件并创建标签
    copy_and_label_images(train, output_path / 'train', 'train')
    copy_and_label_images(val, output_path / 'val', 'val')
    copy_and_label_images(test, output_path / 'test', 'test')

def process_split_dataset(source_path, output_path, classes):
    """处理已经分割好的数据集"""
    print("检测到预分割数据集结构，开始处理...")
    
    for split in ['train', 'val', 'test']:
        split_path = source_path / split
        if not split_path.exists():
            # 如果没有val或test，尝试从train分割
            if split in ['val', 'test']:
                continue
        
        # 收集类别
        for class_folder in sorted(split_path.iterdir()):
            if not class_folder.is_dir():
                continue
            class_name = class_folder.name
            if class_name not in classes:
                classes.append(class_name)
        
        # 复制图片并创建分类标签
        for class_folder in split_path.iterdir():
            if not class_folder.is_dir():
                continue
            
            class_name = class_folder.name
            class_id = classes.index(class_name)
            
            for img_file in class_folder.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # 复制图片
                    dest_img = output_path / split / 'images' / img_file.name
                    shutil.copy2(img_file, dest_img)
                    
                    # 创建分类标签（YOLO分类格式：整张图片的类别）
                    label_file = output_path / split / 'labels' / f"{img_file.stem}.txt"
                    with open(label_file, 'w') as f:
                        # 对于分类任务，标签格式为: class_id
                        f.write(f"{class_id}\n")

def copy_and_label_images(image_pairs, dest_path, split_name):
    """复制图片并创建标签文件"""
    print(f"处理 {split_name} 集: {len(image_pairs)} 张图片")
    
    for img_path, class_id, class_name in image_pairs:
        img_path = Path(img_path)
        
        # 生成唯一文件名（避免不同类别下的同名文件冲突）
        dest_img = dest_path / 'images' / f"{class_name}_{img_path.name}"
        shutil.copy2(img_path, dest_img)
        
        # 创建分类标签文件
        label_file = dest_path / 'labels' / f"{class_name}_{img_path.stem}.txt"
        with open(label_file, 'w') as f:
            f.write(f"{class_id}\n")

def save_dataset_yaml(output_path, classes):
    """保存YOLO数据集配置文件"""
    config = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(classes)},
        'nc': len(classes)  # number of classes
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n数据集配置已保存到: {yaml_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='准备Crop Pests数据集为YOLO格式')
    parser.add_argument('--source', type=str, required=True, help='Kaggle数据集路径')
    parser.add_argument('--output', type=str, default='data/crop_pests', help='输出路径')
    
    args = parser.parse_args()
    
    prepare_crop_pests_dataset(args.source, args.output)