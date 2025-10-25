# 🐞 InsectDetect-YOLO (Windows 版使用说明)

基于 **YOLOv11x** 的农作物害虫检测与分类项目（Windows）

---

## 环境配置

### 创建虚拟环境

```bat
# 创建并激活虚拟环境
python -m venv venv
> - 如果使用的是 Anaconda，可以跳过此步骤，直接在 bat 文件中使用 `conda activate 环境名`。
> - 所有 `.bat` 文件默认使用 `.\venv\Scripts\activate`。
.\venv\Scripts\activate
```
##  安装依赖
# 安装项目所需依赖
```bat
pip install -r requirements.txt
```

- 如果安装报错，可尝试分步安装：
```bat
pip install ultralytics torch torchvision opencv-python pandas numpy matplotlib pyyaml tqdm tensorboard
```
---
## 🚀 二、运行顺序

### 第一步：运行 `run_train.bat`
```bat
- 作用：启动训练流程。
- 默认数据文件路径：`InsectDetect-YOLO\data\data.yaml`
- 默认模型：`yolo11x.pt`
- 默认任务：`classify`
- 模型结果保存目录：`runs\train\crop_pests_时间戳\`
```
---

### 第二步：运行 `run_predict.bat`
```bat
- 作用：使用训练好的模型进行预测。
- 默认模型路径：`runs\train\crop_pests\weights\best.pt`

- 默认输入路径：`InsectDetect-YOLO\data\test_images\`

- 默认任务：`classify`

> 📍可根据需要修改：
> - `--model` → 模型权重文件路径  
> - `--source` → 要预测的图片或文件夹路径  

预测结果会保存到：`runs\predict\`
```
---

## 🧠 三、常见修改项
```yaml
| 项目 | 修改位置 | 默认值 | 说明 |
|------|------------|--------|------|
| 任务类型 | `run_train.bat`、`run_predict.bat` | classify | 可改为 detect |
| 数据集路径 | `run_train.bat` | InsectDetect-YOLO\data\data.yaml | 指向你的 data.yaml 文件 |
| 模型路径 | `run_predict.bat` | runs\train\crop_pests\weights\best.pt | 使用训练好的模型权重 |
| 图片/视频路径 | `run_predict.bat` | InsectDetect-YOLO\data\test_images\ | 可改为你想预测的文件 |
| 虚拟环境路径 | 所有 bat 文件顶部 | .\venv\Scripts\activate | 如果你用 conda，请替换为 conda activate 环境名 |
```
---

## 🧾 四、文件说明
```yaml
| 文件名 | 功能说明 |
|--------|-----------|
| `run_train.bat` | 启动模型训练（自动带时间戳保存结果） |
| `run_predict.bat` | 启动模型预测（支持单张或整文件夹） |
| `run_tensorboard.bat` | 启动 TensorBoard 可视化界面 |
| `requirements.txt` | Python 环境依赖列表 |
| `InsectDetect-YOLO/scripts/train.py` | 实际的训练逻辑文件 |
| `InsectDetect-YOLO/scripts/predict.py` | 实际的预测逻辑文件 |
```
---

## 五、项目结构参考
```yaml
.
├── InsectDetect-YOLO/
│ ├── data/
│ │ └──test/
│ │ └──train/
│ │ └──valid/
│ │ └── data.yaml
│ └── scripts/
│ ├── train.py
│ └── predict.py
├── runs/
│ ├── train/
│ └── predict/
├── venv/
├── run_train.bat
├── run_predict.bat
├── run_tensorboard.bat
└── requirements.txt
```

---

## ✅ 六、完整运行顺序总结

1️⃣ 创建虚拟环境：`python -m venv venv`  
2️⃣ 激活虚拟环境：`.\venv\Scripts\activate`  
3️⃣ 安装依赖：`pip install -r requirements.txt`  
4️⃣ 启动训练：`run_train.bat`   
5️⃣ 执行预测：`run_predict.bat`

---

## 📄 七、附注

- 均使用bat命令，也可以自行使用py/bash命令进行运行，脚本位于scripts/
---

## 🔗 八、参考链接

- YOLOv11 官方文档：https://docs.ultralytics.com  
- Ultralytics GitHub：https://github.com/ultralytics/ultralytics  
- Kaggle 害虫数据集：https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset

---

## 📘 九、许可声明
仅用于unsw COMP9517 project