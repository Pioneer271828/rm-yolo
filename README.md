# 装甲板识别项目 (RM Armor Detection)

基于YOLOv8的RM（RoboMaster）装甲板识别系统，支持14个类别的装甲板检测。

## 项目结构

```
F:\RM-Demo\b\
├── Dataset/              # 数据集目录
│   ├── train/           # 训练集
│   │   ├── images/      # 训练图片 (6809张)
│   │   └── labels/      # 训练标签 (YOLO格式)
│   ├── val/             # 验证集
│   │   ├── images/      # 验证图片 (1742张)
│   │   └── labels/      # 验证标签
│   └── classes.txt      # 类别名称列表
├── dataset.yaml         # 数据集配置文件
├── train.py             # 训练脚本
├── predict.py           # 推理/预测脚本
└── requirements.txt     # 依赖包列表
```

## 类别说明

共14个类别：

- **蓝方装甲板**: B1, B2, B3, B4, B5
- **蓝方特殊**: BO (前哨站), BS (基地)
- **红方装甲板**: R1, R2, R3, R4, R5
- **红方特殊**: RO (前哨站), RS (基地)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练参数可在 `train.py` 中修改：
- `model_type`: 模型类型 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- `epochs`: 训练轮数 (默认100)
- `batch_size`: 批次大小 (默认16)
- `imgsz`: 输入图像尺寸 (默认640)
- `save_period`: 保存间隔 (默认10，即每10轮保存一次)

训练完成后，最佳模型保存在：`runs/train/armor_detection/weights/best.pt`

### 2. 推理/预测

#### 检测单张图像
```bash
python predict.py --source path/to/image.jpg
```

#### 检测视频
```bash
python predict.py --source path/to/video.mp4
```

#### 批量检测文件夹
```bash
python predict.py --source path/to/folder
```

#### 使用摄像头实时检测
```bash
python predict.py --source 0
```

#### 其他参数
```bash
python predict.py --source path/to/image.jpg --model runs/train/armor_detection/weights/best.pt --conf 0.5 --device 0
```

参数说明：
- `--model`: 模型权重文件路径
- `--source`: 输入源（图像/视频/文件夹/摄像头ID）
- `--conf`: 置信度阈值（默认0.5）
- `--device`: 设备（0: GPU 0, cpu: CPU）

## 自定义配置

### 修改训练参数

编辑 `train.py` 文件中的以下参数：

```python
# 模型配置
model_type = 'yolov8n.pt'  # 选择模型大小

# 训练参数
epochs = 100              # 训练轮数
batch_size = 16           # 批次大小
imgsz = 640               # 图像尺寸

# 保存设置
save_period = 10          # 每10轮保存一次
```

### 修改数据集路径

编辑 `dataset.yaml` 文件：

```yaml
path: F:/RM-Demo/b/Dataset  # 修改为你的数据集路径
train: train/images
val: val/images
```

## 训练结果

训练完成后，结果保存在 `runs/train/armor_detection/` 目录：

- `weights/best.pt`: 最佳模型权重
- `weights/last.pt`: 最后一个epoch的模型权重
- `weights/epochX.pt`: 每隔10轮保存的检查点
- `results.png`: 训练曲线图
- `confusion_matrix.png`: 混淆矩阵
- `PR_curve.png`: 精确率-召回率曲线
- `F1_curve.png`: F1分数曲线

## 数据集格式

YOLO格式标签文件（每行一个目标）：
```
<class_id> <center_x> <center_y> <width> <height>
```

所有坐标都是归一化的（0-1之间）。

示例：
```
0 0.72929484375 0.3436165364583334 0.16027906249999999 0.22443151041666667
0 0.45854171875 0.17132955729166666 0.049189062500000005 0.18874921874999998
```

## 常见问题

### 1. 显存不足
减小 `batch_size` 或使用更小的模型（如 `yolov8n.pt`）

### 2. 训练速度慢
- 使用GPU训练
- 减小 `imgsz`（图像尺寸）
- 使用更小的模型（如 `yolov8n.pt`）

### 3. 检测精度低
- 增加训练轮数
- 使用更大的模型（如 `yolov8m.pt`）
- 调整数据增强参数
- 增加训练数据

## 性能指标

在验证集上的性能：
- mAP50: 目标 > 0.85
- mAP50-95: 目标 > 0.70
- Precision: 目标 > 0.85
- Recall: 目标 > 0.80

## 注意事项

1. 确保数据集路径正确
2. 训练前检查GPU是否可用
3. 根据硬件配置调整batch_size
4. 建议使用GPU进行训练以获得更快的速度
5. 每10轮会自动保存一次模型检查点

## 作者

装甲板识别项目团队

## 许可证

本项目仅供学习和研究使用。

## GitHub 仓库文件范围

仓库包含训练、推理、数据预处理脚本及类别配置。原始数据（Dataset、new、picture）、压缩包、模型权重和 runs 训练输出未上传。

请自行将训练集与验证集放入 Dataset/train 与 Dataset/val，目录格式见上文。训练脚本使用的 yolov8l.pt 在本地缺失时会由 Ultralytics 自动下载。推理前需先训练模型或自行准备相应权重。

当前 train.py 实际默认配置为 yolov8l.pt、150 轮、batch=8、imgsz=[480, 768]、GPU 0；以上文档中的旧默认值以脚本为准。数据预处理脚本包含本机路径，使用前请修改配置区；labelme2yolo.py 的 CLASS_MAP 目前为 enermy 单类别示例，处理装甲板数据时需按 dataset.yaml 配置映射。jpg2png.py 与 rename_dataset.py 可能删除或重命名输入文件，使用前请备份数据。
