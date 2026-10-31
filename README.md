# RM YOLO Armor Detection

A YOLOv8 project for detecting RoboMaster armor plates across 14 classes. Includes training, inference, and dataset preparation scripts.

## Repository contents

| File | Purpose |
| --- | --- |
| `train.py` | Train and evaluate an armor detector |
| `predict.py` | Run inference on images, videos, folders, or a camera |
| `dataset.yaml` | Dataset paths and ordered class names |
| `Dataset/classes.txt` | Class list |
| `jpg2png.py` | Convert JPG/JPEG images to PNG |
| `labelme2yolo.py` | Convert LabelMe rectangle annotations to YOLO labels |
| `rename_dataset.py` | Convert and rename images with matching labels |
| `split_dataset.py` | Split image/label pairs into training and validation sets |
| `requirements.txt` | Python dependencies |

Raw datasets, archives, pretrained weights, and generated training outputs are excluded through `.gitignore`. Prepare your own dataset and train a model before running inference.

## Installation

```bash
git clone https://github.com/Pioneer271828/rm-yolo.git
cd rm-yolo
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

For GPU training, use a PyTorch installation compatible with your GPU and CUDA environment.

## Dataset preparation

Arrange your dataset as follows:

```text
Dataset/
  classes.txt
  train/
    images/
    labels/
  val/
    images/
    labels/
```

Each image should have a matching `.txt` label with the same filename stem. Each label row describes one object:

```text
<class_id> <center_x> <center_y> <width> <height>
```

Coordinates are normalized to the image dimensions, in the range [0, 1]. Class IDs follow this order:

```text
0 B1   1 B2   2 B3   3 B4   4 B5   5 BO   6 BS
7 R1   8 R2   9 R3  10 R4  11 R5  12 RO  13 RS
```

`B` and `R` identify blue and red teams. `O` denotes the outpost and `S` denotes the base. Adjust `path` in `dataset.yaml` to your dataset location; an absolute path can avoid ambiguity when running from another environment.

### Preprocessing utilities

Edit the configuration section of each utility before running it: the scripts currently contain local Windows paths. The `CLASS_MAP` in `labelme2yolo.py` is a single-class example using `enermy`; replace it with a mapping matching your annotations and `dataset.yaml` for armor detection.

Back up input data before using `jpg2png.py` or `rename_dataset.py`: their current settings can delete original images or rename files. `split_dataset.py` copies pairs into training and validation directories with a default training ratio of 0.8 and random seed 42.

## Training

Review the settings in `train.py`, then run:

```bash
python train.py
```

Current script defaults:

| Setting | Value |
| --- | --- |
| Model | `yolov8l.pt` |
| Epochs | 150 |
| Batch size | 8 |
| Image size setting | `[480, 768]` |
| Device | GPU 0 (`'0'`) |
| Checkpoint interval | Every 10 epochs |
| Output directory | `runs/train/armor_detection/` |

The script passes its image size setting directly to Ultralytics; check the runtime messages for the effective training size used by your installed version. Set `device = 'cpu'` for CPU training. Reduce the batch size or select a smaller model if GPU memory is insufficient. Missing pretrained weights are downloaded automatically by Ultralytics.

Outputs include `weights/best.pt`, `weights/last.pt`, periodic checkpoints, and training plots. No measured benchmark results or trained checkpoints are included in this repository.

## Inference

Provide the trained weights explicitly:

```bash
python predict.py --source path/to/image.jpg --model runs/train/armor_detection/weights/best.pt
python predict.py --source path/to/video.mp4 --model runs/train/armor_detection/weights/best.pt
python predict.py --source path/to/folder --model runs/train/armor_detection/weights/best.pt
python predict.py --source 0 --model runs/train/armor_detection/weights/best.pt
```

Use `--conf 0.5` to set the confidence threshold and `--device cpu` or `--device 0` to select a device. For all supported options:

```bash
python predict.py --help
```

## Usage notice

This project is intended for learning and research. No standalone license file is currently included.
