# Vehicle License Plate Detection System
31256 Image Processing and Pattern Recognition (Spring 2024) - Group 12

## Team Members
| SID      | Name             |
|----------|------------------|
| 13740802 | Joonghyuk Seong  |
| 24587065 | Jing Ou          |
| 24749867 | Benjamin Balogh  |
| 14417289 | Pansilu Fernando |
| 24580312 | Jiapeng Yang     |
| 14011276 | Yixuan Li        |

## Project Overview
This project implements an advanced license plate detection system using YOLOv11n architecture. The system achieves state-of-the-art performance with an mAP50 of 0.81, making it suitable for real-world applications in various conditions.

## Key Performance Metrics
- mAP50: 0.811 (81.1%)
- Precision: 0.841 (84.1%)
- Recall: 0.763 (76.3%)
- mAP50-95: 0.421 (42.1%)

## Dataset
The project utilizes a comprehensive license plate detection dataset from Kaggle:
- **Total Images**: 27,900
  - Training Set: 25,500 images
  - Validation Set: 1,000 images
  - Test Set: 400 images
- **Source**: [Large License Plate Dataset (Kaggle)](https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset)
- **Format**: YOLO annotation format
- **License**: CC0 (Public Domain)

## Model Architecture
- Base Model: YOLOv11n (Extended YOLOv8n architecture)
- Input Resolution: 640x640
- Backbone Features:
  - C3k2 Backbone with PSA (Polarized Self-Attention)
  - SPPF (Spatial Pyramid Pooling - Fast)
  - Depthwise Separable Convolutions
  - Feature Pyramid Network with Bi-FPN structure

## Training Configuration
```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    device=0,
    workers=8,
    cos_lr=True,
    optimizer='AdamW',
    cache=True
)
```

### Hyperparameters
- Epochs: 50
- Batch Size: 16
- Optimizer: AdamW
- Learning Rate Schedule: Cosine
  - Initial LR: 0.01
  - Final LR: 0.000109

### Data Augmentation
- Random Horizontal Flip (50% probability)
- Scale: ±50%
- Translation: ±10%
- Auto Augment: RandAugment
- Random Erasing: 0.4
- Mosaic: Enabled

## Training Results
The model demonstrated consistent improvement throughout the training process:
- Initial mAP50: 0.379 (Epoch 1)
- Best mAP50: 0.812 (Epoch 46)
- Training Time: ~8,955 seconds (~2.5 hours)

### Loss Convergence
- Final Box Loss: 1.053
- Final Classification Loss: 0.405
- Final DFL Loss: 1.104

## Installation and Usage
### Dependencies
```bash
# Required packages
python>=3.8
pytorch>=2.0
ultralytics>=8.0
opencv-python
numpy
pandas
pyyaml
```

### Basic Setup
```bash
# Install the main package
pip install ultralytics
```

### Dataset Configuration
```python
# dataset.yaml structure
path: ./
train: dataset/images/train
val: dataset/images/val
test: dataset/images/test
names:
  0: license_plate
```

### Inference
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('best.pt')

# Perform detection
results = model('image.jpg')
```

## Model Weights
- Best model weights: `best.pt` (mAP50: 0.812)
- Final model weights: `last.pt` (mAP50: 0.811)

## Example Detection
![Example Detection](https://i.imgur.com/tFn56hE.jpeg)

## Training Progress Graph
```
Final Metrics (Epoch 50):
- Box Loss: 1.053
- Classification Loss: 0.405
- DFL Loss: 1.104
- Precision: 0.841
- Recall: 0.763
- mAP50: 0.811
- mAP50-95: 0.421
```

## Acknowledgments
- Dataset from Kaggle: "Large License Plate Dataset" by Fares Elmenshawi
- Implementation based on the Ultralytics YOLO framework