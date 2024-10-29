# Vehicle License Plate Recognition System

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

This project implements an advanced license plate detection system using YOLOv11n architecture. The system is designed
to accurately detect and localize vehicle license plates under various conditions including different angles, lighting
conditions, and distances. Our implementation achieves high precision and recall rates, making it suitable for
real-world applications.

## Methodology

### Dataset Preparation

- Total Images: 25,470
- Training Set: 24,397 images (~96%)
- Validation Set: 1,073 images (~4%)
- Single Class: License Plate
- Annotations: YOLO format (normalized coordinates)

### Model Architecture

- Base Model: YOLOv11n
- Input Resolution: 640x640
- Backbone: Modified C3K2 architecture
- Feature Pyramid Network for multi-scale detection

### Training Configuration

```python
- Epochs: 50
- Batch
Size: 16
- Optimizer: AdamW
- Initial
Learning
Rate: 0.01
- Final
Learning
Rate: 0.000109
- Cosine
LR
Schedule: Enabled
```

### Data Augmentation

- Random Horizontal Flip (50% probability)
- Scale: ±50%
- Translation: ±10%
- Auto Augment: RandAugment
- Random Erasing: 0.4
- Mosaic: Enabled

## Results

### Performance Metrics

- mAP50: 0.811 (81.1%)
- Precision: 0.841 (84.1%)
- Recall: 0.763 (76.3%)
- mAP50-95: 0.421 (42.1%)

### Training Progress

The training process demonstrated consistent improvement across 50 epochs:

- Initial mAP50: 0.379 (Epoch 1)
- Best mAP50: 0.812 (Epoch 46)
- Final Box Loss: 1.053
- Final Classification Loss: 0.405
- Final DFL Loss: 1.104

[Training Metrics Progress Graph]

### Loss Convergence

- Box Loss: Stabilized around 1.05
- Classification Loss: Steadily decreased to 0.405
- DFL Loss: Converged to 1.10

[Training Losses Graph]

## Implementation Details

### Hardware Requirements

- GPU: NVIDIA RTX 2070 (8GB)
- Training Time: ~8,955 seconds (~2.5 hours)

### Dependencies

- Python 3.8+
- PyTorch 2.0+
- Ultralytics 8.0+
- OpenCV
- numpy
- pandas

### Installation

```bash
pip install ultralytics
```

### Usage

#### For Inference

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('best.pt')

# Perform detection
results = model('image.jpg')
```

#### For Training

```python
from ultralytics import YOLO

# Load the base model
model = YOLO('training/yolo11n.pt')

# Start training
results = model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    device=0,
    workers=8,
    cos_lr=True,
    optimizer='AdamW'
)
```

## Challenges & Solutions

1. **Memory Management**
    - Challenge: Limited GPU memory (8GB)
    - Solution: Optimized batch size and cache settings

2. **Model Convergence**
    - Challenge: Initial training instability
    - Solution: Implemented cosine learning rate schedule

## Future Improvements

1. Model Enhancement
    - Integration of attention mechanisms
    - Exploration of larger model variants (YOLOv11s, YOLOv11m)

2. Performance Optimization
    - Implementation of model quantization
    - Investigation of TensorRT acceleration

## Model Weights

- Best model weights: `best.pt`
- Final model weights: `last.pt`

![Example Detection Image](https://i.imgur.com/tFn56hE.jpeg)

The system successfully achieves robust performance with good precision and recall rates, making it suitable for
real-world license plate detection applications. The balance between computational efficiency and detection accuracy
makes it particularly suitable for deployment in production environments.