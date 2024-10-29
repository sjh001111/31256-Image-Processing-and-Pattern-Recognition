# 31256 Image Processing and Pattern Recognition - Spring 2024

## Assignment Task 2: Vehicle License Plate Recognition System Implementation

## Group 12 Members

| SID      | Name             |
|----------|------------------|
| 13740802 | Joonghyuk Seong  |
| 24587065 | Jing Ou          |
| 24749867 | Benjamin Balogh  |
| 14417289 | Pansilu Fernando |
| 24580312 | Jiapeng Yang     |
| 14011276 | Yixuan Li        |

## Project Overview
This project implements a deep learning model for license plate detection using YOLOv11n. The model is designed to detect and localize license plates in various conditions with high accuracy.

## Technical Details

### Model Architecture
- Base Model: YOLOv11n
- Input Resolution: 640x640
- Backbone: modified C3K2 architecture

### Dataset
- Training Images: 25,470
- Validation Images: 1,073
- Single Class: License Plate
- Data Split: ~96% train, ~4% validation

### Training Configuration
- Epochs: 50
- Batch Size: 16
- Optimizer: AdamW
- Initial Learning Rate: 0.01
- Final Learning Rate: 0.000109
- Cosine LR Schedule: Enabled
- Data Augmentation:
  - Random Flip (50% probability)
  - Scale: ±50%
  - Translation: ±10%
  - Auto Augment: RandAugment
  - Erasing: 0.4
  - Mosaic: Enabled

### Hardware
- GPU: NVIDIA RTX 2070 (8GB)
- Training Time: ~8,955 seconds (~2.5 hours)

## Performance Metrics

### Final Model Performance
- mAP50: 0.811 (81.1%)
- Precision: 0.841 (84.1%)
- Recall: 0.763 (76.3%)
- mAP50-95: 0.421 (42.1%)

### Training Progress
- Best mAP50: 0.812 (Epoch 46)
- Final Box Loss: 1.053
- Final Classification Loss: 0.405
- Final DFL Loss: 1.104

## Usage

### Installation
```bash
pip install ultralytics
```

### Inference
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('best.pt')

# Perform detection
results = model('image.jpg')
```

### Training
```python
from ultralytics import YOLO

# Load the base model
model = YOLO('yolo11n.pt')

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

## Dependencies
- Python 3.8+
- PyTorch 2.0+
- Ultralytics 8.0+
- OpenCV
- numpy
- pandas

## Model Weights
- Best model weights: `best.pt`
- Final model weights: `last.pt`

---
The model achieves robust performance with good precision and recall, making it suitable for real-world license plate detection applications. The balance between computational efficiency and detection accuracy makes it particularly suitable for deployment in production environments.