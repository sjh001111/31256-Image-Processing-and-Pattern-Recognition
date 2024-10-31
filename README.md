# 🚗 Vehicle License Plate Detection System

31256 Image Processing and Pattern Recognition (Spring 2024) - Group 12

## 👥 Team Members
| SID      | Name             |
|----------|------------------|
| 13740802 | Joonghyuk Seong  |
| 24587065 | Jing Ou          |
| 24749867 | Benjamin Balogh  |
| 14417289 | Pansilu Fernando |
| 24580312 | Jiapeng Yang     |
| 14011276 | Yixuan Li        |

## 🔧 Tech Stack
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [YOLOv11](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## 📝 Project Overview
This project implements an advanced license plate detection system using YOLOv11n architecture. The system combines state-of-the-art computer vision techniques with optical character recognition to achieve robust license plate detection and recognition across various conditions.

## 🚀 Getting Started

### Prerequisites
```bash
pytorch
ultralytics
opencv-python
paddleocr
numpy
```

### Installation & Usage
```bash
python main.py
```

### Output Structure
```
result/
├── 1. detection_visualisation.jpg
├── 1. plate_0.jpg
├── 2. preprocessed_plate_0.jpg
├── 3. visualisation_[image_name]
└── result.json
```

## 📊 Dataset
- **Composition**: 27,900 annotated images
  - Training: 25,500 images (91.4%)
  - Validation: 1,000 images (3.6%)
  - Test: 1,400 images (5.0%)
- **Source**: [Large License Plate Dataset (Kaggle)](https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset)
- **License**: CC0 (Public Domain)

## 🏗️ System Architecture

### 1. Detection Stage (`plate_detector.py`)
- Initial license plate localization using YOLOv11n detector
- Image loading and processing
- Outputs cropped plate images with metadata

### 2. Preprocessing Stage (`preprocessing.py`)
- **Geometric Normalization**
  - Aspect ratio preservation
  - Target height resizing (200px)
- **Image Enhancement**
  - LAB color space transformation
  - CLAHE enhancement
  - Contrast and brightness adjustment
- **Noise Reduction & Binarization**
  - Bilateral filtering
  - Otsu's thresholding
  - Morphological operations

### 3. Recognition Stage (`number_recogniser.py`)
- Text extraction using PaddleOCR
- Confidence-based validation
- Structured output generation

## 🔬 Model Architecture
- **Base**: YOLOv11n (Extended YOLOv8n)
- **Input Resolution**: 640×640 pixels
- **Enhancements**:
  - Advanced C3k2 Backbone with PSA
  - Enhanced SPPF
  - Optimized Depthwise Separable Convolutions
  - Bi-directional Feature Pyramid Network

## 📈 Training Details
```python
model = YOLO('yolov11n.pt')
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

### Performance Metrics (Epoch 46)
- mAP50: 0.812
- Precision: 0.841
- Recall: 0.763

## 🖼️ Results

### Pipeline Process
1. **Detection Stage**
![Detection Results](https://i.imgur.com/H9CgtoX.jpeg)

2. **Preprocessing Stage**
![Preprocessing Example](https://i.imgur.com/NPqgRl0.jpeg)

3. **Final Output**
![Complete Pipeline](https://i.imgur.com/5IDGEmR.jpeg)

### Sample Results
Input Image | Detected Text | Confidence
------------|-------------|------------
![Sample](https://i.imgur.com/5IDGEmR.jpeg) | ZUM619 | Detection: 0.72, OCR: 1.00

## 🙏 Acknowledgments
- Dataset provided by Fares Elmenshawi on Kaggle
- Built using the Ultralytics YOLO framework and PaddleOCR