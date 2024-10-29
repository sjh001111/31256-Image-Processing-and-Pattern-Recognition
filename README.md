# 31256 Image Processing and Pattern Recognition - Spring 2024
## Assignment Task 2: Vehicle License Plate Recognition System Implementation

## Group 12 Members
| SID | Name |
|-----|------|
| 13740802 | Joonghyuk Seong |
| 24587065 | Jing Ou |
| 24749867 | Benjamin Balogh |
| 14417289 | Pansilu Fernando |
| 24580312 | Jiapeng Yang |
| 14011276 | Yixuan Li |

## Project Overview
This repository contains an implementation of an Automatic License Plate Recognition (ALPR) system that addresses the challenges of license plate detection across various vehicle models. The system utilizes image processing techniques, machine learning, and deep learning approaches.

## Technical Implementation

### Pre-processing
- Image resizing
- Grayscale conversion
- Noise reduction
- Contrast enhancement

### Main Techniques
1. Traditional Image Processing
   - OCR
   - Template Matching

2. Machine Learning
   - Support Vector Machines (SVM)
   - k-Nearest Neighbors (k-NN)

3. Deep Learning
   - Convolutional Neural Networks (CNN)
   - Recurrent Neural Networks (RNN) with LSTM

## Installation Requirements

### Required Libraries
```bash
pip install opencv-python
pip install numpy
pip install torch
pip install tensorflow
pip install pytesseract
pip install Pillow
```

### Tesseract OCR Installation
#### Windows
1. Download and install Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Add Tesseract path to environment variables

#### Linux
```bash
sudo apt-get install tesseract-ocr
```

## Performance Metrics
- Detection Accuracy: Target > 90%
- Precision: Target > 80%
- Recall: Target > 75%
- Recognition Accuracy: Target > 95%
- Processing Time: < 1 second per image

---
© 2024 31256 IPPR Group 12
