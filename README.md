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
This project implements an advanced licence plate detection system using YOLOv11n architecture. The system combines state-of-the-art computer vision techniques with optical character recognition to achieve robust licence plate detection and recognition across various conditions.

## System Architecture

### Implementation Overview
The system utilises a three-stage pipeline architecture:
1. **Detection Stage**: Initial licence plate localisation
2. **Preprocessing Stage**: Image enhancement and optimisation
3. **Recognition Stage**: Text extraction and validation

### Core Components
```
Input Image → YOLO Detection → Image Preprocessing → OCR Recognition → Visualisation
```

### Module Structure
1. **Detection Module** (`plate_detector.py`)
   - Implements YOLOv11n for plate localisation
   - Handles image loading and initial processing
   - Outputs cropped plate images with metadata
   ```python
   # Example detection output
   {
       "plate_path": "detected_plates/plate_0.jpg",
       "bbox": [77, 417, 145, 455],
       "confidence": 0.72
   }
   ```

2. **Preprocessing Module** (`preprocessing.py`)
   - Implements advanced image enhancement algorithms
   - Performs noise reduction and image normalisation
   - Optimises images for character recognition
   ```python
   # Key preprocessing steps
   - Contrast enhancement (CLAHE)
   - Bilateral filtering
   - Binarisation
   - Morphological operations
   ```

3. **Recognition Module** (`number_recogniser.py`)
   - Utilises PaddleOCR for text extraction
   - Implements confidence-based validation
   - Produces structured recognition results

## Implementation Details

### Detection Stage
![Detection Results](https://i.imgur.com/H9CgtoX.jpeg)

- YOLOv11n detector identifies licence plate regions
- Achieves 0.72 confidence on challenging cases
- Handles various plate orientations and lighting conditions

### Preprocessing Pipeline
![Preprocessing Example](https://i.imgur.com/NPqgRl0.jpeg)

The system employs a multi-stage preprocessing pipeline:
1. **Enhancement Phase**
   - Adaptive contrast enhancement
   - Noise suppression while preserving edges
   - Geometric normalisation

2. **Optimisation Phase**
   - Binary image generation
   - Connected component analysis
   - Morphological refinement

### Recognition Results
<!-- ![Recognition Output](https://i.imgur.com/5IDGEmR.jpeg) -->
The final stage produces structured output including:
- Extracted licence plate text
- Detection confidence metrics
- OCR confidence scores

### Visualisation
![Complete Pipeline](https://i.imgur.com/5IDGEmR.jpeg)
The system provides comprehensive visualisation showing:
- Original image with detection overlay
- Processed plate image
- Binary preprocessed image
- Recognised text and confidence metrics

## Configuration Parameters
The system utilises three configuration classes:

### Detection Configuration
```python
@dataclass
class DetectorConfig:
    MODEL_PATH: str = "best.pt"
    OUTPUT_DIR: str = "result"
    CONFIDENCE_THRESHOLD: float = 0.5
```

### Preprocessing Configuration
```python
@dataclass
class PreprocessConfig:
    TARGET_HEIGHT: int = 200
    CONTRAST_LIMIT: float = 3.0
    BILATERAL_D: int = 11
    MORPH_ITERATIONS: int = 1
```

### Recognition Configuration
```python
@dataclass
class OCRConfig:
    USE_ANGLE_CLS: bool = True
    LANG: str = "en"
    CONFIDENCE_THRESHOLD: float = 0.3
```

## Performance Evaluation

### Sample Results
Our system demonstrates robust performance across various conditions:

Input Image | Detected Text | Confidence
------------|--------------|------------
![Sample](https://i.imgur.com/5IDGEmR.jpeg) | ZUM-619 | Detection: 0.72, OCR: 1.00

### Key Metrics
- Detection Confidence: 0.72 (72%)
- OCR Accuracy: 1.00 (100%)
- End-to-end Processing Time: ~1.2 seconds

## Usage Instructions

### Running the System
```bash
python main.py
```
The system will:
1. Randomly select a test image
2. Process through the detection pipeline
3. Generate visualisation results
4. Save all outputs to the result directory

### Output Structure
```
result/
├── 1. detection_visualisation.jpg
├── 1. plate_0.jpg
├── 2. preprocessed_plate_0.jpg
├── 3. visualisation_[image_name]
└── result.json
```

## Model Training

### Key Performance Metrics
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