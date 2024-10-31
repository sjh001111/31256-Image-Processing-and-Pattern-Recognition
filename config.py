# config.py
from dataclasses import dataclass
from typing import Tuple


@dataclass
class OCRConfig:
    # Image preprocessing
    TARGET_HEIGHT: int = 200
    CONTRAST_LIMIT: float = 3.0
    BILATERAL_D: int = 11
    BILATERAL_SIGMA_COLOR: int = 17
    BILATERAL_SIGMA_SPACE: int = 17

    # Enhancement parameters
    CONTRAST_ALPHA: float = 1.3  # Contrast control
    BRIGHTNESS_BETA: int = 5  # Brightness control

    # Thresholding
    ADAPTIVE_BLOCK_SIZE: int = 19
    ADAPTIVE_C: int = 9
    OTSU_GAUSSIAN_BLUR: Tuple[int, int] = (5, 5)

    # OCR parameters
    CONFIDENCE_THRESHOLD: float = 0.3

    # Morphological operations
    MORPH_KERNEL_SIZE: Tuple[int, int] = (3, 3)
    MORPH_ITERATIONS: int = 1


@dataclass
class DetectorConfig:
    # Model settings
    MODEL_PATH: str = "best.pt"
    DEBUG_MODE: bool = True

    # PaddleOCR settings
    USE_ANGLE_CLS: bool = True
    LANG: str = "en"

    # Debug settings
    DEBUG_DIR: str = "debug_images"
