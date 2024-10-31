from dataclasses import dataclass
from typing import Tuple


@dataclass
class DetectorConfig:
    # Model settings
    MODEL_PATH: str = "best.pt"
    OUTPUT_DIR: str = "result"
    CONFIDENCE_THRESHOLD: float = 0.5


@dataclass
class PreprocessConfig:
    # Image preprocessing
    TARGET_HEIGHT: int = 200
    CONTRAST_LIMIT: float = 3.0
    BILATERAL_D: int = 11
    BILATERAL_SIGMA_COLOR: int = 17
    BILATERAL_SIGMA_SPACE: int = 17

    # Enhancement parameters
    CONTRAST_ALPHA: float = 1.3
    BRIGHTNESS_BETA: int = 5

    # Thresholding
    ADAPTIVE_BLOCK_SIZE: int = 19
    ADAPTIVE_C: int = 9
    OTSU_GAUSSIAN_BLUR: Tuple[int, int] = (5, 5)

    # Morphological operations
    MORPH_KERNEL_SIZE: Tuple[int, int] = (3, 3)
    MORPH_ITERATIONS: int = 1

    # Directory settings
    INPUT_DIR: str = "result"
    OUTPUT_DIR: str = "result"


@dataclass
class OCRConfig:
    # PaddleOCR settings
    USE_ANGLE_CLS: bool = True
    LANG: str = "en"
    CONFIDENCE_THRESHOLD: float = 0.3

    # Directory settings
    INPUT_DIR: str = "result"
    OUTPUT_FILE: str = "recognition_results.json"
