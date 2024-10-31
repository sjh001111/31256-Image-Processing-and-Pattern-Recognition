# config.py
class LicensePlateConfig:
    # Image preprocessing
    TARGET_HEIGHT = 200
    BILATERAL_D = 11
    BILATERAL_SIGMA_COLOR = 17
    BILATERAL_SIGMA_SPACE = 17

    # Thresholding parameters
    ADAPTIVE_BLOCK_SIZE = 19
    ADAPTIVE_C = 9
    ADAPTIVE2_BLOCK_SIZE = 23
    ADAPTIVE2_C = 10

    # OCR parameters
    MIN_CONFIDENCE = 0.3
    ALLOW_LIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # Additional OCR parameters
    HEIGHT_THRESHOLD = 0.3
    WIDTH_THRESHOLD = 0.3
    PARAGRAPH = False
    MIN_SIZE = 10
    CONTRAST_THRESHOLD = 0.3

    # Text cleaning
    MIN_TEXT_LENGTH = 2
    MAX_TEXT_LENGTH = 10
