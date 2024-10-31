import json
import os

import cv2
import numpy as np

from config import PreprocessConfig


def enhance_image(image, config: PreprocessConfig):
    """
    Enhance image quality using CLAHE and contrast adjustment
    """
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=config.CONTRAST_LIMIT, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        enhanced = cv2.convertScaleAbs(
            enhanced,
            alpha=config.CONTRAST_ALPHA,
            beta=config.BRIGHTNESS_BETA,
        )

        return enhanced
    return image


def remove_noise(image, config: PreprocessConfig):
    """
    Remove noise while preserving edges using bilateral filter
    """
    return cv2.bilateralFilter(
        image,
        config.BILATERAL_D,
        config.BILATERAL_SIGMA_COLOR,
        config.BILATERAL_SIGMA_SPACE,
    )


def clean_image(binary_image, config: PreprocessConfig):
    """
    Clean binary image using morphological operations
    """
    kernel = np.ones(config.MORPH_KERNEL_SIZE, np.uint8)

    cleaned = cv2.morphologyEx(
        binary_image,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=config.MORPH_ITERATIONS,
    )

    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_OPEN, kernel, iterations=config.MORPH_ITERATIONS
    )

    return cleaned


def preprocess_plates(config: PreprocessConfig = PreprocessConfig()):
    """
    Preprocess detected license plate images
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    processed_plates = []

    # Load detection results
    with open(os.path.join(config.INPUT_DIR, "1. detection_results.json"), "r") as f:
        plates = json.load(f)

    for i, plate_info in enumerate(plates):
        # Load plate image
        image = cv2.imread(plate_info["plate_path"])
        if image is None:
            continue

        # Resize
        aspect_ratio = image.shape[1] / image.shape[0]
        target_width = int(config.TARGET_HEIGHT * aspect_ratio)
        resized = cv2.resize(image, (target_width, config.TARGET_HEIGHT))

        # Process
        enhanced = enhance_image(resized, config)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        denoised = remove_noise(gray, config)
        blur = cv2.GaussianBlur(denoised, config.OTSU_GAUSSIAN_BLUR, 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cleaned = clean_image(binary, config)

        # Save preprocessed image
        output_path = os.path.join(config.OUTPUT_DIR, f"2. preprocessed_plate_{i}.jpg")
        cv2.imwrite(output_path, cleaned)

        # Update plate info
        plate_info["preprocessed_path"] = output_path
        processed_plates.append(plate_info)

    # Save preprocessing results
    with open(os.path.join(config.OUTPUT_DIR, "2. preprocessing_results.json"), "w") as f:
        json.dump(processed_plates, f, indent=4)

    return processed_plates
