# main.py
import subprocess
import sys
from config import OCRConfig
import json
import os
import random
import cv2
import numpy as np


def get_random_test_image():
    """
    Get a random image from the test dataset
    """
    test_dir = "training/dataset/images/test"
    image_files = [
        f for f in os.listdir(test_dir) if f.endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        raise ValueError(f"No image files found in {test_dir}")

    random_image = random.choice(image_files)
    return os.path.join(test_dir, random_image), random_image


def create_result_visualisation(original_image_path, plate_info):
    """
    Create a visualisation combining original image, detected plate, and results
    """
    # Read original image
    original = cv2.imread(original_image_path)
    h, w = original.shape[:2]

    # Read detected plate image
    plate_img = cv2.imread(plate_info["plate_path"])
    preprocess_img = cv2.imread(plate_info["preprocessed_path"])

    # Create a white background for results
    result_height = max(h, 200)  # At least 200 pixels for text
    result = np.full((result_height, w * 2, 3), 255, dtype=np.uint8)

    # Draw original image with detection box
    result[0:h, 0:w] = original
    x1, y1, x2, y2 = plate_info["bbox"]
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Prepare right side of visualisation
    right_start = w
    margin = 20

    # Draw detected plate
    if plate_img is not None:
        plate_h, plate_w = plate_img.shape[:2]
        max_plate_width = w - margin * 2
        scale = min(1.0, max_plate_width / plate_w)
        new_size = (int(plate_w * scale), int(plate_h * scale))
        plate_img = cv2.resize(plate_img, new_size)
        ph, pw = plate_img.shape[:2]

        y_offset = margin
        x_offset = right_start + margin
        result[y_offset : y_offset + ph, x_offset : x_offset + pw] = plate_img

        # Draw preprocessed image below
        if preprocess_img is not None:
            preprocess_img = cv2.resize(preprocess_img, new_size)
            y_offset = margin * 2 + ph
            result[y_offset : y_offset + ph, x_offset : x_offset + pw] = preprocess_img

    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_x = right_start + margin
    text_y = margin * 3 + ph * 2 if preprocess_img is not None else margin * 3 + ph

    cv2.putText(
        result,
        f"Detected Text: {plate_info['text']}",
        (text_x, text_y),
        font,
        0.7,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        result,
        f"Detection Confidence: {plate_info['confidence']:.2f}",
        (text_x, text_y + 30),
        font,
        0.7,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        result,
        f"OCR Confidence: {plate_info['ocr_confidence']:.2f}",
        (text_x, text_y + 60),
        font,
        0.7,
        (0, 0, 0),
        2,
    )

    return result


def main():
    try:
        # Get random test image
        image_path, image_name = get_random_test_image()
        print(f"Testing with image: {image_name}")

        # Step 1: Detect license plates
        print("Step 1: Detecting license plates...")
        from plate_detector import detect_plates

        plates = detect_plates(image_path)
        print(f"Found {len(plates)} license plates")

        # Step 2: Preprocess images
        print("\nStep 2: Preprocessing detected plates...")
        from preprocessing import preprocess_plates

        processed_plates = preprocess_plates()
        print(f"Preprocessed {len(processed_plates)} plates")

        # Step 3: Perform OCR
        print("\nStep 3: Performing OCR...")
        process = subprocess.run(
            [sys.executable, "number_recogniser.py", image_path],
            check=True,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            print("Error in detection step:")
            print(process.stderr)
            return False
        print("Detection completed")

        # Load results and create visualisation
        with open(OCRConfig.OUTPUT_FILE, "r") as f:
            results = json.load(f)

        for plate_info in results:
            # Create visualisation
            vis_image = create_result_visualisation(image_path, plate_info)

            # Save visualisation
            output_path = f"results/visualisation_{image_name}"
            os.makedirs("results", exist_ok=True)
            cv2.imwrite(output_path, vis_image)

            print(f"\nResults for {image_name}:")
            print(f"Detected Text: {plate_info['text']}")
            print(f"Detection Confidence: {plate_info['confidence']:.2f}")
            print(f"OCR Confidence: {plate_info['ocr_confidence']:.2f}")
            print(f"visualisation saved as: {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
