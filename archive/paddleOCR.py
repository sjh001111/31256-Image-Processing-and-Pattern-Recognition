import cv2
from paddleocr import PaddleOCR
import os


def process_image(img_path):
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en")

    # Load and process image
    image = cv2.imread(img_path)
    result = ocr.ocr(image, cls=True)

    if result and result[0]:
        for line in result[0]:
            print(line)
            text = line[1][0]  # Get recognized text
            confidence = line[1][1]  # Get confidence score
            print(f"Text: {text}, Confidence: {confidence}")


# Test
process_image("../debug_images/plate_0_8_otsu.jpg")
