import json
import os

from paddleocr import PaddleOCR

from config import OCRConfig


def recognise_text(config: OCRConfig = OCRConfig()):
    """
    Perform OCR on preprocessed license plate images
    """
    # initialise PaddleOCR
    reader = PaddleOCR(use_angle_cls=config.USE_ANGLE_CLS, lang=config.LANG)

    # Load preprocessing results
    with open(os.path.join(config.INPUT_DIR, "preprocessing_results.json"), "r") as f:
        plates = json.load(f)

    results = []

    for plate_info in plates:
        # Perform OCR
        result = reader.ocr(plate_info["preprocessed_path"], cls=True)

        text = ""
        confidence = 0.0

        if result and result[0]:
            for line in result[0]:
                if len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]

                    if confidence > config.CONFIDENCE_THRESHOLD:
                        # Clean text (keep only alphanumeric characters)
                        text = "".join(c for c in text if c.isalnum() or c.isspace())
                        break

        # Add OCR results to plate info
        plate_info["text"] = text
        plate_info["ocr_confidence"] = confidence
        results.append(plate_info)

    # Save OCR results
    with open(config.OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    recognise_text()
