# plate_detector.py
import os

import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

from config import OCRConfig, DetectorConfig


class LicensePlateDetector:
    def __init__(self, ocr_config: OCRConfig = None, detector_config: DetectorConfig = None):
        # Initialize configurations
        self.ocr_config = ocr_config or OCRConfig()
        self.detector_config = detector_config or DetectorConfig()

        # Initialize models
        self.model = YOLO(self.detector_config.MODEL_PATH)
        self.reader = PaddleOCR(
            use_angle_cls=self.detector_config.USE_ANGLE_CLS,
            lang=self.detector_config.LANG
        )

        if self.detector_config.DEBUG_MODE:
            # Remove existing debug directory and create new one
            if os.path.exists(self.detector_config.DEBUG_DIR):
                for file in os.listdir(self.detector_config.DEBUG_DIR):
                    os.remove(os.path.join(self.detector_config.DEBUG_DIR, file))
            os.makedirs(self.detector_config.DEBUG_DIR, exist_ok=True)

    def save_debug_image(self, image, stage_name, plate_index):
        """Save intermediate images for debugging"""
        if self.detector_config.DEBUG_MODE:
            filename = (
                f"{self.detector_config.DEBUG_DIR}/plate_{plate_index}_{stage_name}.jpg"
            )
            cv2.imwrite(filename, image)
            print(f"Saved {stage_name} image to {filename}")
            return filename
        return None

    def enhance_image(self, image):
        """Enhance image quality"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.ocr_config.CONTRAST_LIMIT, tileGridSize=(8, 8)
            )
            cl = clahe.apply(l)

            # Merge channels and convert back
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # Additional contrast enhancement
            enhanced = cv2.convertScaleAbs(
                enhanced,
                alpha=self.ocr_config.CONTRAST_ALPHA,
                beta=self.ocr_config.BRIGHTNESS_BETA,
            )

            return enhanced
        return image

    def remove_noise(self, image):
        """Remove noise while preserving edges"""
        return cv2.bilateralFilter(
            image,
            self.ocr_config.BILATERAL_D,
            self.ocr_config.BILATERAL_SIGMA_COLOR,
            self.ocr_config.BILATERAL_SIGMA_SPACE,
        )

    def clean_image(self, binary_image):
        """Clean binary image using morphological operations"""
        kernel = np.ones(self.ocr_config.MORPH_KERNEL_SIZE, np.uint8)

        # Remove noise
        cleaned = cv2.morphologyEx(
            binary_image,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.ocr_config.MORPH_ITERATIONS,
        )

        # Fill small holes
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_OPEN, kernel, iterations=self.ocr_config.MORPH_ITERATIONS
        )

        return cleaned

    def preprocess_plate(self, plate_img, plate_index):
        """Preprocess license plate image"""
        # Original crop
        orig_path = self.save_debug_image(plate_img, "1_original", plate_index)

        # Resize
        aspect_ratio = plate_img.shape[1] / plate_img.shape[0]
        target_width = int(self.ocr_config.TARGET_HEIGHT * aspect_ratio)
        resized = cv2.resize(plate_img, (target_width, self.ocr_config.TARGET_HEIGHT))
        resized_path = self.save_debug_image(resized, "2_resized", plate_index)

        # Enhance
        enhanced = self.enhance_image(resized)
        enhanced_path = self.save_debug_image(enhanced, "3_enhanced", plate_index)

        # Convert to grayscale
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray_path = self.save_debug_image(gray, "4_gray", plate_index)

        # Remove noise
        denoised = self.remove_noise(gray)
        denoised_path = self.save_debug_image(denoised, "5_denoised", plate_index)

        # Apply Otsu's thresholding
        blur = cv2.GaussianBlur(denoised, self.ocr_config.OTSU_GAUSSIAN_BLUR, 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_cleaned = self.clean_image(otsu)
        otsu_path = self.save_debug_image(otsu_cleaned, "6_otsu", plate_index)

        return [
            orig_path,
            resized_path,
            enhanced_path,
            gray_path,
            denoised_path,
            otsu_path,
        ]

    def recognize_plate(self, plate_img, plate_index):
        """Perform OCR using PaddleOCR"""
        image_paths = self.preprocess_plate(plate_img, plate_index)
        all_results = []

        for img_path in image_paths:
            if img_path:  # Check if debug mode is on and path exists
                result = self.reader.ocr(img_path, cls=True)

                if result and result[0]:
                    for line in result[0]:
                        if len(line) >= 2:
                            text = line[1][0]
                            confidence = line[1][1]

                            # Basic cleaning
                            cleaned_text = ''.join(c for c in text if c.isalnum() or c.isspace())

                            if confidence > self.ocr_config.CONFIDENCE_THRESHOLD:
                                all_results.append((cleaned_text, confidence))

        if all_results:
            # Sort by confidence score
            all_results.sort(key=lambda x: x[1], reverse=True)
            return all_results[0][0]

        return ""

    def detect_and_recognize(self, image_path, save_results=True):
        """Main detection and recognition function"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")

        self.save_debug_image(image, "0_original_input", 0)
        visual_output = image.copy()

        results = self.model(image)[0]
        detected_plates = []

        for i, result in enumerate(results.boxes.data.tolist()):
            x1, y1, x2, y2, confidence, class_id = result
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            plate_img = image[y1:y2, x1:x2]
            plate_text = self.recognize_plate(plate_img, i)

            detected_plates.append(
                {"bbox": (x1, y1, x2, y2), "confidence": confidence, "text": plate_text}
            )

            # Visualization
            cv2.rectangle(visual_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{plate_text} ({confidence:.2f})"
            cv2.putText(
                visual_output,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        if save_results:
            output_path = f"detected_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, visual_output)
            print(f"Final results saved to {output_path}")

        return detected_plates, visual_output


def main():
    image_path = "test.jpg"

    try:
        detector = LicensePlateDetector()
        detected_plates, _ = detector.detect_and_recognize(image_path)

        print("\nDetected License Plates:")
        for i, plate in enumerate(detected_plates, 1):
            print(f"\nPlate {i}:")
            print(f"Text: {plate['text']}")
            print(f"Confidence: {plate['confidence']:.2f}")
            print(f"Location: {plate['bbox']}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
