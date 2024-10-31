import os
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from config import LicensePlateConfig as cfg


class LicensePlateDetector:
    def __init__(self, model_path="best.pt", debug_mode=False):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(["en"])
        self.debug_mode = debug_mode

        if self.debug_mode:
            self.debug_dir = "debug_images"
            os.makedirs(self.debug_dir, exist_ok=True)

    def save_debug_image(self, image, stage_name, plate_index):
        if self.debug_mode:
            filename = f"{self.debug_dir}/plate_{plate_index}_{stage_name}.jpg"
            cv2.imwrite(filename, image)
            print(f"Saved {stage_name} image to {filename}")

    def enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    def preprocess_plate(self, plate_img, plate_index):
        """Enhanced preprocessing pipeline"""
        # Save original crop
        self.save_debug_image(plate_img, "1_original_crop", plate_index)

        # Resize while maintaining aspect ratio
        aspect_ratio = plate_img.shape[1] / plate_img.shape[0]
        target_width = int(cfg.TARGET_HEIGHT * aspect_ratio)
        plate_img = cv2.resize(plate_img, (target_width, cfg.TARGET_HEIGHT))
        self.save_debug_image(plate_img, "2_resized", plate_index)

        # Enhance contrast
        enhanced = self.enhance_contrast(plate_img)
        self.save_debug_image(enhanced, "3_enhanced", plate_index)

        # Convert to grayscale
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        self.save_debug_image(gray, "4_grayscale", plate_index)

        # Apply bilateral filter
        denoised = cv2.bilateralFilter(
            gray, cfg.BILATERAL_D, cfg.BILATERAL_SIGMA_COLOR, cfg.BILATERAL_SIGMA_SPACE
        )
        self.save_debug_image(denoised, "5_denoised", plate_index)

        preprocessed_images = []

        # Multiple threshold attempts
        # 1. Gaussian adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,  # Not inverting for white text on black background
            cfg.ADAPTIVE_BLOCK_SIZE,
            cfg.ADAPTIVE_C,
        )
        preprocessed_images.append(adaptive)
        self.save_debug_image(adaptive, "6a_adaptive", plate_index)

        # 2. Inverted adaptive threshold for dark text
        adaptive_inv = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Inverting for black text on white background
            cfg.ADAPTIVE_BLOCK_SIZE,
            cfg.ADAPTIVE_C,
        )
        preprocessed_images.append(adaptive_inv)
        self.save_debug_image(adaptive_inv, "6b_adaptive_inv", plate_index)

        # Also add grayscale for direct OCR
        preprocessed_images.append(denoised)

        return preprocessed_images

    def recognize_plate(self, plate_img, plate_index):
        """Improved OCR with multiple attempts"""
        preprocessed_images = self.preprocess_plate(plate_img, plate_index)
        all_results = []

        for idx, processed_img in enumerate(preprocessed_images):
            results = self.reader.readtext(
                processed_img,
                allowlist=cfg.ALLOW_LIST,
                detail=1,
                paragraph=cfg.PARAGRAPH,
                height_ths=cfg.HEIGHT_THRESHOLD,
                width_ths=cfg.WIDTH_THRESHOLD,
                contrast_ths=cfg.CONTRAST_THRESHOLD,
            )

            for bbox, text, conf in results:
                cleaned_text = "".join(c for c in text if c.isalnum())
                if (
                    len(cleaned_text) >= cfg.MIN_TEXT_LENGTH
                    and len(cleaned_text) <= cfg.MAX_TEXT_LENGTH
                    and conf > cfg.MIN_CONFIDENCE
                ):
                    all_results.append((cleaned_text, conf))

        if not all_results:
            return ""

        # Sort by confidence and select best result
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[0][0]

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
        detector = LicensePlateDetector(debug_mode=True)
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
