import json
import os

import cv2
from ultralytics import YOLO

from config import DetectorConfig


def detect_plates(image_path, config: DetectorConfig = DetectorConfig()):
    """
    Detect license plates in the image using YOLO model
    """
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Load YOLO model
    model = YOLO(config.MODEL_PATH)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Detect license plates
    results = model(image)[0]
    plates = []

    # Save original image with detections
    original_with_boxes = image.copy()

    for i, result in enumerate(results.boxes.data.tolist()):
        x1, y1, x2, y2, confidence, class_id = result

        if confidence < config.CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        plate_img = image[y1:y2, x1:x2]

        # Save individual plate image
        plate_path = os.path.join(config.OUTPUT_DIR, f"1. plate_{i}.jpg")
        cv2.imwrite(plate_path, plate_img)

        # Draw rectangle on original image
        cv2.rectangle(original_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            original_with_boxes,
            f"conf: {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        plates.append(
            {
                "plate_path": plate_path,
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
            }
        )

    # Save visualisation
    cv2.imwrite(
        os.path.join(config.OUTPUT_DIR, "1. detection_visualisation.jpg"),
        original_with_boxes,
    )

    # Save detection results
    with open(os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILE), "w") as f:
        json.dump(plates, f, indent=4)

    return plates
