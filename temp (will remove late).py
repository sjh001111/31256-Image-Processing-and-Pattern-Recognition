import os
import cv2
from ultralytics import YOLO

def detect_license_plate(image_path):
    # Load trained model (best.pt is the best performing model)
    model = YOLO('best.pt')

    # Detect license plates in the image
    results = model(image_path)[0]

    # Load original image
    image = cv2.imread(image_path)

    # For each detected license plate
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = result

        # Draw bounding box
        cv2.rectangle(image,
                     (int(x1), int(y1)),
                     (int(x2), int(y2)),
                     (0, 255, 0), 2)

        # Display confidence score
        text = f'{confidence:.2f}'
        cv2.putText(image, text,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Save results
    output_path = f'detected_{os.path.basename(image_path)}'
    cv2.imwrite(output_path, image)
    print(f'Results saved to {output_path}')

    return results


# Usage example
if __name__ == "__main__":
    # Test image path
    image_path = "test.jpg"  # Change this to your desired image path
    results = detect_license_plate(image_path)

    # Print locations and confidence scores for all detected plates
    for box in results.boxes:
        print(f'License plate detected: confidence = {box.conf.item():.2f}')