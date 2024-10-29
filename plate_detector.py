import cv2
import numpy as np
from ultralytics import YOLO
import os


class LicensePlateDetector:
    def __init__(self):
        model_path = "best.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        print(f"Loading model from: {model_path}")
        self.detector = YOLO(model_path)
        print("Model loaded successfully")

    def geometric_correction(self, image):
        """
        번호판 이미지의 기하학적 보정을 수행합니다.
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러로 노이즈 제거
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 엣지 검출
        edges = cv2.Canny(blur, 50, 150)

        # 컨투어 찾기
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        # 가장 큰 컨투어 찾기
        largest_contour = max(contours, key=cv2.contourArea)

        # 근사 다각형 찾기
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # 4개의 코너를 찾지 못한 경우 원본 반환
        if len(approx) != 4:
            return image

        # 코너 포인트 정렬
        pts = np.float32(approx.reshape(4, 2))
        rect = np.zeros((4, 2), dtype=np.float32)

        # 합이 가장 작은 것이 좌상단, 가장 큰 것이 우하단
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 좌상단
        rect[2] = pts[np.argmax(s)]  # 우하단

        # 차이가 가장 작은 것이 우상단, 가장 큰 것이 좌하단
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 우상단
        rect[3] = pts[np.argmax(diff)]  # 좌하단

        # 변환할 이미지의 너비와 높이
        width = int(
            max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))
        )
        height = int(
            max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2]))
        )

        # 변환 행렬 계산
        dst_points = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )

        # 투시 변환 수행
        matrix = cv2.getPerspectiveTransform(rect, dst_points)
        warped = cv2.warpPerspective(image, matrix, (width, height))

        return warped

    def detect_and_save(
        self, image_path, output_dir="detected_plates", conf_threshold=0.3
    ):
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return

        original_img = image.copy()

        # 이미지 크기 출력
        print(f"Image shape: {image.shape}")

        # Detection 수행
        results = self.detector(image)[0]
        print(f"Number of detections: {len(results.boxes)}")

        # 결과 시각화를 위한 이미지 복사
        output_img = image.copy()

        # 감지된 모든 번호판에 대해
        for i, box in enumerate(results.boxes):
            # Confidence가 threshold보다 높은 것만 처리
            confidence = float(box.conf[0])
            if confidence < conf_threshold:
                continue

            # 박스 좌표 추출
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 좌표 출력
            print(
                f"Detection {i+1}: conf={confidence:.3f}, coords=({x1},{y1},{x2},{y2})"
            )

            # 번호판 영역 추출
            plate_img = original_img[y1:y2, x1:x2]

            # 기하학적 보정 수행
            corrected_plate = self.geometric_correction(plate_img)

            # 박스 그리기
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Confidence 표시
            label = f"{confidence:.2f}"
            cv2.putText(
                output_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            # 원본 번호판 이미지 저장
            if plate_img.size > 0:
                plate_path = os.path.join(
                    output_dir, f"plate_{i+1}_original_conf_{confidence:.2f}.jpg"
                )
                cv2.imwrite(plate_path, plate_img)
                print(f"Saved original plate image to: {plate_path}")

                # 보정된 번호판 이미지 저장
                corrected_path = os.path.join(
                    output_dir, f"plate_{i+1}_corrected_conf_{confidence:.2f}.jpg"
                )
                cv2.imwrite(corrected_path, corrected_plate)
                print(f"Saved corrected plate image to: {corrected_path}")

        # 전체 결과 저장
        output_path = os.path.join(output_dir, "detection_result.jpg")
        cv2.imwrite(output_path, output_img)
        print(f"Saved detection result to: {output_path}")


# 사용 예시
if __name__ == "__main__":
    try:
        # Detector 객체 생성
        detector = LicensePlateDetector()

        # 이미지 경로 지정
        image_path = "test.jpg"  # 테스트할 이미지 경로로 변경하세요

        # Detection 수행 및 결과 저장
        detector.detect_and_save(image_path, conf_threshold=0.3)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
