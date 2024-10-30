import os
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from ultralytics import YOLO

class LicensePlateRecognizer:
    def __init__(self, debug_mode=False):
        # YOLO 모델 로드
        self.detector = YOLO('best.pt')
        # PaddleOCR 초기화 - 최신 설정
        self.reader = PaddleOCR(
            use_angle_cls=True,  # 텍스트 방향 감지
            lang='en',          # 영어 모델 사용
            use_gpu=False,      # GPU 사용 여부
            show_log=False,     # 로그 출력 제거
            # 최신 버전 모델 사용
            det_model_dir=None,  # 자동으로 최신 모델 다운로드
            rec_model_dir=None,
            cls_model_dir=None,
            # 인식 파라미터 최적화
            rec_char_dict_path=None,  # 기본 딕셔너리 사용
            drop_score=0.5,          # 신뢰도 임계값
            rec_image_shape="3, 48, 320"  # 이미지 shape 최적화
        )
        self.debug_mode = debug_mode
        if debug_mode:
            os.makedirs('debug', exist_ok=True)

    def save_debug_image(self, img, title="image"):
        """디버그용 이미지 저장"""
        if not self.debug_mode:
            return
        output_path = os.path.join('debug', f'{title}.jpg')
        cv2.imwrite(output_path, img)
        print(f'이미지가 {output_path}에 저장되었습니다.')

    def enhance_plate(self, img):
        """번호판 이미지 향상"""
        # 이미지가 너무 작으면 리사이즈
        min_height = 100
        height, width = img.shape[:2]
        if height < min_height:
            scale = min_height / height
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 대비 향상
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def detect_and_read_plate(self, image_path):
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("이미지를 불러올 수 없습니다.")

        if self.debug_mode:
            self.save_debug_image(image, "0_input_image")

        # 번호판 감지
        results = self.detector(image_path)[0]
        plates_info = []

        for idx, result in enumerate(results.boxes.data.tolist(), 1):
            x1, y1, x2, y2, confidence, class_id = result

            # 번호판 영역 추출 (여백 추가)
            y_padding = int((y2 - y1) * 0.1)
            x_padding = int((x2 - x1) * 0.05)

            y1_pad = max(0, int(y1) - y_padding)
            y2_pad = min(image.shape[0], int(y2) + y_padding)
            x1_pad = max(0, int(x1) - x_padding)
            x2_pad = min(image.shape[1], int(x2) + x_padding)

            plate_img = image[y1_pad:y2_pad, x1_pad:x2_pad]
            if plate_img.size == 0:
                continue

            if self.debug_mode:
                print(f"\n=== 번호판 {idx} 처리 중 ===")
                print(f"감지 신뢰도: {confidence:.2f}")
                self.save_debug_image(plate_img, f"{idx}_plate_original")

            # 이미지 향상
            enhanced_plate = self.enhance_plate(plate_img)
            if self.debug_mode:
                self.save_debug_image(enhanced_plate, f"{idx}_plate_enhanced")

            # OCR 수행 - 원본과 향상된 이미지 모두 시도
            ocr_results = []

            # 1. 원본 이미지로 시도
            result1 = self.reader.ocr(plate_img, cls=True)
            if result1:
                for line in result1:
                    for item in line:
                        text = item[1][0]  # 텍스트
                        conf = item[1][1]  # 신뢰도
                        ocr_results.append((text, conf))

            # 2. 향상된 이미지로 시도
            result2 = self.reader.ocr(enhanced_plate, cls=True)
            if result2:
                for line in result2:
                    for item in line:
                        text = item[1][0]
                        conf = item[1][1]
                        ocr_results.append((text, conf))

            # 가장 높은 신뢰도의 결과 선택
            if ocr_results:
                # 신뢰도가 가장 높은 결과 선택
                plate_text, ocr_confidence = max(ocr_results, key=lambda x: x[1])
                # 공백 제거 및 대문자 변환
                plate_text = ''.join(plate_text.split()).upper()
            else:
                plate_text, ocr_confidence = "", 0.0

            if self.debug_mode:
                print(f"인식된 텍스트: {plate_text}")
                print(f"OCR 신뢰도: {ocr_confidence:.2f}")

            # 결과 저장
            plates_info.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'text': plate_text,
                'ocr_confidence': ocr_confidence
            })

            # 결과 표시
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, plate_text,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        if self.debug_mode:
            self.save_debug_image(image, "final_result")

        # 결과 저장
        output_path = f'detected_{os.path.basename(image_path)}'
        cv2.imwrite(output_path, image)

        return plates_info

# 사용 예시
if __name__ == "__main__":
    recognizer = LicensePlateRecognizer(debug_mode=True)

    image_path = "test.jpg"

    try:
        results = recognizer.detect_and_read_plate(image_path)

        for idx, plate in enumerate(results, 1):
            print(f"\n번호판 {idx}:")
            print(f"텍스트: {plate['text']}")
            print(f"감지 신뢰도: {plate['confidence']:.2f}")
            print(f"OCR 신뢰도: {plate['ocr_confidence']:.2f}")

    except Exception as e:
        print(f"오류 발생: {e}")