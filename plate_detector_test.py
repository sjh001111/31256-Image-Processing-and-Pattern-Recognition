import os

import cv2
from ultralytics import YOLO


def detect_license_plate(image_path):
    # 학습된 모델 로드 (runs/detect/train/weights/best.pt가 최고 성능 모델)
    model = YOLO('best.pt')

    # 이미지에서 번호판 감지
    results = model(image_path)[0]

    # 원본 이미지 로드
    image = cv2.imread(image_path)

    # 각 감지된 번호판에 대해
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = result

        # 박스 그리기
        cv2.rectangle(image,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0), 2)

        # 신뢰도 표시
        text = f'{confidence:.2f}'
        cv2.putText(image, text,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # 결과 저장
    output_path = f'detected_{os.path.basename(image_path)}'
    cv2.imwrite(output_path, image)
    print(f'결과가 {output_path}에 저장되었습니다.')

    return results


# 사용 예시
if __name__ == "__main__":
    # 테스트할 이미지 경로
    image_path = "test.jpg"  # 테스트하고 싶은 이미지 경로로 변경하세요
    results = detect_license_plate(image_path)

    # 감지된 모든 번호판의 위치와 신뢰도 출력
    for box in results.boxes:
        print(f'번호판 감지: 신뢰도 = {box.conf.item():.2f}')