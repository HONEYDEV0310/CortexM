import cv2
import numpy as np
import tensorflow as tf

# 1. 설정
model_path = "custom_green_model.tflite"
IMG_SIZE = 96
GRID_SIZE = 12
NUM_TEST_IMAGES = 30

print("========================================")
print("📦 수치 정확도 + 시각화 TFLite 모델 검증 시작")
print("========================================")

# 2. 모델 로드
try:
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  print("✅ 모델 로드 성공!\n")
except Exception as e:
  print(f"❌ 모델 로드 실패: {e}")
  exit()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

window_name = "Custom Model Detailed Test"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 450, 450)

print(f"총 {NUM_TEST_IMAGES}개의 테스트 이미지를 검증합니다.")
print("아무 키나 누르면 다음 이미지로 넘어가며, ESC를 누르면 종료됩니다.\n")

success_count = 0

for i in range(NUM_TEST_IMAGES):
  # 3. 흰색 배경 및 무작위 크기/위치의 초록색 큐브 생성 (정답 위치 기억)
  test_img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

  obj_size = np.random.randint(6, 33)
  x = np.random.randint(obj_size, IMG_SIZE - obj_size)
  y = np.random.randint(obj_size, IMG_SIZE - obj_size)

  # 정답 바운딩 박스
  box_pts = (
      x - obj_size // 2,
      y - obj_size // 2,
      x + obj_size // 2,
      y + obj_size // 2,
  )

  # 초록색 큐브 그리기
  cv2.rectangle(
      test_img, (box_pts[0], box_pts[1]), (box_pts[2], box_pts[3]), (0, 255, 0), -1
  )

  # 4. 모델 입력 전처리 및 추론
  input_data = test_img.astype(np.float32) / 255.0
  input_data = np.expand_dims(input_data, axis=0)

  interpreter.set_tensor(input_details[0]["index"], input_data)
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]["index"])

  preds = np.squeeze(output_data)  # 형태: (12, 12, 2)

  # 5. 수치 분석 및 최고 신뢰도 셀 탐색
  grid_h, grid_w, _ = preds.shape
  max_conf = 0.0
  best_grid = (-1, -1)

  for h in range(grid_h):
    for w in range(grid_w):
      conf = float(preds[h, w, 1])  # 1번 클래스(초록색) 확률
      if conf > max_conf:
        max_conf = conf
        best_grid = (h, w)

  # 6. 시각화 (450x450 크기로 확대)
  display_img = cv2.resize(
      test_img, (450, 450), interpolation=cv2.INTER_NEAREST
  )
  scale = 450 / IMG_SIZE

  # 정답 박스 표시 (빨간색 테두리)
  cv2.rectangle(
      display_img,
      (int(box_pts[0] * scale), int(box_pts[1] * scale)),
      (int(box_pts[2] * scale), int(box_pts[3] * scale)),
      (0, 0, 255),
      1,
  )

  detected = False
  # 신뢰도 임계값 0.6 이상일 때 타겟 감지로 판정
  if max_conf >= 0.6:
    detected = True
    success_count += 1
    gh, gw = best_grid
    cell_w = 450 / grid_w
    cell_h = 450 / grid_h
    cx = int((gw + 0.5) * cell_w)
    cy = int((gh + 0.5) * cell_h)

    # 모델이 예측한 중심점 표시 (초록색 원)
    cv2.circle(display_img, (cx, cy), 10, (0, 200, 0), 2)
    cv2.putText(
        display_img,
        f"Conf: {max_conf:.4f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 150, 0),
        2,
    )
  else:
    cv2.putText(
        display_img,
        f"Conf: {max_conf:.4f} (Miss)",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

  # 7. 터미널 수치 출력
  print(
      f"[테스트 #{i+1:02d}] 감지 여부: {'성공' if detected else '실패'} |"
      f" 최고 신뢰도(Confidence): {max_conf:.6f} | 예측 그리드 셀: {best_grid}"
  )

  # 상단 상태 텍스트
  status_text = f"Test [{i+1}/{NUM_TEST_IMAGES}] - Acc: {success_count/(i+1)*100:.1f}%"
  cv2.putText(
      display_img,
      status_text,
      (10, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (0, 0, 0),
      2,
  )

  cv2.imshow(window_name, display_img)

  key = cv2.waitKey(0)
  if key == 27:  # ESC 누르면 종료
    break

cv2.destroyAllWindows()
print("========================================")
print(
    f"🏁 최종 검증 완료! (총 {NUM_TEST_IMAGES}장 중 {success_count}장 감지 성공,"
    f" 성공률: {success_count/NUM_TEST_IMAGES*100:.1f}%)"
)
print("========================================")