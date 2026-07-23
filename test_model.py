import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 설정 및 폴더 준비
DATASET_DIR = "dataset_green"
IMG_SIZE = 96
NUM_SAMPLES = 2000  # 💡 데이터 다양성을 위해 2,000장으로 대폭 증가

os.makedirs(os.path.join(DATASET_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "labels"), exist_ok=True)

print("📦 [1단계] 대규모 합성 데이터셋 생성 중 (흰색 배경 & 다양한 크기)...")

X_data = []
y_data = []
grid_size = 12
cell_size = IMG_SIZE // grid_size

for i in range(NUM_SAMPLES):
  # 흰색 배경 생성
  img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

  label_map = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
  label_map[:, :, 0] = 1.0  # 기본 배경 확률 1.0

  # 초록색 물체의 크기 및 위치 무작위 설정
  obj_size = np.random.randint(6, 33)
  x = np.random.randint(obj_size, IMG_SIZE - obj_size)
  y = np.random.randint(obj_size, IMG_SIZE - obj_size)

  # 초록색 큐브(사각형) 그리기
  cv2.rectangle(
      img,
      (x - obj_size // 2, y - obj_size // 2),
      (x + obj_size // 2, y + obj_size // 2),
      (0, 255, 0),
      -1,
  )

  # 해당 위치의 그리드 셀 라벨링
  gx = x // cell_size
  gy = y // cell_size
  if 0 <= gx < grid_size and 0 <= gy < grid_size:
    label_map[gy, gx, 0] = 0.0
    label_map[gy, gx, 1] = 1.0  # 초록색 타겟 존재

  X_data.append(img)
  y_data.append(label_map)

X_data = np.array(X_data, dtype=np.float32) / 255.0
y_data = np.array(y_data, dtype=np.float32)

print(f"✅ 데이터셋 생성 완료! (총 {NUM_SAMPLES}장)\n")

print("🧠 [2단계] CNN 모델 구조 개선 및 학습 중 (Epochs: 50)...")
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),  # 48x48
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),  # 24x24
    layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    ),  # 💡 채널 수 확장으로 표현력 강화
    layers.MaxPooling2D((2, 2)),  # 12x12
    layers.Conv2D(2, (1, 1), activation="softmax"),
])

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

# 💡 에폭을 50회로 늘리고, 배치 사이즈를 조절하여 충분히 학습시킵니다.
model.fit(
    X_data, y_data, epochs=50, batch_size=32, validation_split=0.1, verbose=1
)

print("\n⚙️ [3단계] TFLite 모델로 변환 및 저장 중...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = "custom_green_model.tflite"
with open(tflite_path, "wb") as f:
  f.write(tflite_model)

print(f"🎉 성공! 성능이 개선된 TFLite 파일이 저장되었습니다: '{tflite_path}'")