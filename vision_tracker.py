# import cv2
# import numpy as np
# import tensorflow as tf


# class PIDController:

#   def __init__(self, kp, ki, kd):
#     self.kp = kp
#     self.ki = ki
#     self.kd = kd
#     self.previous_error = 0.0
#     self.integral = 0.0

#   def compute(self, error, dt=1.0 / 240.0):
#     self.integral += error * dt
#     derivative = (error - self.previous_error) / dt
#     output = self.kp * error + self.ki * self.integral + self.kd * derivative
#     self.previous_error = error
#     return output


# class VisionTracker:

#   def __init__(
#       self,
#       model_path="custom_green_model.tflite",
#       target_label_index=1,
#       kp=0.002,
#       ki=0.0,
#       kd=0.0001,
#   ):
#     # PID 제어기 초기화
#     self.pan_pid = PIDController(kp, ki, kd)
#     self.tilt_pid = PIDController(kp, ki, kd)

#     # 라벨 정의 (0: 배경, 1: 초록색)
#     self.labels = ["background", "green"]
#     self.target_label_index = target_label_index

#     # TFLite 모델 로드
#     try:
#       self.interpreter = tf.lite.Interpreter(model_path=model_path)
#       self.interpreter.allocate_tensors()
#       print(f"✅ [Vision] 커스텀 TFLite 모델 로드 성공: {model_path}")
#     except Exception as e:
#       print(f"❌ [Vision] TFLite 모델 로드 실패: {e}")
#       exit()

#     self.input_details = self.interpreter.get_input_details()
#     self.output_details = self.interpreter.get_output_details()
#     self.input_shape = self.input_details[0]["shape"]  # [1, 96, 96, 3]

#   def process_frame(self, frame):
#     """프레임을 96x96으로 전처리 후 TFLite 추론 수행 및 타겟 추적"""
#     orig_height, orig_width, _ = frame.shape

#     # 1. 모델 입력 크기(96x96)에 맞게 전처리
#     input_size = (self.input_shape[1], self.input_shape[2])
#     resized_frame = cv2.resize(frame, input_size)

#     # OpenCV BGR -> RGB 변환 및 정규화
#     input_data = (
#         cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
#         / 255.0
#     )
#     input_data = np.expand_dims(input_data, axis=0)

#     # 2. TFLite 추론 실행
#     self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
#     self.interpreter.invoke()
#     output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

#     has_target = False
#     pan_adj, tilt_adj = 0.0, 0.0
#     best_cx, best_cy = 0, 0
#     max_confidence = 0.0

#     try:
#       preds = np.squeeze(output_data)  # 형태: (12, 12, 2)

#       if len(preds.shape) == 3:
#         grid_h, grid_w, num_classes = preds.shape

#         for h in range(grid_h):
#           for w in range(grid_w):
#             scores = preds[h, w]  # [배경 확률, 초록색 확률]
#             if len(scores) > self.target_label_index:
#               confidence = float(scores[self.target_label_index])

#               # 신뢰도 임계값 0.8 이상일 때 타겟 인정
#               if confidence > 0.8 and confidence > max_confidence:
#                 max_confidence = confidence
#                 best_cx = int((w + 0.5) * (orig_width / grid_w))
#                 best_cy = int((h + 0.5) * (orig_height / grid_h))
#                 has_target = True

#     except Exception as e:
#       print(f"❌ 파싱 에러 상세 내용: {e}")

#     # 3. 타겟이 포착된 경우에만 PID 오차 계산 및 시각화
#     if has_target:
#       img_center_x = orig_width // 2
#       img_center_y = orig_height // 2

#       error_x = best_cx - img_center_x
#       error_y = best_cy - img_center_y

#       pan_adj = self.pan_pid.compute(error_x)
#       tilt_adj = self.tilt_pid.compute(error_y)

#       # 타겟 위치에 초록색 원과 신뢰도 표시
#       cv2.circle(frame, (best_cx, best_cy), 10, (0, 255, 0), 2)
#       cv2.putText(
#           frame,
#           f"Green Conf: {max_confidence:.2f}",
#           (max(0, best_cx - 40), max(20, best_cy - 15)),
#           cv2.FONT_HERSHEY_SIMPLEX,
#           0.5,
#           (0, 255, 0),
#           2,
#       )

#     # 화면 중앙 십자가 라인 (기준선)
#     cv2.line(
#         frame,
#         (orig_width // 2, 0),
#         (orig_width // 2, orig_height),
#         (255, 255, 255),
#         1,
#     )
#     cv2.line(
#         frame,
#         (0, orig_height // 2),
#         (orig_width, orig_height // 2),
#         (255, 255, 255),
#         1,
#     )

#     return has_target, pan_adj, tilt_adj, frame

import cv2
import numpy as np
import tensorflow as tf


class PIDController:

  def __init__(self, kp, ki, kd, max_output=0.1):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.previous_error = 0.0
    self.integral = 0.0
    self.max_output = (
        max_output  # 🛑 한 번에 움직일 수 있는 최대 각도 변화량 제한 (클램핑)
    )

  def compute(self, error, dt=1.0 / 240.0):
    self.integral += error * dt
    derivative = (error - self.previous_error) / dt
    output = self.kp * error + self.ki * self.integral + self.kd * derivative

    # 출력값이 너무 크게 튀지 않도록 상하한 제한 걸기
    output = max(-self.max_output, min(self.max_output, output))

    self.previous_error = error
    return output


class VisionTracker:

  def __init__(
      self,
      model_path="custom_green_model.tflite",
      target_label_index=1,
      kp=0.001,
      ki=0.0,
      kd=0.0002,
  ):
    # PID 제어기 초기화 (max_output으로 한 번에 꺾이는 최대 각도 제한)
    self.pan_pid = PIDController(kp, ki, kd, max_output=0.03)
    self.tilt_pid = PIDController(kp, ki, kd, max_output=0.03)

    # 라벨 정의 (0: 배경, 1: 초록색)
    self.labels = ["background", "green"]
    self.target_label_index = target_label_index

    # TFLite 모델 로드
    try:
      self.interpreter = tf.lite.Interpreter(model_path=model_path)
      self.interpreter.allocate_tensors()
      print(f"✅ [Vision] 커스텀 TFLite 모델 로드 성공: {model_path}")
    except Exception as e:
      print(f"❌ [Vision] TFLite 모델 로드 실패: {e}")
      exit()

    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()
    self.input_shape = self.input_details[0]["shape"]  # [1, 96, 96, 3]

  def process_frame(self, frame):
    """프레임을 96x96으로 전처리 후 TFLite 추론 수행 및 타겟 추적"""
    orig_height, orig_width, _ = frame.shape

    # 1. 모델 입력 크기(96x96)에 맞게 전처리
    input_size = (self.input_shape[1], self.input_shape[2])
    resized_frame = cv2.resize(frame, input_size)

    # OpenCV BGR -> RGB 변환 및 정규화
    input_data = (
        cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        / 255.0
    )
    input_data = np.expand_dims(input_data, axis=0)

    # 2. TFLite 추론 실행
    self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
    self.interpreter.invoke()
    output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

    has_target = False
    pan_adj, tilt_adj = 0.0, 0.0
    best_cx, best_cy = 0, 0
    max_confidence = 0.0

    try:
      preds = np.squeeze(output_data)  # 형태: (12, 12, 2)

      if len(preds.shape) == 3:
        grid_h, grid_w, num_classes = preds.shape

        for h in range(grid_h):
          for w in range(grid_w):
            scores = preds[h, w]  # [배경 확률, 초록색 확률]
            if len(scores) > self.target_label_index:
              confidence = float(scores[self.target_label_index])

              # 신뢰도 임계값 0.7 이상일 때 타겟 인정
              if confidence > 0.7 and confidence > max_confidence:
                max_confidence = confidence
                best_cx = int((w + 0.5) * (orig_width / grid_w))
                best_cy = int((h + 0.5) * (orig_height / grid_h))
                has_target = True

    except Exception as e:
      print(f"❌ 파싱 에러 상세 내용: {e}")

    # 3. 타겟이 포착된 경우에만 PID 오차 계산 및 시각화
    if has_target:
      img_center_x = orig_width // 2
      img_center_y = orig_height // 2

      error_x = best_cx - img_center_x
      error_y = best_cy - img_center_y

      # 🛑 [데드존 추가 추천 적용 예시] 중앙 ±10 픽셀 안쪽은 오차를 0으로 무시
      dead_zone = 10
      if abs(error_x) < dead_zone:
        error_x = 0.0
      if abs(error_y) < dead_zone:
        error_y = 0.0

      pan_adj = self.pan_pid.compute(error_x)
      tilt_adj = self.tilt_pid.compute(error_y)

      # 타겟 위치에 초록색 원과 신뢰도 표시
      cv2.circle(frame, (best_cx, best_cy), 10, (0, 255, 0), 2)
      cv2.putText(
          frame,
          f"Green Conf: {max_confidence:.2f}",
          (max(0, best_cx - 40), max(20, best_cy - 15)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          2,
      )
# 이거 제일 많이 바뀌었는데;;
    # 화면 중앙 십자가 라인 (기준선)
    cv2.line(
        frame,
        (orig_width // 2, 0),
        (orig_width // 2, orig_height),
        (255, 255, 255),
        1,
    )
    cv2.line(
        frame,
        (0, orig_height // 2),
        (orig_width, orig_height // 2),
        (255, 255, 255),
        1,
    )

    return has_target, pan_adj, tilt_adj, frame