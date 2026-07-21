import cv2
import numpy as np


class PIDController:

  def __init__(self, kp, ki, kd):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.previous_error = 0.0
    self.integral = 0.0

  def compute(self, error, dt=1.0 / 240.0):
    self.integral += error * dt
    derivative = (error - self.previous_error) / dt
    output = self.kp * error + self.ki * self.integral + self.kd * derivative
    self.previous_error = error
    return output


class VisionTracker:

  def __init__(self, kp=0.002, ki=0.0, kd=0.0001):
    self.pan_pid = PIDController(kp, ki, kd)
    self.tilt_pid = PIDController(kp, ki, kd)

  def process_frame(self, frame):
    """프레임에서 초록색 타겟을 찾아 오차를 계산하고 PID 제어값을 반환"""
    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 초록색 HSV 범위
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    moments = cv2.moments(mask)
    has_target = False
    pan_adj, tilt_adj = 0.0, 0.0

    if moments["m00"] > 0:
      has_target = True
      cx = int(moments["m10"] / moments["m00"])
      cy = int(moments["m01"] / moments["m00"])

      img_center_x = width // 2
      img_center_y = height // 2

      error_x = cx - img_center_x
      error_y = cy - img_center_y

      # PID 출력 계산
      pan_adj = self.pan_pid.compute(error_x)
      tilt_adj = self.tilt_pid.compute(error_y)

      # 화면 시각화 표시
      cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
      cv2.line(
          frame,
          (img_center_x, 0),
          (img_center_x, height),
          (255, 255, 255),
          1,
      )
      cv2.line(
          frame, (0, img_center_y), (width, img_center_y), (255, 255, 255), 1
      )

    return has_target, pan_adj, tilt_adj, frame