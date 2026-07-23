import time
import cv2
import numpy as np
import pybullet as p
import pybullet_data
from gimbal import GimbalSystem
from vision_tracker import VisionTracker

# 1. 시뮬레이터 연결 및 환경 설정
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 바닥 생성
p.loadURDF("plane.urdf")

# [수정] 초록색 타겟을 초기 랜덤 위치(또는 지정된 위치)에 생성하고 ID를 반환받음
initial_target_pos = [0.2, 0.6, 0.1]
target_visual = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], rgbaColor=[0, 1, 0, 1]
)
targetId = p.createMultiBody(
    baseMass=0.0,
    baseVisualShapeIndex=target_visual,
    basePosition=initial_target_pos,
    baseOrientation=[0, 0, 0, 1],
)
print("🟢 초록색 타겟 큐브가 생성되었습니다.")
print("=== 조작 안내 ===")
print(" [방향키 (←/→/↑/↓)] : 짐벌 수동 조작 (Pan/Tilt)")
print(" [W / S]           : 타겟 Y축 이동 (앞 / 뒤)")
print(" [A / D]           : 타겟 X축 이동 (좌 / 우)")
print(" [C / X]           : 타겟 Z축 이동 (상승 / 하강)")
print(" [Space Bar]       : PID 자동 추적 모드 토글 (ON / OFF)")

# 2. 클래스 인스턴스화
gimbal = GimbalSystem()
#tracker = VisionTracker(kp=0.002, ki=0.0, kd=0.0001)
tracker = VisionTracker(kp=0.001, ki=0.0, kd=0.0002)

# 제어 상태 변수
auto_tracking = False  # 스페이스바로 켜고 끌 수 있는 자동 추적 모드 플래그
step_size_gimbal = 0.05
step_size_target = 0.02
current_target_pos = list(initial_target_pos)

# 키보드 입력 상태를 안정적으로 받기 위한 직전 키 기록용 딕셔너리
last_keys = {}

try:
  while True:
    p.stepSimulation()

    # 1. 키보드 입력 처리 (수동 조작 및 타겟 이동, 모드 전환)
    keys = p.getKeyboardEvents()
    for key, value in keys.items():
      # 키가 방금 눌렸거나(TRIGGERED) 꾹 눌려있는(IS_DOWN) 상태 감지
      if value & p.KEY_IS_DOWN or value & p.KEY_WAS_TRIGGERED:

        # [짐벌 수동 조작: 방향키] (자동 모드가 아닐 때만 수동 조작 가능)
        if not auto_tracking:
          if key == p.B3G_LEFT_ARROW:
            gimbal.pan_angle += step_size_gimbal
          elif key == p.B3G_RIGHT_ARROW:
            gimbal.pan_angle -= step_size_gimbal
          elif key == p.B3G_UP_ARROW:
            gimbal.tilt_angle += step_size_gimbal
          elif key == p.B3G_DOWN_ARROW:
            gimbal.tilt_angle -= step_size_gimbal

        # [초록색 타겟 위치 이동: WASD + CX]
        if key == ord("w"):
          current_target_pos[1] += step_size_target  # 앞
        elif key == ord("s"):
          current_target_pos[1] -= step_size_target  # 뒤
        elif key == ord("a"):
          current_target_pos[0] -= step_size_target  # 좌
        elif key == ord("d"):
          current_target_pos[0] += step_size_target  # 우
        elif key == ord("c"):
          current_target_pos[2] += step_size_target  # 상승
        elif key == ord("x"):
          current_target_pos[2] -= step_size_target  # 하강

        # [스페이스바: 자동 추적 모드 토글 (ON/OFF)]
        # 키가 '방금 눌린 순간(TRIGGERED)'에만 토글되도록 처리
        if (
            key == ord(" ")
            and key not in last_keys
            and (value & p.KEY_WAS_TRIGGERED)
        ):
          auto_tracking = not auto_tracking
          mode_str = "🟢 [AUTO TRACKING ON]" if auto_tracking else "🔴 [MANUAL MODE]"
          print(mode_str)

    # 타겟 큐브의 실제 위치를 갱신 (화면에서 움직이는 것이 실시간 반영됨)
    p.resetBasePositionAndOrientation(
        targetId, current_target_pos, [0, 0, 0, 1]
    )

    last_keys = keys  # 현재 키 상태 저장

    # 2. 카메라 위치 동기화 및 이미지 캡처
    cam_pos, cam_target, cam_up = gimbal.update_camera_pose()

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_pos,
        cameraTargetPosition=cam_target,
        cameraUpVector=cam_up,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
    )

    width, height, rgbImg, _, _ = p.getCameraImage(
        width=320, height=320, viewMatrix=view_matrix, projectionMatrix=proj_matrix
    )

    frame = np.reshape(rgbImg, (height, width, 4)).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # 3. 비전 트래커를 통한 타겟 검출 및 오차 계산
    has_target, pan_adj, tilt_adj, processed_frame = tracker.process_frame(frame)

    # ==========================================
    # 4. 모터 제어 모드 분기 (수동 vs PID 자동 추적)
    # ==========================================
    if auto_tracking and has_target:
      # [자동 추적 모드] 비전 오차를 기반으로 PID 제어값 반영
      gimbal.pan_angle -= pan_adj
      gimbal.tilt_angle -= tilt_adj
      gimbal.set_target_angles(gimbal.pan_angle, gimbal.tilt_angle)

      # 화면에 현재 모드 표시
      cv2.putText(
          processed_frame,
          "MODE: AUTO TRACKING",
          (10, 25),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 255, 0),
          2,
      )
    else:
      # [수동 모드 또는 타겟 미발견 시] 사용자가 조작한 각도 또는 현재 각도 유지
      gimbal.set_target_angles(gimbal.pan_angle, gimbal.tilt_angle)

      mode_text = (
          "MODE: MANUAL" if not auto_tracking else "MODE: AUTO (TARGET LOST)"
      )
      cv2.putText(
          processed_frame,
          mode_text,
          (10, 25),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 0, 255),
          2,
      )

    # 화면 출력
    cv2.imshow("Smart Turret Simulation", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

    time.sleep(1.0 / 240.0)

except p.error:
  pass

cv2.destroyAllWindows()
p.disconnect()