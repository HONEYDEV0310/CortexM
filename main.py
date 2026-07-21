import time
import cv2
import pybullet as p
import pybullet_data
from gimbal import GimbalSystem
from vision_tracker import VisionTracker

# 1. 시뮬레이터 연결 및 환경 설정
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 바닥 및 타겟 생성
p.loadURDF("plane.urdf")
target_visual = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], rgbaColor=[0, 1, 0, 1]
)
targetId = p.createMultiBody(
    baseMass=0.0,
    baseVisualShapeIndex=target_visual,
    basePosition=[0.0, 0.6, 0.05],
    baseOrientation=[0, 0, 0, 1],
)
print("🟢 시뮬레이터 및 타겟 생성 완료")

# 2. 클래스 인스턴스화
gimbal = GimbalSystem()
tracker = VisionTracker(kp=0.002, ki=0.0, kd=0.0001)

try:
  while True:
    p.stepSimulation()

    # 카메라 위치 동기화 및 뷰 행렬 얻기
    cam_pos, cam_target, cam_up = gimbal.update_camera_pose()

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_pos,
        cameraTargetPosition=cam_target,
        cameraUpVector=cam_up,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
    )

    # 카메라 이미지 캡처
    width, height, rgbImg, _, _ = p.getCameraImage(
        width=320, height=320, viewMatrix=view_matrix, projectionMatrix=proj_matrix
    )

    frame = __import__("numpy").reshape(rgbImg, (height, width, 4)).astype(
        __import__("numpy").uint8
    )
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # 비전 트래커를 통한 타겟 추적 및 PID 제어값 계산
    has_target, pan_adj, tilt_adj, processed_frame = tracker.process_frame(frame)

    if has_target:
      gimbal.pan_angle -= pan_adj
      gimbal.tilt_angle -= tilt_adj
      gimbal.set_target_angles(gimbal.pan_angle, gimbal.tilt_angle)

    # 화면 출력
    cv2.imshow("Smart Turret Simulation", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

    time.sleep(1.0 / 240.0)

except p.error:
  pass

cv2.destroyAllWindows()
p.disconnect()