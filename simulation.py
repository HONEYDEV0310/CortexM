import time
import cv2
import numpy as np
import pybullet as p
import pybullet_data

# 1. 시뮬레이터 연결 및 환경 설정
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 2. 바닥 생성
planeId = p.loadURDF("plane.urdf")

# 3. 짐벌 구조 생성
link_Masses = [0.1, 0.05]
link_CollisionShapeIndices = [
    p.createCollisionShape(p.GEOM_CYLINDER, radius=0.04, height=0.03),
    p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.01]),
]
link_VisualShapeIndices = [
    p.createVisualShape(
        p.GEOM_CYLINDER, radius=0.04, length=0.04, rgbaColor=[0.1, 0.1, 0.1, 1]
    ),
    p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.01], rgbaColor=[0.2, 0.6, 0.9, 1]
    ),
]

link_Positions = [[0, 0, 0.03], [0, 0, 0.04]]
link_Orientations = [[0, 0, 0, 1], [0, 0, 0, 1]]
link_JointTypes = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
link_JointAxis = [[0, 0, 1], [1, 0, 0]]
linkParentIndices = [0, 1]

gimbalId = p.createMultiBody(
    baseMass=0.0,
    baseCollisionShapeIndex=p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.015]
    ),
    baseVisualShapeIndex=p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.015], rgbaColor=[0.1, 0.1, 0.1, 1]
    ),
    basePosition=[0, 0, 0.015],
    baseOrientation=[0, 0, 0, 1],
    linkMasses=link_Masses,
    linkCollisionShapeIndices=link_CollisionShapeIndices,
    linkVisualShapeIndices=link_VisualShapeIndices,
    linkPositions=link_Positions,
    linkOrientations=link_Orientations,
    linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
    linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
    linkParentIndices=linkParentIndices,
    linkJointTypes=link_JointTypes,
    linkJointAxis=link_JointAxis,
)

# 카메라(빨간색 큐브) 시각화 오브젝트 생성
cam_cube_visual = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[0.01, 0.015, 0.01], rgbaColor=[1, 0, 0, 1]
)
camCubeId = p.createMultiBody(
    baseMass=0.0,
    baseVisualShapeIndex=cam_cube_visual,
    basePosition=[0, 0, 0],
    baseOrientation=[0, 0, 0, 1],
)
# 카메라 정면(약 0.05m 앞)에 초록색 타겟 큐브 생성
target_visual = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], rgbaColor=[0, 1, 0, 1]
)  # 초록색 정육면체
targetId = p.createMultiBody(
    baseMass=0.0,  # 고정된 물체
    baseVisualShapeIndex=target_visual,
    basePosition=[0.0, 0.5, 0.05],  # 짐벌 정면(Y축 방향) 0.5m 앞에 위치
    baseOrientation=[0, 0, 0, 1],
)
print("🟢 초록색 타겟 큐브가 카메라 정면에 추가되었습니다.")

# 조작용 변수
pan_angle = 0.0
tilt_angle = 0.0
step_size = 0.05

print("=== 조작 방법 ===")
print(" [←] / [→] : Pan (좌우 회전)")
print(" [↑] / [↓] : Tilt (상하 끄덕임)")
print(
    " 정면을 똑바로 보는 카메라 위치에 맞춰 빨간색 큐브가 정확히 장착되었습니다."
)

try:
  while True:
    # 1. 키보드 입력 처리
    keys = p.getKeyboardEvents()
    for key, value in keys.items():
      if (
          value & p.KEY_IS_DOWN
          or value & p.KEY_WAS_TRIGGERED
          or value & p.KEY_WAS_RELEASED
      ):
        if key == p.B3G_LEFT_ARROW:
          pan_angle += step_size
        elif key == p.B3G_RIGHT_ARROW:
          pan_angle -= step_size
        elif key == p.B3G_UP_ARROW:
          tilt_angle += step_size
        elif key == p.B3G_DOWN_ARROW:
          tilt_angle -= step_size

    # 각도 제한
    pan_angle = max(-1.57, min(1.57, pan_angle))
    tilt_angle = max(-0.78, min(0.78, tilt_angle))

    # 모터 제어 명령
    p.setJointMotorControl2(
        gimbalId, 0, p.POSITION_CONTROL, targetPosition=pan_angle, force=5.0
    )
    p.setJointMotorControl2(
        gimbalId, 1, p.POSITION_CONTROL, targetPosition=tilt_angle, force=5.0
    )

    p.stepSimulation()

    # 2. 틸트 링크 위치/회전 상태 가져오기
    link_state = p.getLinkState(gimbalId, 1)
    link_pos = link_state[4]
    link_orn = link_state[5]

    rot_matrix = p.getMatrixFromQuaternion(link_orn)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)

    # [핵심] 카메라가 정면(Y축 방향)을 바라보도록 설정한 정확한 위치
    cam_pos = np.array(link_pos) + np.dot(
        rot_matrix, np.array([0.0, 0.04, 0.0])
    )

    # [수정된 부분] 카메라가 바라보는 바로 그 위치로 빨간색 큐브를 정확히 동기화
    p.resetBasePositionAndOrientation(camCubeId, cam_pos, link_orn)

    # 카메라 시점 계산 (정면인 +Y 방향 조준)
    cam_target = cam_pos + np.dot(rot_matrix, np.array([0.0, 1.0, 0.0]))
    cam_up = np.dot(rot_matrix, np.array([0.0, 0.0, 1.0]))

    # 3. 뷰 및 프로젝션 행렬 생성
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_pos,
        cameraTargetPosition=cam_target,
        cameraUpVector=cam_up,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
    )

    # 4. 카메라 이미지 캡처
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=320, height=320, viewMatrix=view_matrix, projectionMatrix=proj_matrix
    )

    frame = np.reshape(rgbImg, (height, width, 4)).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    cv2.imshow("Gimbal Camera View", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

    time.sleep(1.0 / 240.0)

except p.error:
  pass

cv2.destroyAllWindows()
p.disconnect()