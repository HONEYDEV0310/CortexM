import pybullet as p


class GimbalSystem:

  def __init__(self):
    # 짐벌 링크 구조 정의
    link_Masses = [0.1, 0.05]
    link_CollisionShapeIndices = [
        p.createCollisionShape(p.GEOM_CYLINDER, radius=0.04, height=0.03),
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.01]),
    ]
    link_VisualShapeIndices = [
        p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.04,
            length=0.04,
            rgbaColor=[0.1, 0.1, 0.1, 1],
        ),
        p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.04, 0.04, 0.01],
            rgbaColor=[0.2, 0.6, 0.9, 1],
        ),
    ]

    link_Positions = [[0, 0, 0.03], [0, 0, 0.04]]
    link_Orientations = [[0, 0, 0, 1], [0, 0, 0, 1]]
    link_JointTypes = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
    link_JointAxis = [[0, 0, 1], [1, 0, 0]]
    linkParentIndices = [0, 1]

    # 짐벌 본체 생성
    self.gimbalId = p.createMultiBody(
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

    # 카메라 위치 시각화용 큐브 생성
    cam_cube_visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.01, 0.015, 0.01], rgbaColor=[1, 0, 0, 1]
    )
    self.camCubeId = p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=cam_cube_visual,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
    )

    self.pan_angle = 0.0
    self.tilt_angle = 0.0

  def update_camera_pose(self):
    """틸트 링크의 움직임에 맞춰 카메라 시각화 큐브와 뷰 행렬을 동기화"""
    link_state = p.getLinkState(self.gimbalId, 1)
    link_pos = link_state[4]
    link_orn = link_state[5]

    rot_matrix = p.getMatrixFromQuaternion(link_orn)
    rot_matrix = list(rot_matrix)
    # 3x3 행렬 변환
    rot_matrix_np = (
        __import__("numpy")
        .array(rot_matrix)
        .reshape(3, 3)
    )

    cam_pos = __import__("numpy").array(link_pos) + __import__("numpy").dot(
        rot_matrix_np, __import__("numpy").array([0.0, 0.04, 0.0])
    )
    p.resetBasePositionAndOrientation(self.camCubeId, cam_pos, link_orn)

    cam_target = cam_pos + __import__("numpy").dot(
        rot_matrix_np, __import__("numpy").array([0.0, 1.0, 0.0])
    )
    cam_up = __import__("numpy").dot(
        rot_matrix_np, __import__("numpy").array([0.0, 0.0, 1.0])
    )

    return cam_pos, cam_target, cam_up

  def set_target_angles(self, pan, tilt):
    """모터에 목표 각도 명령 전달"""
    self.pan_angle = max(-1.57, min(1.57, pan))
    self.tilt_angle = max(-0.78, min(0.78, tilt))

    p.setJointMotorControl2(
        self.gimbalId,
        0,
        p.POSITION_CONTROL,
        targetPosition=self.pan_angle,
        force=10.0,
    )
    p.setJointMotorControl2(
        self.gimbalId,
        1,
        p.POSITION_CONTROL,
        targetPosition=self.tilt_angle,
        force=10.0,
    )