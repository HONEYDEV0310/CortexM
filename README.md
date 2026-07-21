# CortexM

# PyBullet 2-Axis Gimbal Simulation

파이썬과 **PyBullet** 물리 엔진을 활용하여 실제 하드웨어 구조를 정확히 모사한 **2축 짐벌(Gimbal) 시뮬레이션 환경**입니다. 
로봇 기구학의 기초인 다물체(MultiBody) 구조 설계, 관절(Joint) 제어, 그리고 카메라와 타겟 오브젝트 간의 좌표 정렬 원리를 학습하기 위해 제작되었습니다.

---

## 🚀 주요 기능 및 특징 (Features)

1. **계층형 다물체(MultiBody) 구조 설계**
   - 베이스(고정) $\rightarrow$ Pan 링크(Z축 회전) $\rightarrow$ Tilt 링크(X축 회전)로 이어지는 트리 구조 구현
2. **정밀한 관절 축(Joint Axis) 매핑**
   - Pan 모터: 수직 Z축(`[0, 0, 1]`) 기준 360도 좌우 회전
   - Tilt 모터: 수평 X축(`[1, 0, 0]`) 기준 상하 끄덕임(Pitch) 동작 구현 (짐벌 락 방지)
3. **시각화 및 충돌체(Collision/Visual Shape) 분리**
   - `halfExtents`와 기하학적 프리미티브(`GEOM_BOX`, `GEOM_CYLINDER`)를 활용한 직관적인 링크 형태 정의
   - 카메라 렌더링 시각화를 위한 전용 큐브 오브젝트 및 타겟 오브젝트(초록색 큐브) 배치

---

## 📂 프로젝트 구조 (Project Structure)

```text
├── main.py          # PyBullet 시뮬레이터 초기화 및 짐벌/타겟 생성
├── control.py       # 실시간 제어 루프 및 카메라 동기화 로직 (작성 중)
└── README.md        # 프로젝트 설명 문서
