import os
import pickle
import numpy as np
from datetime import datetime

from generators import Environment, BirdDyn, DroneDyn

class BatchRunner:
    def __init__(self, output_dir: str = "data", fps: int = 30):
        """
        Phase 1 : Batch Runner 클래스
        :param output_dir : 생성된 기동 데이터 (.pkl)를 저장할 상대 경로
        :param fps : 시뮬레이션의 초당 프레임 수 (기존 generators.py 기본값 30 적용)
        """
        self.output_dir = output_dir
        self.fps = fps
        self.dt = 1.0 / fps
        self.max_frames = 300 #고정변수 : 300 프레임 (약 10초 비행)
        
        # 저장 디렉토리 자동 생성
        os.makedirs(self.output_dir, exist_ok=True)

    def _run_single_simulation(self, scenario: str, agent_type: str, sub_type: str, sample_idx: int) -> dict:
        """
        단일 비행 시퀀스를 물리 엔진 상에서 가동 & 2D 가상 카메라 관측 데이터를 추출
        """
        
        # 1. 고유 결정론적 Seed 생성을 통한 세션 간 완벽한 실험 재현성 확보
        scenario_map = {"steady_cruise": 1, "sudden_dash": 2, "sharp_turns": 3, "multi_mode": 4}
        agent_map = {"bird": 10, "drone": 20}
        sub_map = {"pigeon": 1, "seagull": 2, "falcon": 3, "quadcopter": 4}

        unique_seed = (sample_idx * 10000) + (agent_map[agent_type] * 100) + (scenario_map[scenario] * 10) + sub_map[sub_type]
        np.random.seed(unique_seed)

        # 2️. 시나리오별 맞춤형 환경 변수(가변 변수) 세분화 설정
        base_goal = [300.0, 50.0, 50.0]  # 기본 목적지 공간 좌표
        start_pos = [0.0, np.random.uniform(75.0, 85.0), np.random.uniform(95.0, 105.0)]
        start_speed = np.random.uniform(10.0, 14.0)

        if scenario == "steady_cruise":
            # 시나리오 A: 낮은 풍속과 안정적인 직선 기조 유도
            wind_speed = np.random.uniform(1.0, 3.0)
            gust_intensity = np.random.uniform(0.1, 0.3)
            goal_pos = [base_goal[0] + np.random.uniform(-10, 10), base_goal[1], base_goal[2]]

        elif scenario == "sudden_dash":
            # 시나리오 B: 중간 풍속 및 전방 대시 기동 유도
            wind_speed = np.random.uniform(2.0, 5.0)
            gust_intensity = np.random.uniform(0.2, 0.4)
            goal_pos = base_goal.copy()

        elif scenario == "sharp_turns":
            # 시나리오 C: 강력한 측풍 외란 및 지그재그 회전 유도
            wind_speed = np.random.uniform(5.0, 8.0)  # 논문 기준 강풍 조건
            gust_intensity = np.random.uniform(0.6, 0.9)
            goal_pos = base_goal.copy()

        elif scenario == "multi_mode":
            # 시나리오 D: 복합 모드 비행 환경 세팅
            wind_speed = np.random.uniform(2.0, 6.0)
            gust_intensity = np.random.uniform(0.3, 0.6)
            goal_pos = base_goal.copy()

        # 3️. 환경 및 역학 에이전트 인스턴스화
        env = Environment(fps=self.fps, wind_speed=wind_speed, gust_intensity=gust_intensity, goal_pos=goal_pos)
        
        if agent_type == "bird":
            agent = BirdDyn(env, species=sub_type, start_pos=start_pos, start_speed=start_speed)
        else:
            agent = DroneDyn(env, model=sub_type, start_pos=start_pos, start_speed=start_speed)
        
        # 설계서 명세 규격에 맞춘 계층 구조 사전 정의
        sample_id = f"{sub_type}_{scenario}_{sample_idx:03d}"
        sim_entry = {
            "metadata": {
                "sample_id": sample_id,
                "label": "bird" if agent_type == "bird" else "drone",
                "scenario": scenario,
                "wind_speed": round(wind_speed, 2),
                "fps": self.fps
            },
            "observations": []
        }

        # 4️. 내부 루프 (Sample Loop): 300 프레임 시뮬레이션 타임라인 제어
        for frame in range(self.max_frames):
            
            # [시나리오 동적 제어 레이어 구현]
            if scenario == "sudden_dash":
                if frame == 100:
                    # 목적지를 순식간에 전방으로 멀리 이동시켜 급가속(Dash) 유도
                    env.x_goal[0] += 250.0
                elif frame == 200:
                    # 목적지를 기체 바로 뒤쪽으로 배치하여 급브레이크(Braking) 기동 강제
                    env.x_goal = agent.pos - (agent.u * 60.0)

            elif scenario == "sharp_turns":
                # 지그재그 및 연속적인 예각 선회 유도 (슬라롬 기동)
                if frame == 60:
                    env.x_goal = np.array([agent.pos[0] + 50.0, 180.0, 90.0])
                elif frame == 140:
                    env.x_goal = np.array([agent.pos[0] + 50.0, -80.0, 40.0])
                elif frame == 220:
                    env.x_goal = np.array([agent.pos[0] + 80.0, 50.0, 60.0])

            elif scenario == "multi_mode":
                # 임무 기반 다중 모드: 100~180 프레임 구간 동안 목적지를 현재 위치로 고정하여
                # 드론에게는 호버링(Hovering)을, 새에게는 제자리 선회(Circling) 루프 유도
                if 100 <= frame <= 180:
                    env.x_goal = agent.pos.copy()
                elif frame == 181:
                    env.x_goal = np.array([base_goal[0] + 150.0, base_goal[1], base_goal[2]])

            # 물리 모델 1스텝 구동 (3D 좌표 변위 계산)
            agent.step()

            # 5️⃣ Early Stopping 예외 제어 (지면 추락 검사)
            if agent.pos[2] <= 0:
                break

            # 2D 관측 데이터 슬라이싱 투영 및 기록
            obs = agent.get_observation(frame_index=frame)
            sim_entry["observations"].append(obs)

        return sim_entry

        