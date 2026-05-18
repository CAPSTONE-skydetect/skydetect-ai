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

        