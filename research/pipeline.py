import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import circmean

from generators import Environment, BirdDyn, DroneDyn

# 전영 물리 한계 상수 선언
V_MIN = 0.0
V_MAX = 5.0
H_MIN = 0.0
H_MAX = 1.0

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

        rng = np.random.default_rng(unique_seed)

        # 2️. 시나리오별 맞춤형 환경 변수(가변 변수) 세분화 설정
        base_goal = [300.0, 50.0, 50.0]  # 기본 목적지 공간 좌표
        start_pos = [0.0, rng.uniform(75.0, 85.0), rng.uniform(95.0, 105.0)]
        start_speed = rng.uniform(10.0, 14.0)

        if scenario == "steady_cruise":
            # 시나리오 A: 낮은 풍속과 안정적인 직선 기조 유도
            wind_speed = rng.uniform(1.0, 3.0)
            gust_intensity = rng.uniform(0.1, 0.3)
            goal_pos = [base_goal[0] + rng.uniform(-10, 10), base_goal[1], base_goal[2]]

        elif scenario == "sudden_dash":
            # 시나리오 B: 중간 풍속 및 전방 대시 기동 유도
            wind_speed = rng.uniform(2.0, 5.0)
            gust_intensity = rng.uniform(0.2, 0.4)
            goal_pos = base_goal.copy()

        elif scenario == "sharp_turns":
            # 시나리오 C: 강력한 측풍 외란 및 지그재그 회전 유도
            wind_speed = rng.uniform(5.0, 8.0)  # 논문 기준 강풍 조건
            gust_intensity = rng.uniform(0.6, 0.9)
            goal_pos = base_goal.copy()

        elif scenario == "multi_mode":
            # 시나리오 D: 복합 모드 비행 환경 세팅
            wind_speed = rng.uniform(2.0, 6.0)
            gust_intensity = rng.uniform(0.3, 0.6)
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

            # 5️. Early Stopping 예외 제어 (지면 추락 검사)
            if agent.pos[2] <= 0:
                break

            # 2D 관측 데이터 슬라이싱 투영 및 기록
            obs = agent.get_observation(frame_index=frame)
            sim_entry["observations"].append(obs)

        return sim_entry

    def execute_batch_pipeline(self, bird_samples_per_species: int = 50, drone_samples_per_model: int = 150) -> str:
        """
        4대 세분화 시나리오 전체를 순회하며 새 600개, 드론 600개(총 1,200개)의 유효 데이터셋을 대량 생산합니다.
        """
        scenarios = ["steady_cruise", "sudden_dash", "sharp_turns", "multi_mode"]
        birds = ["pigeon", "seagull", "falcon"]

        results = []
        print("=" * 65)
        print(f" [Phase 1: Batch Runner] 4대 특화 시나리오 공정 가동 시작")
        print(f"   └ 목표 수량: 조류 600개 (3종 × 50개 × 4시나리오)")
        print(f"   └ 목표 수량: 드론 600개 (1종 × 150개 × 4시나리오)")
        print("=" * 65)

        # 외부 루프 (Scenario Loop)
        for scenario in scenarios:
            print(f"현재 가동 중인 시나리오 파이프라인: [{scenario.upper()}]")
            
            # 내부 루프 1: 조류 군집 데이터 획득 (시나리오당 종별 50개 샘플)
            for species in birds:
                valid_count = 0
                idx = 1
                while valid_count < bird_samples_per_species:
                    sim_data = self._run_single_simulation(scenario, "bird", species, idx)
                    # 데이터 유효 품질 방어벽 (최소 50프레임 이상 비행한 데이터만 인정)
                    if len(sim_data["observations"]) >= 150:
                        results.append(sim_data)
                        valid_count += 1
                    idx += 1

            # 내부 루프 2: 드론 군집 데이터 획득 (시나리오당 150개 샘플로 클래스 균형 추정 증폭)
            valid_count = 0
            idx = 1
            while valid_count < drone_samples_per_model:
                sim_data = self._run_single_simulation(scenario, "drone", "quadcopter", idx)
                if len(sim_data["observations"]) >= 150:
                    results.append(sim_data)
                    valid_count += 1
                idx += 1
        
        # 6️. 데이터 대량 생산 완료 후 pkl 직렬화 물리 저장
        output_path = os.path.join(self.output_dir, "batch_raw_trajectories.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        print("-" * 65)
        print(f"✅ [Phase 1 완료] 클래스 균형 데이터셋 원천 생산 공정 완료.")
        print(f" 총 저장된 유효 샘플 수: {len(results)} 개")
        print(f" 바이너리 팩토리 파일 저장 완료 경로: {output_path}")
        print("=" * 65)
        return output_path

class CoreFeatureExtractor:
    def __init__(self, data_dir: str):
        """
        :param data_dir: pkl 파일이 있고 최종 csv 파일이 생성될 디렉토리 경로
        """
        self.data_dir = data_dir
        self.input_path = os.path.join(data_dir, "batch_raw_trajectories.pkl")
        self.output_path = os.path.join(data_dir, "simulation_features.csv")

    def extract_features(self) -> str:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"원천 시계열 바이너리가 {self.input_path}에 존재하지 않습니다.")

        print("=" * 65)
        print("[Phase 2] 물리/동역학 논문 기반 핵심 특징 추출(Feature Engineering) 시작")
        print("=" * 65)

        with open(self.input_path, "rb") as f:
            raw_datasets = pickle.load(f)

        feature_rows = []

        for sample in raw_datasets:
            obs = sample["observations"]
            label = sample["metadata"]["label"]
            dt = 1.0 / sample["metadata"]["fps"]
            N = len(obs)

            # 초해상도 및 미분 연산 한계선 방어벽
            if N < 3:
                continue

            # 벡터 연산 속도 향상을 위한 데이터 배열 변환
            cx = np.array([p["cx"] for p in obs])
            cy = np.array([p["cy"] for p in obs])
            w = np.array([p["w"] for p in obs])

            # 1️. [수식 적용] 순간 변위 및 신체 길이 정규화 속도(v_norm) 계산
            dx = np.diff(cx)
            dy = np.diff(cy)
            displacement = np.sqrt(dx**2 + dy**2)
            
            # 원근 왜곡 제거를 위해 프레임의 몸길이(w)로 나눈 후 속도로 환산 (BL/s)
            v_norm_series = displacement / (w[1:] * dt + 1e-6)
            
            v_mean = float(np.mean(v_norm_series))
            v_std = float(np.std(v_norm_series))

            # 2️. [수식 적용] 정규화 속도의 시간에 대한 1차 미분 (가속도 a_mean)
            a_norm_series = np.abs(np.diff(v_norm_series)) / dt
            a_mean = float(np.mean(a_norm_series))

            # 3️. [수식 적용] 3번 논문 식 (3),(4) 360도 경계면 보정 방향 편차(heading_change_ratio) 정밀 산출
            headings = np.degrees(np.arctan2(dy, dx))
            headings = (headings + 360) % 360  # 0~360도로 변환 및 스케일 바인딩
            
            # 원형 통계학을 적용한 순환 평균 방향 획득
            h_mean = circmean(headings, high=360, low=0)
            
            delta_h_list = []
            for h_i in headings:
                diff = abs(h_i - h_mean)
                # 원형 공간 최단 기하 거리를 도출하는 수학적 공식 적용
                shortest_diff = min(diff, 360.0 - diff)
                delta_h_list.append(shortest_diff ** 2)
            
            raw_h_std = np.sqrt(np.sum(delta_h_list) / len(headings))
            
            # 최대 편차 한계(180도)로 나눠 shared schema 규격 (0.0~1.0 ratio)에 동기화
            heading_change_ratio = float(raw_h_std / 180.0)

            # 일차적인 파생 변수 로우 적재 (maneuverability_sigma 계산 전 임시 풀링)
            feature_rows.append({
                "v_mean": v_mean,
                "v_std": v_std,
                "a_mean": a_mean,
                "heading_change_ratio": heading_change_ratio,
                "label": label
            })

        df = pd.DataFrame(feature_rows)

        # 4️. [수식 적용] 복합 기동성 지표 (maneuverability_sigma) 글로벌 풀 기반 정규화 산출
        v_scaled = (df["v_mean"] - V_MIN) / (V_MAX - V_MIN + 1e-6)
        h_scaled = (df["heading_change_ratio"] - H_MIN) / (H_MAX - H_MIN + 1e-6)

        df["maneuverability_sigma"] = v_scaled / (h_scaled + 1e-6)

        # 5️. C파트 담당자의 RandomForest 주입 변수 리스트 컬럼 명세 정렬 동기화
        ordered_columns = ["v_mean", "v_std", "a_mean", "heading_change_ratio", "maneuverability_sigma", "label"]
        df = df[ordered_columns]

        # 물리 데이터 저장 발행
        df.to_csv(self.output_path, index=False)
        print(f"✅ [Phase 2 완료] CSV 피처 매트릭스 테이블 발행 완료.")
        print(f"저장 경로: {self.output_path}")
        print("=" * 65)
        return self.output_path

if __name__ == "__main__":

    # 실행 환경에 구애받지 않는 안정적인 절대 경로 기저 동적 바인딩
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    target_data_dir = os.path.join(current_script_dir, "data")

    # 정밀 명세 스펙 가동 (4 시나리오 × 조류종별 50개 / 드론 150개 = 1200개 Balanced 데이터 구축)
    runner = BatchRunner(output_dir=target_data_dir, fps=30)
    runner.execute_batch_pipeline(bird_samples_per_species=50, drone_samples_per_model=150)
    
    # 논문 기반 피처 엔지니어링 수행 (.csv)
    extractor = CoreFeatureExtractor(data_dir=target_data_dir)
    extractor.extract_features()