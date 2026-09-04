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
        self.max_frames = 420 #최대 생성 프레임 (30FPS 기준 약 14초 비행)
        
        # 저장 디렉토리 자동 생성
        os.makedirs(self.output_dir, exist_ok=True)

    def _run_single_simulation(self, scenario: str, agent_type: str, sub_type: str, sample_idx: int, apply_noise: bool = False) -> dict:
        """
        단일 비행 시퀀스를 물리 엔진 상에서 가동 & 2D 가상 카메라 관측 데이터를 추출
        """
        
        # 1. 고유 결정론적 Seed 생성을 통한 세션 간 완벽한 실험 재현성 확보
        scenario_map = {"steady_cruise": 1, "sudden_dash": 2, "sharp_turns": 3, "multi_mode": 4}
        agent_map = {"bird": 10, "drone": 20}
        sub_map = {"pigeon": 1, "seagull": 2, "falcon": 3, "quadcopter": 4}

        unique_seed = (sample_idx * 10000) + (agent_map[agent_type] * 100) + (scenario_map[scenario] * 10) + sub_map[sub_type]

        rng = np.random.default_rng(unique_seed)

        frame_length_bucket, current_max_frames = self._sample_frame_length(rng)

        # 2️. 시나리오별 맞춤형 환경 변수(가변 변수) 세분화 설정
        bbox_depth_profile = self._sample_bbox_depth_profile(
            rng,
            current_max_frames,
        )
        start_pos = self._sample_start_position(rng, bbox_depth_profile)
        base_goal = self._sample_goal_position(rng, start_pos, bbox_depth_profile)
        start_speed = rng.uniform(10.0, 14.0)
        event_schedule = self._sample_event_schedule(
            scenario,
            current_max_frames,
            rng,
        )
        dropout_profile = self._sample_dropout_profile(
            rng,
            current_max_frames,
            apply_noise,
        )
        drift_profile = self._sample_tracking_drift_profile(
            rng,
            current_max_frames,
            apply_noise,
        )

        if scenario == "steady_cruise":
            # 시나리오 A: 낮은 풍속과 안정적인 직선 기조 유도
            wind_speed = rng.uniform(1.0, 3.0)
            gust_intensity = rng.uniform(0.1, 0.3)
            goal_pos = base_goal.copy()

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
            agent = BirdDyn(env, species=sub_type, start_pos=start_pos, start_speed=start_speed, apply_noise=apply_noise)
        else:
            agent = DroneDyn(env, model=sub_type, start_pos=start_pos, start_speed=start_speed, apply_noise=apply_noise)
        
        # 설계서 명세 규격에 맞춘 계층 구조 사전 정의
        sample_id = f"{sub_type}_{scenario}_{sample_idx:03d}"
        sim_entry = {
            "metadata": {
                "sample_id": sample_id,
                "label": "bird" if agent_type == "bird" else "drone",
                "scenario": scenario,
                "start_pos": [round(float(value), 2) for value in start_pos],
                "start_speed": round(float(start_speed), 2),
                "initial_goal_pos": [round(float(value), 2) for value in goal_pos],
                "bbox_depth_profile": bbox_depth_profile,
                "event_schedule": event_schedule,
                "frame_length_bucket": frame_length_bucket,
                "target_frame_count": int(current_max_frames),
                "tracking_drift_profile": drift_profile,
                "dropout_profile": dropout_profile,
                "wind_speed": round(wind_speed, 2),
                "fps": self.fps
            },
            "observations": []
        }

        # 4️. 내부 루프 (Sample Loop): 300 프레임 시뮬레이션 타임라인 제어
        for frame in range(current_max_frames):
            self._apply_bbox_depth_waypoint(env, frame, bbox_depth_profile)
            
            # [시나리오 동적 제어 레이어 구현]
            if scenario == "sudden_dash":
                if frame == event_schedule["dash_frame"]:
                    # 목적지를 순식간에 전방으로 멀리 이동시켜 급가속(Dash) 유도
                    env.x_goal[0] += 250.0
                elif frame == event_schedule["brake_frame"]:
                    # 목적지를 기체 바로 뒤쪽으로 배치하여 급브레이크(Braking) 기동 강제
                    env.x_goal = agent.pos - (agent.u * 60.0)

            elif scenario == "sharp_turns":
                # 지그재그 및 연속적인 예각 선회 유도 (슬라롬 기동)
                turn_frames = event_schedule["turn_frames"]
                if frame == turn_frames[0]:
                    env.x_goal = np.array(
                        [
                            max(agent.pos[0] + 140.0, base_goal[0] * 0.70),
                            base_goal[1] + 120.0,
                            np.clip(base_goal[2] + 35.0, 20.0, 180.0),
                        ]
                    )
                elif frame == turn_frames[1]:
                    env.x_goal = np.array(
                        [
                            max(agent.pos[0] + 140.0, base_goal[0] * 0.88),
                            base_goal[1] - 120.0,
                            np.clip(base_goal[2] - 45.0, 20.0, 180.0),
                        ]
                    )
                elif frame == turn_frames[2]:
                    env.x_goal = np.array(
                        [
                            max(agent.pos[0] + 180.0, base_goal[0]),
                            base_goal[1],
                            base_goal[2],
                        ]
                    )

            elif scenario == "multi_mode":
                # 임무 기반 다중 모드: 100~180 프레임 구간 동안 목적지를 현재 위치로 고정하여
                # 드론에게는 호버링(Hovering)을, 새에게는 제자리 선회(Circling) 루프 유도
                if event_schedule["hover_start"] <= frame <= event_schedule["hover_end"]:
                    env.x_goal = agent.pos.copy()
                elif frame == event_schedule["hover_end"] + 1:
                    env.x_goal = np.array([base_goal[0] + 150.0, base_goal[1], base_goal[2]])

            # 물리 모델 1스텝 구동 (3D 좌표 변위 계산)
            agent.step(apply_noise=apply_noise)

            # 5️. Early Stopping 예외 제어 (지면 추락 검사)
            if agent.pos[2] <= 0:
                break

            # 2D 관측 데이터 슬라이싱 투영 및 기록
            obs = agent.get_observation(frame_index=frame, apply_noise=apply_noise)
            sim_entry["observations"].append(obs)

        sim_entry["observations"], sim_entry["metadata"]["tracking_drift_profile"] = (
            self._apply_tracking_drift_noise(
                sim_entry["observations"],
                drift_profile,
            )
        )
        sim_entry["observations"], sim_entry["metadata"]["dropout_profile"] = (
            self._apply_dropout_noise(
                sim_entry["observations"],
                dropout_profile,
                rng,
            )
        )

        return sim_entry

    def _sample_frame_length(self, rng: np.random.Generator) -> tuple[str, int]:
        """
        실제 A track 길이 편차를 반영하기 위해 short/medium/long 길이를 섞어 생성한다.
        짧은 track은 C rule filter 경계와 운영 검증에 필요하므로 별도 bucket으로 남긴다.
        """
        bucket = str(
            rng.choice(
                ["short", "medium", "long"],
                p=[0.25, 0.45, 0.30],
            )
        )
        ranges = {
            "short": (60, 120),
            "medium": (121, 240),
            "long": (241, self.max_frames),
        }
        low, high = ranges[bucket]
        return bucket, int(rng.integers(low, high + 1))

    def _sample_bbox_depth_profile(
        self,
        rng: np.random.Generator,
        frame_count: int,
    ) -> dict:
        """
        bbox 크기가 y 거리 변화와 연결되도록 depth 이동 패턴을 샘플링한다.
        """
        mode = str(
            rng.choice(
                ["approaching", "receding", "crossing", "passing_by"],
                p=[0.30, 0.30, 0.25, 0.15],
            )
        )
        profile = {
            "mode": mode,
            "switch_frame": None,
            "near_y": None,
            "far_y": None,
        }

        if mode == "passing_by":
            profile["switch_frame"] = self._ratio_frame(
                rng.uniform(0.42, 0.65),
                frame_count,
            )
            profile["near_y"] = round(float(rng.uniform(50.0, 75.0)), 2)
            profile["far_y"] = round(float(rng.uniform(130.0, 210.0)), 2)

        return profile

    def _sample_start_position(
        self,
        rng: np.random.Generator,
        bbox_depth_profile: dict,
    ) -> list[float]:
        """
        실제 촬영 상황의 다양성을 반영하기 위해 초기 위치 범위를 넓게 샘플링한다.
        A가 넘기는 history는 관측된 객체에서 시작하므로, 초기 x는 화면 안쪽으로 제한한다.
        현재 관측 모델은 goal 기반 화면 스케일을 쓰므로, 시작 z도 보수적으로 넓힌다.
        """
        mode = bbox_depth_profile["mode"]
        if mode in ("approaching", "passing_by"):
            start_y = rng.uniform(125.0, 180.0)
        elif mode == "receding":
            start_y = rng.uniform(55.0, 95.0)
        else:
            start_y = rng.uniform(70.0, 170.0)

        return [
            float(rng.uniform(0.0, 70.0)),
            float(start_y),
            float(rng.uniform(50.0, 105.0)),
        ]

    def _sample_goal_position(
        self,
        rng: np.random.Generator,
        start_pos: list[float],
        bbox_depth_profile: dict,
    ) -> list[float]:
        """
        샘플별 목표 위치를 다양화하되 시작점과 너무 가까운 목표는 피한다.
        가까운 목표는 짧은 회전이나 정지 패턴을 과도하게 만들 수 있다.
        """
        start = np.array(start_pos, dtype=float)
        for _ in range(100):
            goal_y = self._sample_goal_y_for_bbox_mode(
                rng,
                start[1],
                bbox_depth_profile,
            )
            candidate = np.array(
                [
                    rng.uniform(120.0, 600.0),
                    goal_y,
                    rng.uniform(20.0, 180.0),
                ],
                dtype=float,
            )
            if (
                np.linalg.norm(candidate - start) >= 150.0
                and abs(candidate[0] - start[0]) >= 90.0
            ):
                return [float(value) for value in candidate]

        fallback = np.array(
            [
                max(start[0] + 180.0, 180.0),
                self._sample_goal_y_for_bbox_mode(rng, start[1], bbox_depth_profile),
                np.clip(start[2] + rng.uniform(-45.0, 45.0), 20.0, 180.0),
            ],
            dtype=float,
        )
        return [float(value) for value in fallback]

    def _sample_goal_y_for_bbox_mode(
        self,
        rng: np.random.Generator,
        start_y: float,
        bbox_depth_profile: dict,
    ) -> float:
        mode = bbox_depth_profile["mode"]
        if mode == "approaching":
            return float(rng.uniform(50.0, 85.0))
        if mode == "receding":
            return float(rng.uniform(140.0, 220.0))
        if mode == "crossing":
            return float(np.clip(start_y + rng.uniform(-18.0, 18.0), 50.0, 220.0))
        return float(bbox_depth_profile["far_y"])

    def _apply_bbox_depth_waypoint(
        self,
        env: Environment,
        frame: int,
        bbox_depth_profile: dict,
    ) -> None:
        if bbox_depth_profile["mode"] != "passing_by":
            return

        if frame <= bbox_depth_profile["switch_frame"]:
            env.x_goal[1] = bbox_depth_profile["near_y"]
        else:
            env.x_goal[1] = bbox_depth_profile["far_y"]

    def _sample_event_schedule(
        self,
        scenario: str,
        frame_count: int,
        rng: np.random.Generator,
    ) -> dict[str, int | list[int]]:
        """
        시나리오 이벤트가 특정 프레임 번호에 고정되지 않도록 전체 길이 비율로 샘플링한다.
        """
        if scenario == "sudden_dash":
            dash_frame = self._ratio_frame(rng.uniform(0.25, 0.55), frame_count)
            brake_frame = self._ratio_frame(rng.uniform(0.60, 0.85), frame_count)
            brake_frame = max(brake_frame, min(frame_count - 2, dash_frame + 8))
            return {
                "dash_frame": dash_frame,
                "brake_frame": brake_frame,
            }

        if scenario == "sharp_turns":
            turn_frames = self._sample_spaced_frames(
                rng,
                frame_count=frame_count,
                count=3,
                low_ratio=0.20,
                high_ratio=0.82,
                min_gap=6,
            )
            return {"turn_frames": turn_frames}

        if scenario == "multi_mode":
            hover_start = self._ratio_frame(rng.uniform(0.25, 0.50), frame_count)
            hover_end = self._ratio_frame(rng.uniform(0.58, 0.82), frame_count)
            hover_end = max(hover_end, min(frame_count - 3, hover_start + 12))
            return {
                "hover_start": hover_start,
                "hover_end": hover_end,
            }

        return {}

    def _ratio_frame(self, ratio: float, frame_count: int) -> int:
        return int(np.clip(round(frame_count * ratio), 1, max(frame_count - 2, 1)))

    def _sample_spaced_frames(
        self,
        rng: np.random.Generator,
        *,
        frame_count: int,
        count: int,
        low_ratio: float,
        high_ratio: float,
        min_gap: int,
    ) -> list[int]:
        for _ in range(100):
            frames = sorted(
                self._ratio_frame(ratio, frame_count)
                for ratio in rng.uniform(low_ratio, high_ratio, size=count)
            )
            if all(
                current - previous >= min_gap
                for previous, current in zip(frames, frames[1:])
            ):
                return frames

        return [
            self._ratio_frame(ratio, frame_count)
            for ratio in np.linspace(low_ratio, high_ratio, count)
        ]

    def _sample_tracking_drift_profile(
        self,
        rng: np.random.Generator,
        frame_count: int,
        apply_noise: bool,
    ) -> dict:
        """
        추적기가 일정 구간 한 방향으로 서서히 밀렸다가 회복되는 관측 좌표 drift를 계획한다.
        drift 크기는 정규화 화면 좌표 기준이며, 실제 물체 운동과 섞이지 않도록 작게 제한한다.
        """
        if not apply_noise or frame_count < 30:
            return {
                "enabled": False,
                "start": None,
                "peak": None,
                "end": None,
                "recover": False,
                "max_dx": 0.0,
                "max_dy": 0.0,
                "duration": 0,
                "affected_frame_count": 0,
                "clipped_frame_count": 0,
            }

        if rng.random() < 0.25:
            return {
                "enabled": False,
                "start": None,
                "peak": None,
                "end": None,
                "recover": False,
                "max_dx": 0.0,
                "max_dy": 0.0,
                "duration": 0,
                "affected_frame_count": 0,
                "clipped_frame_count": 0,
            }

        drift_start = self._ratio_frame(rng.uniform(0.15, 0.55), frame_count)
        ramp_len = int(rng.integers(10, min(45, max(11, frame_count // 5)) + 1))
        hold_len = int(rng.integers(8, min(55, max(9, frame_count // 4)) + 1))
        recover = bool(rng.random() < 0.70)
        recover_len = (
            int(rng.integers(10, min(50, max(11, frame_count // 5)) + 1))
            if recover
            else 0
        )

        drift_peak = min(drift_start + ramp_len, frame_count - 2)
        drift_end = min(drift_peak + hold_len + recover_len, frame_count - 2)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        magnitude = float(rng.uniform(0.008, 0.035))

        return {
            "enabled": True,
            "start": drift_start,
            "peak": drift_peak,
            "end": drift_end,
            "recover": recover,
            "max_dx": round(float(np.cos(angle) * magnitude), 4),
            "max_dy": round(float(np.sin(angle) * magnitude), 4),
            "duration": int(drift_end - drift_start + 1),
            "affected_frame_count": 0,
            "clipped_frame_count": 0,
        }

    def _apply_tracking_drift_noise(
        self,
        observations: list[dict],
        profile: dict,
    ) -> tuple[list[dict], dict]:
        """
        frame_index를 기준으로 drift offset을 누적 적용한다. bbox와 conf는 건드리지 않는다.
        """
        if not profile["enabled"]:
            return observations, profile

        drifted = []
        affected_frame_count = 0
        clipped_frame_count = 0

        for obs in observations:
            frame_index = int(obs["frame_index"])
            if not profile["start"] <= frame_index <= profile["end"]:
                drifted.append(obs)
                continue

            if frame_index <= profile["peak"]:
                phase = (frame_index - profile["start"]) / max(profile["peak"] - profile["start"], 1)
                scale = np.clip(phase, 0.0, 1.0)
            elif profile["recover"]:
                phase = (frame_index - profile["peak"]) / max(profile["end"] - profile["peak"], 1)
                scale = 1.0 - np.clip(phase, 0.0, 1.0)
            else:
                scale = 1.0

            raw_cx = obs["cx"] + (profile["max_dx"] * scale)
            raw_cy = obs["cy"] + (profile["max_dy"] * scale)
            cx = float(np.clip(raw_cx, 0.0, 1.0))
            cy = float(np.clip(raw_cy, 0.0, 1.0))

            updated_obs = obs.copy()
            updated_obs["cx"] = round(cx, 4)
            updated_obs["cy"] = round(cy, 4)
            drifted.append(updated_obs)

            affected_frame_count += 1
            if raw_cx != cx or raw_cy != cy:
                clipped_frame_count += 1

        updated_profile = profile.copy()
        updated_profile["affected_frame_count"] = affected_frame_count
        updated_profile["clipped_frame_count"] = clipped_frame_count
        return drifted, updated_profile

    def _sample_dropout_profile(
        self,
        rng: np.random.Generator,
        frame_count: int,
        apply_noise: bool,
    ) -> dict:
        """
        실제 A 산출물처럼 일부 프레임은 history에서 사라지도록 dropout 계획을 만든다.
        ideal 데이터는 연속 프레임 유지, noisy 데이터만 gap/dropout을 갖는다.
        """
        if not apply_noise or frame_count < 4:
            return {
                "enabled": False,
                "random_dropout_rate": 0.0,
                "random_dropout_frames": [],
                "burst_ranges": [],
                "low_conf_ranges": [],
                "dropped_frame_count": 0,
                "low_conf_frame_count": 0,
                "missing_ratio": 0.0,
            }

        candidate_frames = np.arange(1, frame_count - 1)
        random_dropout_rate = float(rng.uniform(0.02, 0.08))
        random_count = int(
            np.clip(
                round(frame_count * random_dropout_rate),
                1,
                len(candidate_frames),
            )
        )
        random_dropout_frames = sorted(
            int(frame)
            for frame in rng.choice(candidate_frames, size=random_count, replace=False)
        )

        burst_ranges = []
        max_burst_len = min(20, max(3, frame_count // 8))
        burst_count = int(rng.choice([0, 1, 2], p=[0.25, 0.60, 0.15]))
        for _ in range(burst_count):
            if frame_count <= 8:
                break
            length = int(rng.integers(3, max_burst_len + 1))
            start = int(rng.integers(1, max(frame_count - length, 2)))
            burst_ranges.append(
                {
                    "start": start,
                    "end": min(start + length - 1, frame_count - 2),
                }
            )

        low_conf_ranges = []
        low_conf_count = int(rng.choice([0, 1], p=[0.35, 0.65]))
        max_low_conf_len = min(45, max(8, frame_count // 4))
        for _ in range(low_conf_count):
            if frame_count <= 12:
                break
            length = int(rng.integers(8, max_low_conf_len + 1))
            start = int(rng.integers(1, max(frame_count - length, 2)))
            low_conf_ranges.append(
                {
                    "start": start,
                    "end": min(start + length - 1, frame_count - 2),
                    "conf_min": round(float(rng.uniform(0.45, 0.55)), 2),
                    "conf_max": round(float(rng.uniform(0.62, 0.72)), 2),
                }
            )

        return {
            "enabled": True,
            "random_dropout_rate": round(random_dropout_rate, 4),
            "random_dropout_frames": random_dropout_frames,
            "burst_ranges": burst_ranges,
            "low_conf_ranges": low_conf_ranges,
            "dropped_frame_count": 0,
            "low_conf_frame_count": 0,
            "missing_ratio": 0.0,
        }

    def _apply_dropout_noise(
        self,
        observations: list[dict],
        profile: dict,
        rng: np.random.Generator,
    ) -> tuple[list[dict], dict]:
        """
        frame_index/timestamp_ms는 보존하고 관측 row만 제거하여 실제 track gap을 만든다.
        """
        if not profile["enabled"]:
            return observations, profile

        available_frames = {int(obs["frame_index"]) for obs in observations}
        drop_frames = set(profile["random_dropout_frames"])
        for burst_range in profile["burst_ranges"]:
            drop_frames.update(range(burst_range["start"], burst_range["end"] + 1))

        drop_frames &= available_frames
        retained = []
        low_conf_frame_count = 0

        for obs in observations:
            frame_index = int(obs["frame_index"])
            if frame_index in drop_frames:
                continue

            updated_obs = obs.copy()
            for low_conf_range in profile["low_conf_ranges"]:
                if low_conf_range["start"] <= frame_index <= low_conf_range["end"]:
                    updated_obs["conf"] = round(
                        float(
                            min(
                                updated_obs["conf"],
                                rng.uniform(
                                    low_conf_range["conf_min"],
                                    low_conf_range["conf_max"],
                                ),
                            )
                        ),
                        2,
                    )
                    low_conf_frame_count += 1
                    break
            retained.append(updated_obs)

        updated_profile = profile.copy()
        updated_profile["dropped_frame_count"] = len(observations) - len(retained)
        updated_profile["low_conf_frame_count"] = low_conf_frame_count
        updated_profile["missing_ratio"] = round(
            updated_profile["dropped_frame_count"] / max(len(observations), 1),
            4,
        )
        return retained, updated_profile

    def execute_batch_pipeline(self, bird_samples_per_species: int = 50, drone_samples_per_model: int = 150, apply_noise: bool = False) -> str:
        """
        4대 세분화 시나리오 전체를 순회하며 새 600개, 드론 600개(총 1,200개)의 유효 데이터셋을 대량 생산합니다.
        """
        scenarios = ["steady_cruise", "sudden_dash", "sharp_turns", "multi_mode"]
        birds = ["pigeon", "seagull", "falcon"]
        results = []

        # [Sim2Real 추가] 짧은 track도 학습/운영 필터 검증에 쓰기 위해 최소 허들을 낮춘다.
        min_frame_cutoff = 60

        print("=" * 65)
        print(f" [Phase 1: Batch Runner] v2 고도화 공정 가동 (Noise 주입: {apply_noise})")
        print("=" * 65)

        # 외부 루프 (Scenario Loop)
        for scenario in scenarios:
            
            # 내부 루프 1: 조류 군집 데이터 획득 (시나리오당 종별 50개 샘플)
            for species in birds:
                valid_count = 0
                idx = 1
                while valid_count < bird_samples_per_species:
                    sim_data = self._run_single_simulation(scenario, "bird", species, idx, apply_noise=apply_noise)
                    # 데이터 유효 품질 방어벽 (최소 50프레임 이상 비행한 데이터만 인정)
                    if len(sim_data["observations"]) >= min_frame_cutoff:
                        results.append(sim_data)
                        valid_count += 1
                    idx += 1

            # 내부 루프 2: 드론 군집 데이터 획득 (시나리오당 150개 샘플로 클래스 균형 추정 증폭)
            valid_count = 0
            idx = 1
            while valid_count < drone_samples_per_model:
                sim_data = self._run_single_simulation(scenario, "drone", "quadcopter", idx, apply_noise=apply_noise)
                if len(sim_data["observations"]) >= min_frame_cutoff:
                    results.append(sim_data)
                    valid_count += 1
                idx += 1
        
        # 6️. 데이터 대량 생산 완료 후 pkl 직렬화 물리 저장
        # [수정] 구버전(v1) 자산을 훼손하지 않기 위해 파일명 버전 분리 정책 수립
        pkl_name = "batch_raw_trajectories_v2.pkl" if apply_noise else "batch_raw_trajectories.pkl"
        output_path = os.path.join(self.output_dir, pkl_name)
        
        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        return output_path

class CoreFeatureExtractor:
    def __init__(self, data_dir: str, version: str = "v1"):
        """
        :param version: 'v1' (Ideal 기존형) 또는 'v2' (Noisy 실측형) 지정
        """
        self.data_dir = data_dir
        # 버전에 따른 입출력 확장자 스키마 동적 매핑
        if version == "v2":
            self.input_path = os.path.join(data_dir, "batch_raw_trajectories_v2.pkl")
            self.output_path = os.path.join(data_dir, "simulation_features_v2.csv")
        else:
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
        print(f"[Phase 2 완료] CSV 피처 매트릭스 테이블 발행 완료.")
        print(f"저장 경로: {self.output_path}")
        print("=" * 65)
        return self.output_path

if __name__ == "__main__":

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    target_data_dir = os.path.join(current_script_dir, "data")

    # -----------------------------------------------------------------
    # [실험 관리 트랙] 버전 2 (Sim2Real 고도화 노이즈 데이터셋 배포)
    # -----------------------------------------------------------------
    # 1. 50 FPS 가변 윈도우 및 노이즈가 주입된 v2 원천 pkl 팩토리 가동
    runner_v2 = BatchRunner(output_dir=target_data_dir, fps=30)
    runner_v2.execute_batch_pipeline(bird_samples_per_species=50, drone_samples_per_model=150, apply_noise=True)
    
    # 2. v2 전용 특징량 압축기 가동 -> simulation_features_v2.csv 최종 발행
    extractor_v2 = CoreFeatureExtractor(data_dir=target_data_dir, version="v2")
    extractor_v2.extract_features()
