import numpy as np
import json
import os
from datetime import datetime
from abc import ABC, abstractmethod


# 조류의 생물학적 특성을 반영한 설정 사전
"""
    논문 Table 1 및 종별 특성을 반영한 설정 사전
    s_star : 선호 속도
    k_star : 속도 적응률
    phi_max : 최대 뱅크각
    sigma_s : 속도 노이즈
    sigma_phi : 뱅크각 노이즈
    k_g : 목표 추적 강도
"""
SPECIES_CONFIG = {
    "pigeon": { "s_star": 12.0, "k_s" : 1.5, "phi_max": 45, "sigma_s": 0.8, "sigma_phi": 0.12, "k_g": 1.5,
               "real_w": 0.6, "real_h": 0.2 },
    "seagull": { "s_star": 15.0, "k_s" : 0.8, "phi_max": 35, "sigma_s": 0.4, "sigma_phi": 0.06, "k_g": 1.0, 
                "real_w": 1.2, "real_h": 0.3 },
    "falcon":  { "s_star": 25.0, "k_s" : 1.2, "phi_max": 60, "sigma_s": 0.3, "sigma_phi": 0.05, "k_g": 2.5,
                "real_w": 1.0, "real_h": 0.25 }
}

# 드론의 기계적 특성을 반영한 설정 사전
DRONE_CONFIG = {
        "quadcopter": {
            "s_star": 15.0,      # 선호 속도 (촬영용 드론 평균)
            "a_max": 8.0,        # 최대 가속도 (m/s^2)
            "k_a": 2.0,          # 가속도 반응 계수
            "sigma_s": 0.05,     # 속도 노이즈 (조류의 1/10 수준으로 매우 낮음)
            "sigma_u": 0.02,     # 방향 노이즈 (매우 낮음)
            "k_h": 0.5,           # 고도 제어 강도
            "real_w": 0.5, "real_h": 0.15
        }
    }

class Environment:
    def __init__(self, fps=30, wind_speed=1.0, gust_intensity=0.5, goal_pos=None):
        """
        시나리오 제어를 위한 환경 클래스
        : fps: 초당 프레임 수
        : param wind_speed: 기본 풍속 (m/s)
        : param gust_intensity: 바람의 강도 (sigma_w) 
        : param goal_pos: 비행체가 향할 고정 목표 지점 [x, y, z]
        """
        self.fps = fps

        # 한 프레임과 다음 프레임 사이의 시간 간격(dt)
        self.dt = 1.0 / fps
        
        # 1. 엄격한 바람 조건 설정
        # 고정된 방향의 기본 풍속 (x축 방향으로 설정)
        self.wind_base = np.array([wind_speed, 0.0, 0.0]) 
        self.gust_std = gust_intensity  # 시나리오에 따라 조절 가능
        self.gust_tau = 1.2             # 돌풍의 시간 상관성 (논문 기준값) 
        
        # 2. OU 프로세스 초기화
        self.current_gust = np.array([0.0, 0.0, 0.0])

        # 3. 시나리오용 목표 지점 (Waypoint) 및 선호 고도 (h*)
        # 지정하지 않으면 기본 위치로 설정
        if goal_pos is None:
            self.x_goal = np.array([300.0, 50.0, 50.0]) 
        else:
            self.x_goal = np.array(goal_pos)
            
        self.h_star = self.x_goal[2]  # 선호 비행 고도 

    def update_and_get_wind(self):
        """
        Ornstein-Uhlenbeck(OU) 프로세스를 이용한 풍속 계산
        이전 프레임의 바람 상태를 유지하면서 새로운 변화량을 더함
        """
        # 논문 수식 기반: dW = -(1/tau)*W*dt + sigma*sqrt(dt)*N(0,1)
        dw = np.random.normal(0, self.gust_std, 3)
        self.current_gust += (-self.current_gust / self.gust_tau) * self.dt + dw * np.sqrt(self.dt)
        
        # 기본 풍속 + 동적 돌풍 
        return self.wind_base + self.current_gust

class BaseAgent(ABC) :
    def __init__(self, env, start_pos=None, start_speed = 10) :
        """
        비행체(Agent)의 공통 속성을 정의하는 기저 클래스
        : param env: Environment 객체 (물리적 환경 정보 전달)
        : param start_pos: 초기 위치 [x, y, z]
        : param start_speed: 초기 대기 속도 (m/s)
        """

        self.env = env
        self.dt = env.dt

        # 1. 상태 변수 초기화
        # 초기 위치 없는 경우 [0, 0, h*] 부근에서 시작
        if start_pos is None : 
            self.pos = np.array([0.0, 0.0, env.h_star])
        else: 
            self.pos = np.array(start_pos, dtype=float)

        # 2. 방향 및 속도 초기화
        # u: 단위 헤딩 벡터 (Unit Heading Vector)
        self.u = np.array([1.0, 0.0, 0.0]) # 초기에는 x 방향을 바라봄
        # s: 대기 속도 (Airspeed)
        self.s = float(start_speed)

        self.history = []

        # 3. 투영을 위한 물리적 제원 (자식 클래스에서 오버라이드 가능)
        # 실제 서비스에서 bbox 크기(w, h)를 결정하는 기준 (단위: m)
        self.real_width = 0.5   # 객체의 실제 가로 크기
        self.real_height = 0.2  # 객체의 실제 세로 크기

    def get_unit_vector_to_goal(self):
        """
        현재 위치에서 목표 지점(Environment.x_goal)을 향하는 단위 벡터 계산
        목표 추적(Target Tarcking)을 위한 기초 연산
        """
        vec = self.env.x_goal - self.pos
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return self.u # 이미 도착했다면 현재 방향 유지
        return vec / dist
    
    @abstractmethod
    def step(self):
        """
        매 프레임마다 호출되어 비행체의 물리 상태를 업데이트하는 함수
        자식 클래스(BirdDyn, DroneDyn)에서 반드시 구현해야 함 
        """
        pass
    
    def update_position(self, v_ground) :
        """
        계산된 지면 속도(v_ground)를 이용해 위치를 업데이트하고 기록
        x(t + dt) = x(t) + v_ground * dt
        """
        self.pos += v_ground * self.dt
        self.history.append(self.pos.copy())
        return self.pos
    
    def get_observation(self, frame_index):
        """
        [Simplified Lateral Mapping]
        - CX: X축(전진 거리)을 화면 가로에 매핑 
        - CY: Z축(고도)을 화면 세로에 매핑 
        - BBox: Y축(관측자와의 거리)에 반비례하여 크기 결정 
        """
        # 1. 현재 3D 위치 좌표 가져오기 
        x, y, z = self.pos
        
        # 2. 화면 가시 범위 설정 (고정된 도화지 크기)
        # X축(전진) 130m, Z축(고도) 110m를 0.0 ~ 1.0 범위에 매핑합니다.
        max_x = 130.0
        max_z = 110.0

        # 3. 좌표 선형 매핑 (Normalization) [cite: 18]
        # cx: 0.0(왼쪽 끝) ~ 1.0(오른쪽 끝) 
        cx = x / max_x
        
        # cy: 0.0(하늘/Top) ~ 1.0(지면/Bottom) 
        # 고도(z)가 높을수록 0에 가깝게 계산합니다.
        cy = 1.0 - (z / max_z)

        # 4. 바운딩 박스 크기 계산 (물리적 거리감 반영) 
        # 관측자와의 거리(Y)가 80m 이상인 컨셉을 유지합니다.
        distance = max(y, 50.0) 
        
        # focal_constant를 1.0으로 설정하여 80m 밖의 실제 크기 비중을 구현합니다.
        # 박스가 너무 작아 소실되는 것을 방지하기 위해 최소 크기(0.005)를 보장합니다.
        focal_constant = 10.0
        w = (self.real_width / distance) * focal_constant
        h = (self.real_height / distance) * focal_constant

        # 5. 서비스 규격(JSON) 데이터 반환 
        return {
            "frame_index": frame_index,
            "timestamp_ms": int(frame_index * (1000 / self.env.fps)),
            "cx": round(float(np.clip(cx, 0.0, 1.0)), 4),
            "cy": round(float(np.clip(cy, 0.0, 1.0)), 4),
            "w": round(float(np.clip(w, 0.005, 0.2)), 4),
            "h": round(float(np.clip(h, 0.005, 0.2)), 4),
            "conf": round(float(np.random.uniform(0.92, 0.99)), 2)
        }


class BirdDyn(BaseAgent):
    def __init__(self, env, species="pigeon", **kwargs):
        super().__init__(env, **kwargs)

        # 1. 종별 파라미터 로드 (기본값 : 비둘기)
        config = SPECIES_CONFIG.get(species.lower(), SPECIES_CONFIG["pigeon"])

        self.species = species
        self.s_star = config["s_star"]
        self.k_s = config["k_s"]
        self.phi_max = np.radians(config["phi_max"])
        self.sigma_s = config["sigma_s"]
        self.sigma_phi = config["sigma_phi"]
        self.k_g = config["k_g"]

        self.real_width = config["real_w"]
        self.real_height = config["real_h"]

        # 2. 공통 파라미터 (논문 기준값)
        self.k_phi = 4.0      # 뱅크각 반응 속도
        self.k_h = 0.02       # 고도 유지 강도
        
        # 3. 상태 변수 초기화
        self.phi = 0.0        # 현재 뱅크각 (Roll)
        self.gamma = 0.0      # 현재 피치각 (Pitch)
    
    def step(self):
        """
        매 프레임마다 조류의 물리 상태를 업데이트
        """
        # --- (1) 대기 속도(s) 업데이트: s_dot = k_s * (s_star - s) + noise ---
        ds = self.k_s * (self.s_star - self.s) * self.dt
        noise_s = self.sigma_s * np.random.normal(0, 1) * np.sqrt(self.dt)
        self.s = max(2.0, self.s + ds + noise_s)

        # --- (2) 목표 방향 분석 및 뱅크각 결정 ---
        u_goal = self.get_unit_vector_to_goal()
        
        # 수평 오차(Yaw error) 계산
        target_yaw = np.arctan2(u_goal[1], u_goal[0])
        current_yaw = np.arctan2(self.u[1], self.u[0])
        yaw_error = (target_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi

        # 뱅크각 결정: 목표 방향으로 몸을 기울임
        phi_target = np.arctan(self.s * self.k_g * yaw_error / 9.81)
        phi_target = np.clip(phi_target, -self.phi_max, self.phi_max)
        
        d_phi = self.k_phi * (phi_target - self.phi) * self.dt
        noise_phi = self.sigma_phi * np.random.normal(0, 1) * np.sqrt(self.dt)
        self.phi += d_phi + noise_phi

        # --- (3) 방향 벡터(u) 회전: psi_dot = g * tan(phi) / s ---
        yaw_rate = (9.81 * np.tan(self.phi)) / self.s
        
        # 고도 유지 로직 (수직 가속도)
        h_error = self.env.h_star - self.pos[2]
        pitch_rate = self.k_h * h_error
        
        # 방향 업데이트
        new_yaw = current_yaw + yaw_rate * self.dt
        self.gamma += pitch_rate * self.dt
        self.gamma = np.clip(self.gamma, -np.radians(25), np.radians(25))
        
        # 새로운 단위 헤딩 벡터 설정
        self.u = np.array([
            np.cos(new_yaw) * np.cos(self.gamma),
            np.sin(new_yaw) * np.cos(self.gamma),
            np.sin(self.gamma)
        ])

        # --- (4) 최종 위치 업데이트 (바람 반영) ---
        wind = self.env.update_and_get_wind()
        v_ground = self.s * self.u + wind
        return self.update_position(v_ground)


class DroneDyn(BaseAgent):
    def __init__(self, env, model='quadcopter', **kwargs):
        super().__init__(env, **kwargs)
    
        # 1. 드론 파라미터 로드
        config = DRONE_CONFIG.get(model.lower(), DRONE_CONFIG["quadcopter"])

        self.model = model
        self.s_star = config["s_star"]      # 선호 대기 속도
        self.a_max = config["a_max"]        # 모터 출력 한계
        self.k_a = config["k_a"]            # 제어기 민감도
        self.sigma_s = config["sigma_s"]    # 대기 속도 노이즈
        self.sigma_u = config["sigma_u"]    # 헤딩 노이즈
        self.k_h = config["k_h"]            # 고도 유지 강도
        
        # 수정 포인트: 실물 크기 반영 (Review 반영)
        self.real_width = config["real_w"]
        self.real_height = config["real_h"]

        # 2. 상태 변수 초기화 
        # 드론은 현재 대기 속도 벡터(v_air)를 직접 관리
        self.v_air = self.s * self.u

    def step(self):
        """
        매 프레임마다 드론의 가속도 제어 및 위치 업데이트
        """ 
        # --- (1) 목표 대기 속도 벡터 계산 ---
        # 드론은 현재 위치에서 목표물을 향해 s_star의 속도로 가고 싶어함
        u_goal = self.get_unit_vector_to_goal()
        
        # 고도 보정 로직 (z축 속도 성분 조정)
        h_error = self.env.h_star - self.pos[2]
        u_goal[2] += self.k_h * (h_error / self.s_star)
        u_goal = u_goal / np.linalg.norm(u_goal) # 재정규화
        
        v_target = self.s_star * u_goal

        # --- (2) 가속도 제어 (Acceleration Control) ---
        # 현재 속도와 목표 속도의 차이를 메우기 위한 가속도 계산
        # a = k_a * (v_target - v_air)
        accel = self.k_a * (v_target - self.v_air)
        
        # 물리적 한계(모터 출력)에 따른 가속도 제한(Clipping)
        accel_mag = np.linalg.norm(accel)
        if accel_mag > self.a_max:
            accel = (accel / accel_mag) * self.a_max

        # --- (3) 속도 벡터 업데이트 [Euler Integration] ---
        # v_air = v_air + accel * dt + noise
        noise_v = self.sigma_s * np.random.normal(0, 1, 3) * np.sqrt(self.dt)
        self.v_air += accel * self.dt + noise_v
        
        # 업데이트된 v_air로부터 현재 속도(s)와 방향(u) 추출
        self.s = np.linalg.norm(self.v_air)
        self.u = self.v_air / (self.s + 1e-6)

        # --- (4) 최종 지면 속도(v_ground) 계산 및 이동 ---
        # 지면 속도 = 비행체의 추진 속도 + 환경풍
        wind = self.env.update_and_get_wind()
        v_ground = self.v_air + wind
        
        return self.update_position(v_ground)
        # 위치 업데이트 및 이력 저장
        # new_pos = self.update_position(v_ground)
        # self.history.append(new_pos.copy())

        # return new_pos

class TrajectoryGenerator:
    def __init__(self, env, output_dir="research/output"):
        """
        시뮬레이션 에이전트를 구동하고 결과를 서비스 규격 JSON으로 내보내는 클래스
        : param env: Environment 객체
        : param output_dir: JSON 파일이 저장될 경로
        """
        self.env = env
        self.output_dir = output_dir

        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def generate(self, agent, num_frames=150, track_id=1):
        """
        특정 에이전트를 시뮬레이션하여 궤적 데이터 생성
        : param agent: BirdDyn 또는 DroneDyn 인스턴스
        : param num_frames: 생성할 프레임 수 (기본 150fps = 5초)
        : param track_id: 객체 식별 번호
        : return: 서비스 규격에 맞춘 Dictionary 데이터 
        """
        history_2d = []

        # 1. 시뮬레이션 루프 수행
        for i in range(num_frames):
            # 물리 상태 업데이트 (3D)
            agent.step()
            
            # 현재 상태 관측 (3D -> 2D 투영)
            # BaseAgent에 구현한 get_observation 호출
            observation = agent.get_observation(frame_index=i)
            history_2d.append(observation)
        
        # 2. 품질 메트릭 계산 (Part B 서비스 규격 반영)
        total_conf = sum(p['conf'] for p in history_2d)
        mean_conf = round(total_conf / len(history_2d), 3)

        # 3. 최종 JSON 구조체 매핑
        # agent의 클래스명과 종/모델 정보를 조합하여 식별자 생성
        agent_type = agent.__class__.__name__
        sub_type = getattr(agent, 'species', getattr(agent, 'model', 'unknown'))
        
        data = {
            "track_id": track_id,
            "source_video_id": f"SIM_{agent_type.upper()}_{sub_type.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "stabilization": {
                "applied": True,
                "method": "simulation_ideal"
            },
            "history": history_2d,
            "quality": {
                "num_points": len(history_2d),
                "mean_conf": mean_conf,
                "missing_ratio": 0.0,
                "track_stability": "good" if mean_conf > 0.9 else "fair"
            }
        }
        return data

    def save(self, data, filename=None):
        """
        생성된 데이터를 JSON 파일로 물리적 저장
        """
        if filename is None:
            filename = f"{data['source_video_id']}_track{data['track_id']}.json"
        
        file_path = os.path.join(self.output_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 데이터 생성 완료: {file_path}")
        return file_path