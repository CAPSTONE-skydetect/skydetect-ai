import numpy as np
import json
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np

SPECIES_CONFIG = {
    """
    논문 Table 1 및 종별 특성을 반영한 설정 사전
    s_star : 선호 속도
    k_star : 속도 적응률
    phi_max : 최대 뱅크각
    sigma_s : 속도 노이즈
    sigma_phi : 뱅크각 노이즈
    k_g : 목표 추적 강도
    """
    "pigeon": { "s_star": 12.0, "k_s" : 1.5, "phi_max": 45, "sigma_s": 0.8, "sigma_phi": 0.12, "k_g": 1.5 },
    "seagull": { "s_star": 15.0, "k_s" : 0.8, "phi_max": 35, "sigma_s": 0.4, "sigma_phi": 0.06, "k_g": 1.0 },
    "falcon":  { "s_star": 25.0, "k_s" : 1.2, "phi_max": 60, "sigma_s": 0.3, "sigma_phi": 0.05, "k_g": 2.5 }
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
        return self.pos

class BirdDyn(BaseAgent):
    def __init__(self, env, species="pigeon", **kwargs):
        super().__init__(env, **kwargs)

        # 1. 종별 파라미터 로드 (기본값 : 비둘기)
        config = self.SPECIES_CONFIG.get(species.lower(), self.SPECIES_CONFIG["pigeon"])

        self.species = species
        self.s_star = config["s_star"]
        self.k_s = config["k_s"]
        self.phi_max = np.radians(config["phi_max"])
        self.sigma_s = config["sigma_s"]
        self.sigma_phi = config["sigma_phi"]
        self.k_g = config["k_g"]

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
        
        new_pos = self.update_position(v_ground)
        self.history.append(new_pos.copy())

        return new_pos
        