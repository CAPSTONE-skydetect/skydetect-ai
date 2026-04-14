import numpy as np
import json
from datetime import datetime

import numpy as np

class Environment:
    def __init__(self, fps=30, wind_speed=1.0, gust_intensity=0.5, goal_pos=None):
        """
        시나리오 제어를 위한 환경 클래스
        :param wind_speed: 기본 풍속 (m/s)
        :param gust_intensity: 돌풍의 강도 (sigma_w) [cite: 518]
        :param goal_pos: 비행체가 향할 고정 목표 지점 [x, y, z]
        """
        self.dt = 1.0 / fps
        
        # 1. 엄격한 바람 조건 설정
        # 고정된 방향의 기본 풍속 (x축 방향으로 설정)
        self.wind_base = np.array([wind_speed, 0.0, 0.0]) 
        self.gust_std = gust_intensity  # 시나리오에 따라 조절 가능
        self.gust_tau = 1.2             # 돌풍의 시간 상관성 (논문 기준값) [cite: 715]
        
        # 2. OU 프로세스 초기화
        self.current_gust = np.array([0.0, 0.0, 0.0])

        # 3. 시나리오용 목표 지점 (Waypoint) 및 선호 고도 (h*)
        # 지정하지 않으면 기본 위치로 설정
        if goal_pos is None:
            self.x_goal = np.array([300.0, 50.0, 50.0]) 
        else:
            self.x_goal = np.array(goal_pos)
            
        self.h_star = self.x_goal[2]  # 선호 비행 고도 [cite: 713]

    def update_and_get_wind(self):
        """
        Ornstein-Uhlenbeck(OU) 프로세스를 이용한 풍속 계산 [cite: 715-717]
        이전 프레임의 바람 상태를 유지하면서 새로운 변화량을 더함
        """
        # 논문 수식 기반: dW = -(1/tau)*W*dt + sigma*sqrt(dt)*N(0,1)
        dw = np.random.normal(0, self.gust_std, 3)
        self.current_gust += (-self.current_gust / self.gust_tau) * self.dt + dw * np.sqrt(self.dt)
        
        # 기본 풍속 + 동적 돌풍 [cite: 717]
        return self.wind_base + self.current_gust