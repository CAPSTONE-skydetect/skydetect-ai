import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import ttest_ind, norm

class StatisticalEngine :
    def __init__(self, data_dir: str = "research/data", output_dir: str = "research/output"):
        """
        Phase 3 : Statistical Engine 핵심 클래스
        :param data_dir: simulation_features.csv가 위치한 경로
        :param output_dir: stat_summary.json을 저장할 목적지 경로
        """
        self.input_path = os.path.join(data_dir, "simulation_features.csv")
        self.output_dir = output_dir
        self.output_path = os.path.join(output_dir, "stat_summary.json")

        # 출력 디렉토리 사전 확보
        os.makedirs(self.output_dir, exist_ok=True)

    def _calculate_overlap_percentage(self, cohens_d: float) -> float:
        """
        Cohen's d를 기반으로 두 가우시안 분포의 누적분포함수(CDF) 상 중첩 비율(%)을 추정
        """
        overlap = 2 * norm.cdf(-abs(cohens_d) / 2.0)
        return round(float(overlap * 100), 2)

    def run_analysis(self) -> str:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"입력 데이터셋이 {self.input_path}에 존재하지 않습니다. Phase 2를 먼저 가동해주세요.")
        
        # 1. 데이터 로드
        df = pd.read_csv(self.input_path)
        features = ["v_mean", "v_std", "a_mean", "heading_change_ratio", "maneuverability_sigma"]

        bird_df = df[df["label"] == "bird"]
        drone_df = df[df["label"] == "drone"]

        metrics_summary = {}
        all_features_valid = True
        max_cohens_d = -1.0
        top_discriminant_feature = ""