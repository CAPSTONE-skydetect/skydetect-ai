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

        # 2. 핵심 제어 로직 (Feature별 연산 루프)
        for f_name in features:
            b_data = bird_df[f_name].to_numpy()
            d_data = drone_df[f_name].to_numpy()

            # 기술 통계량 산출 (표본 표준편차 ddof=1 적용)
            descriptive = {
                "bird": {
                    "mean": round(float(np.mean(b_data)), 4),
                    "std": round(float(np.std(b_data, ddof=1)), 4),
                    "min": round(float(np.min(b_data)), 4),
                    "max": round(float(np.max(b_data)), 4)
                },
                "drone": {
                    "mean": round(float(np.mean(d_data)), 4),
                    "std": round(float(np.std(d_data, ddof=1)), 4),
                    "min": round(float(np.min(d_data)), 4),
                    "max": round(float(np.max(d_data)), 4)
                }
            }

            # 독립표본 T-검정 (Welch's T-test, 이분산 가정)
            t_stat, p_val = ttest_ind(b_data, d_data, equal_var=False)
            
            is_significant = bool(p_val < 0.05)
            if not is_significant:
                all_features_valid = False

            hypothesis_test = {
                "t_statistic": round(float(t_stat), 4) if not np.isnan(t_stat) else 0.0,
                "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                "is_significant": is_significant
            }
