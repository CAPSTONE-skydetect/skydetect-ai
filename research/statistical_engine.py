# 실행 가이드 : skydetect_ai/에서 -> python research/statistical_engine.py
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import ttest_ind, norm

class StatisticalEngine:
    def __init__(self, data_dir: str = "research/data", output_dir: str = "research/output"):
        """
        Phase 3: Statistical Engine 핵심 클래스
        :param data_dir: simulation_features.csv 가 위치한 경로
        :param output_dir: stat_summary.json 을 저장할 목적지 경로
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

            # [A] 기술 통계량 산출 ( 표본 표준편차 ddof=1 적용 )
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

            # [B] 독립표본 T-검정 (Welch's T-test, 이분산 가정)
            t_stat, p_val = ttest_ind(b_data, d_data, equal_var=False)
            
            is_significant = bool(p_val < 0.05)
            if not is_significant:
                all_features_valid = False

            hypothesis_test = {
                "t_statistic": round(float(t_stat), 4) if not np.isnan(t_stat) else 0.0,
                "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                "is_significant": is_significant
            }

            # [C] Cohen's d 및 분포 중첩도 산출
            n1, n2 = len(b_data), len(d_data)
            s1, s2 = descriptive["bird"]["std"], descriptive["drone"]["std"]
            
            pooled_std = np.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / (n1 + n2 - 2))
            cohens_d = abs(descriptive["bird"]["mean"] - descriptive["drone"]["mean"]) / (pooled_std + 1e-6)
            cohens_d = round(float(cohens_d), 4)
            
            if cohens_d > max_cohens_d:
                max_cohens_d = cohens_d
                top_discriminant_feature = f_name

            overlap_pct = self._calculate_overlap_percentage(cohens_d)

            separation = {
                "cohens_d": cohens_d,
                "overlap_percentage_estimate": overlap_pct
            }

            metrics_summary[f_name] = {
                "descriptive": descriptive,
                "hypothesis_test": hypothesis_test,
                "separation": separation
            }

        # 3. Output 데이터 구조화
        output_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_samples": {
                "bird": int(len(bird_df)),
                "drone": int(len(drone_df))
            },
            "metrics_summary": metrics_summary,
            "engine_conclusion": {
                "top_discriminant_feature": top_discriminant_feature,
                "all_features_valid": all_features_valid
            }
        }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print("=" * 65)
        print(f"[Phase 3 완료] Welch's T-test 및 Cohen's d 통계 연산 성공.")
        print(f" 출력 파일 저장 경로: {self.output_path}")
        print(f" 최고 변별력 피처: {top_discriminant_feature} (Cohen's d: {max_cohens_d})")
        print("=" * 65)
        return self.output_path

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    engine = StatisticalEngine(
        data_dir=os.path.join(current_dir, "data"),
        output_dir=os.path.join(current_dir, "output")
    )
    engine.run_analysis()