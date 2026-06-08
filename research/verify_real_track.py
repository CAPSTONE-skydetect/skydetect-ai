# research/verify_real_track.py
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

class RealTrackVerifier:
    def __init__(self, data_dir: str = "research/data", output_dir: str = "research/output"):
        """
        Phase 5: Real Track Verification 핵심 클래스
        :param data_dir: selected_track.json 및 stat_summary.json 이 위치한 경로
        :param output_dir: 최종 검증 보고서 및 C파트 인도용 CSV를 저장할 경로
        """
        self.input_json_path = os.path.join(data_dir, "selected_track.json")
        self.stat_summary_path = os.path.join(output_dir, "stat_summary.json")
        
        self.output_report_path = os.path.join(output_dir, "verification_report.json")
        self.output_csv_path = os.path.join(output_dir, "real_track_features.csv")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_track_data(self) -> list:
        """
        JSON 내부 구조를 자동 탐색하여 프레임 시계열 리스트를 강건하게 추출
        """
        if not os.path.exists(self.input_json_path):
            raise FileNotFoundError(f"실측 데이터 파일이 {self.input_json_path}에 존재하지 않습니다. 로컬 배치를 확인해주세요.")
            
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            root_data = json.load(f)
            
        # 딕셔너리 내부에서 실제 프레임 데이터가 담긴 리스트를 동적 탐색
        if isinstance(root_data, dict):
            for key, value in root_data.items():
                if isinstance(value, list) and len(value) > 0 and "cx" in value[0]:
                    return value
        elif isinstance(root_data, list):
            return root_data
            
        raise ValueError("selected_track.json 내부에서 유효한 'cx' 기반의 트랙 리스트를 찾을 수 없습니다.")

    def extract_real_features(self, track_data: list) -> dict:
        """
        [핵심] 50 FPS (dt = 0.02s) 타임스텝 보정을 적용한 5대 코어 피처 추출 로직
        """
        # 50 FPS 동기화를 위한 시간 오프셋 강제 설정 (\u0394t = 0.02초)
        dt = 0.02 
        
        cx = np.array([p["cx"] for p in track_data])
        cy = np.array([p["cy"] for p in track_data])
        w = np.array([p["w"] for p in track_data])
        
        # 프레임 간 격차 미분 연산
        dcx = np.diff(cx)
        dcy = np.diff(cy)
        
        # 1. 속도 계산 (몸길이 대비 상대 속도 스케일링 반영)
        distances = np.sqrt(dcx**2 + dcy**2)
        w_safeguard = w[1:] + 1e-6 # 분모 0 방지 방어벽
        velocities = distances / (dt * w_safeguard)
        
        v_mean = float(np.mean(velocities))
        v_std = float(np.std(velocities, ddof=1)) # 표본 표준편차 일치
        
        # 2. 가속도 계산
        d_velocities = np.abs(np.diff(velocities))
        accelerations = d_velocities / dt
        a_mean = float(np.mean(accelerations))
        
        # 3. 방향 전환 비율 계산 (최단 기하 거리 및 0~1 비율 규격화 적용)
        angles = np.arctan2(dcy, dcx)
        angle_diffs = np.diff(angles)
        # [-pi, pi] 범위 아키텍처 래핑 보정
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        abs_angle_diffs = np.abs(angle_diffs)
        heading_ratios = abs_angle_diffs / np.pi # 0~1 규격화
        heading_change_ratio = float(np.mean(heading_ratios))
        
        # 4. 복합 기동성 계수 (\u03c3) 연산
        maneuverability_sigma = float(v_mean / (heading_change_ratio + 1e-6))
        
        return {
            "v_mean": round(v_mean, 4),
            "v_std": round(v_std, 4),
            "a_mean": round(a_mean, 4),
            "heading_change_ratio": round(heading_change_ratio, 4),
            "maneuverability_sigma": round(maneuverability_sigma, 4)
        }

    def calculate_z_scores(self, real_features: dict) -> dict:
        """
        시뮬레이션 통계 성적표를 기반으로 실측 피처의 Z-score 차원 거리 측정
        """
        if not os.path.exists(self.stat_summary_path):
            raise FileNotFoundError(f"기준 통계서가 {self.stat_summary_path}에 없습니다. Phase 3를 먼저 구동해주세요.")
            
        with open(self.stat_summary_path, "r", encoding="utf-8") as f:
            stat_summary = json.load(f)
            
        metrics_summary = stat_summary["metrics_summary"]
        z_scores = {}
        
        for f_name, value in real_features.items():
            sim_desc = metrics_summary[f_name]["descriptive"]
            
            # 조류 집단 가우시안 대비 거리 계산
            bird_mu, bird_std = sim_desc["bird"]["mean"], sim_desc["bird"]["std"]
            z_bird = (value - bird_mu) / (bird_std + 1e-6)
            
            # 드론 집단 가우시안 대비 거리 계산
            drone_mu, drone_std = sim_desc["drone"]["mean"], sim_desc["drone"]["std"]
            z_drone = (value - drone_mu) / (drone_std + 1e-6)
            
            z_scores[f_name] = {
                "z_score_vs_bird": round(float(z_bird), 4),
                "z_score_vs_drone": round(float(z_drone), 4)
            }
            
        return z_scores

    def execute_verification(self):
        print("=" * 65)
        print("[Phase 5] 실측 데이터(50 FPS) 검정 및 파이프라인 가동")
        print("=" * 65)
        
        # 1. 데이터 로드 및 피처 추출 (50 FPS 타임 보정)
        track_data = self._load_track_data()
        print(f"-> 수령된 실측 시계열 노드 로드 완료 (총 프레임 수: {len(track_data)} 프레임)")
        
        real_features = self.extract_real_features(track_data)
        
        # 2. 시뮬레이션 데이터셋 분포 대조 및 Z-score 측정
        z_scores = self.calculate_z_scores(real_features)
        
        # 3. 검증 보고서 JSON 생성 및 저장
        report_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_track_file": os.path.basename(self.input_json_path),
            "total_processed_frames": len(track_data),
            "extracted_real_features": real_features,
            "simulation_distance_mapping": z_scores
        }
        
        with open(self.output_report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
        # 4. [R&R 이관 핵심] C파트 인도용 단일 행 표준 CSV 데이터셋 발행
        df_handoff = pd.DataFrame([real_features])
        df_handoff.to_csv(self.output_csv_path, index=False)
        
        print("\n[실측 피처 추출 및 시뮬레이션 대조 성적표]")
        for f_name, val in real_features.items():
            print(f" * {f_name:<22}: 값={val:<8} | Z(Bird)={z_scores[f_name]['z_score_vs_bird']:>7} | Z(Drone)={z_scores[f_name]['z_score_vs_drone']:>7}")
            
        print("\n" + "=" * 65)
        print(f"[실측 검증 파이프라인 연동 성공]")
        print(f"분석 대조 보고서 배포 완료 : {self.output_report_path}")
        print(f"[C파트 인도용] 실측 1행 CSV 배포 완료 : {self.output_csv_path}")
        print("=" * 65)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    verifier = RealTrackVerifier(
        data_dir=os.path.join(current_dir, "data"),
        output_dir=os.path.join(current_dir, "output")
    )
    verifier.execute_verification()