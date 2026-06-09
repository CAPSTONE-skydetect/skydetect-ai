# research/verify_real_track.py
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import circmean  # 원형 통계학 함수 동기화

# pipeline.py와 소수점 단위까지 맞추기 위한 글로벌 물리 상수 바인딩
V_MIN = 0.0
V_MAX = 5.0
H_MIN = 0.0
H_MAX = 1.0

class RealTrackVerifier:
    def __init__(self, data_dir: str = "research/data", output_dir: str = "research/output", fps: int = 50):
        """
        Phase 5: 시연 및 실측 데이터 검증 통합 제어 클래스 (통계 파일 덮어쓰기 로직 반영)
        :param data_dir: selected_track.json 이 위치한 경로
        :param output_dir: 검증 보고서, stat_summary.json, C파트 인도용 CSV가 위치한 경로
        :param fps: A파트가 시연 영상에서 추출한 실제 프레임 레이트 (30, 50, 60 등 가변 가능)
        """
        self.input_json_path = os.path.join(data_dir, "selected_track.json")
        
        # 버전 분리 없이 기존 덮어쓰기 파일(stat_summary.json)을 고정 참조
        self.stat_summary_path = os.path.join(output_dir, "stat_summary.json")
        
        self.output_report_path = os.path.join(output_dir, "verification_report.json")
        self.output_csv_path = os.path.join(output_dir, "real_track_features.csv")
        
        self.fps = fps
        self.dt = 1.0 / fps  # 시연 영상 FPS에 따른 타임스텝 자동 계산
        os.makedirs(output_dir, exist_ok=True)

    def _load_tracks(self) -> list:
        """
        JSON 구조를 자동 분석하여 단일 트랙 혹은 멀티 트랙 리스트를 유연하게 반환
        """
        if not os.path.exists(self.input_json_path):
            raise FileNotFoundError(f"실측 파일 누락: {self.input_json_path}")
            
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            root_data = json.load(f)
            
        # Case A: 단일 트랙 리스트가 최상위에 바로 있는 경우
        if isinstance(root_data, list):
            return [root_data]
            
        # Case B: 최상위가 딕셔너리 구조인 경우 (기존 selected_track 스펙 및 멀티트랙 스펙 대응)
        if isinstance(root_data, dict):
            if "tracks" in root_data and isinstance(root_data["tracks"], list):
                return root_data["tracks"]
                
            for key, value in root_data.items():
                if isinstance(value, list) and len(value) > 0 and "cx" in value[0]:
                    return [value]
                    
        raise ValueError("selected_track.json 내부에서 유효한 트랙 구조를 찾을 수 없습니다.")

    def extract_features_exact(self, track_data: list) -> dict:
        """
        [100% Sync with pipeline.py] 논문 및 파이프라인과 완벽히 일치하는 정밀 피처 추출
        """
        N = len(track_data)
        if N < 3:
            return None

        cx = np.array([p["cx"] for p in track_data])
        cy = np.array([p["cy"] for p in track_data])
        w = np.array([p["w"] for p in track_data])
        
        dx = np.diff(cx)
        dy = np.diff(cy)
        displacement = np.sqrt(dx**2 + dy**2)  # 오타 수정 완료 (dcy -> dy)
        
        # 1. 속도 계산 (pipeline.py와 동일하게 ddof=0 모표준편차 동기화)
        v_norm_series = displacement / (w[1:] * self.dt + 1e-6)
        v_mean = float(np.mean(v_norm_series))
        v_std = float(np.std(v_norm_series)) 
        
        # 2. 가속도 계산
        a_norm_series = np.abs(np.diff(v_norm_series)) / self.dt
        a_mean = float(np.mean(a_norm_series))
        
        # 3. [수식 완전 동기화] circmean 기반의 원형 방향 편차(heading_change_ratio) 산출
        headings = np.degrees(np.arctan2(dy, dx))
        headings = (headings + 360) % 360
        
        h_mean = circmean(headings, high=360, low=0)
        
        delta_h_list = []
        for h_i in headings:
            diff = abs(h_i - h_mean)
            shortest_diff = min(diff, 360.0 - diff)
            delta_h_list.append(shortest_diff ** 2)
            
        raw_h_std = np.sqrt(np.sum(delta_h_list) / len(headings))
        heading_change_ratio = float(raw_h_std / 180.0)
        
        # 4. [수식 완전 동기화] 글로벌 상수 기반 MinMax 스케일링 후 복합 기동성 계수 산출
        v_scaled = (v_mean - V_MIN) / (V_MAX - V_MIN + 1e-6)
        h_scaled = (heading_change_ratio - H_MIN) / (H_MAX - H_MIN + 1e-6)
        maneuverability_sigma = float(v_scaled / (h_scaled + 1e-6))
        
        return {
            "v_mean": round(v_mean, 4),
            "v_std": round(v_std, 4),
            "a_mean": round(a_mean, 4),
            "heading_change_ratio": round(heading_change_ratio, 4),
            "maneuverability_sigma": round(maneuverability_sigma, 4)
        }

    def calculate_z_scores(self, real_features: dict) -> dict:
        """
        덮어씌워진 최신 stat_summary.json 가이드를 기반으로 가우시안 Z-score 측정
        """
        if not os.path.exists(self.stat_summary_path):
            raise FileNotFoundError(f"기준 통계서가 {self.stat_summary_path}에 없습니다. 통계 분석 코드를 먼저 실행해주세요.")
            
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
        print(f"[통합 시연 모드] 실측 데이터 통합 피처 파이프라인 가동")
        print(f" └ 참조 통계 파일: {os.path.basename(self.stat_summary_path)}")
        print("=" * 65)
        
        # 멀티/단일 트랙 로드
        all_tracks = self._load_tracks()
        print(f"-> A파트 원천 JSON에서 총 [{len(all_tracks)}]개의 비행 트랙 로드 완료.")
        
        feature_rows = []
        verification_results = []
        
        for idx, track in enumerate(all_tracks):
            features = self.extract_features_exact(track)
            if features is not None:
                feature_rows.append(features)
                
                # 각 트랙별 시뮬레이션 데이터셋 대비 Z-Score 거리 정밀 연산
                z_scores = self.calculate_z_scores(features)
                verification_results.append({
                    "track_index": idx,
                    "total_frames": len(track),
                    "extracted_features": features,
                    "simulation_distance_mapping": z_scores
                })
                
        if not feature_rows:
            print("유효한 프레임을 가진 트랙이 존재하지 않습니다.")
            return

        # 최종 검증 보고서 JSON 규격화 매핑 및 물리 저장소 출력
        report_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_track_file": os.path.basename(self.input_json_path),
            "total_processed_tracks": len(feature_rows),
            "tracks_report": verification_results
        }
        
        with open(self.output_report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # C파트 인도용 단일/멀티 행 CSV 데이터프레임 빌드
        df_handoff = pd.DataFrame(feature_rows)
        ordered_columns = ["v_mean", "v_std", "a_mean", "heading_change_ratio", "maneuverability_sigma"]
        df_final = df_handoff[ordered_columns]
        
        # CSV 파일 발행
        df_final.to_csv(self.output_csv_path, index=False)
        
        print(f"\n[시연 피처 압축 배포 완료 - 총 {len(df_final)}행 발행]")
        print(df_final.to_string(index=False))
        
        # 터미널 콘솔창에서도 Z-score 대조표를 직관적으로 확인할 수 있도록 로그 추가
        print("\n[실측 특징량 대조 성적표 (Z-Score Summary)]")
        for res in verification_results:
            print(f"--- Track Index: {res['track_index']} (총 {res['total_frames']}프레임) ---")
            for f_name, val in res["extracted_features"].items():
                z_bird = res["simulation_distance_mapping"][f_name]["z_score_vs_bird"]
                z_drone = res["simulation_distance_mapping"][f_name]["z_score_vs_drone"]
                print(f" * {f_name:<22}: 값={val:<8} | Z(Bird)={z_bird:>7} | Z(Drone)={z_drone:>7}")
        
        print("\n" + "=" * 65)
        print(f"[시연 통합 인터페이스 빌드 완료]")
        print(f" [보고서 출력 성공] 실측 검증 JSON 경로: {self.output_report_path}")
        print(f" [C파트 즉시 주입용] 실측 CSV 경로: {self.output_csv_path}")
        print("=" * 65)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    verifier = RealTrackVerifier(
        data_dir=os.path.join(current_dir, "data"),
        output_dir=os.path.join(current_dir, "output"),
        fps=50              # 시연용 영상 촬영 스펙에 맞게 30, 50, 60 등으로 자유롭게 조정 가능!
    )
    verifier.execute_verification()