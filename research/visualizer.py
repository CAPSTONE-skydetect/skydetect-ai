# research/visualizer.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AcademicVisualizer:
    def __init__(self, data_dir: str = "research/data", plots_dir: str = "research/plots"):
        """
        Phase 4: Academic Visualizer 클래스
        :param data_dir: simulation_features.csv 가 위치한 경로
        :param plots_dir: 시각화 리포트 파일(.png, .pdf)이 저장될 목적지 경로
        """
        self.input_path = os.path.join(data_dir, "simulation_features.csv")
        self.plots_dir = plots_dir
        self.output_png = os.path.join(plots_dir, "report_plots.png")
        self.output_pdf = os.path.join(plots_dir, "report_plots.pdf")
        
        # 출력 디렉토리 사전 확보
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 학술 논문 스타일 전역 폰트 및 스타일 초기화 (OS 독립적 렌더링 세팅)
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["axes.unicode_minus"] = False

    def generate_dashboard(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"입력 데이터셋이 {self.input_path}에 존재하지 않습니다. Phase 2를 먼저 가동해주세요.")

        # 1. 데이터 로드 및 클래스 컬러 팔레트 바인딩 (조류: SkyBlue, 드론: Coral)
        df = pd.read_csv(self.input_path)
        palette = {"bird": "#4a90e2", "drone": "#e65100"}
        
        # 2. 2x2 격자 구조의 메인 피규어 생성 (논문 삽입 규격 14x12 인치 설정)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("SkyDetect AI Engine - Maneuver Analytics & Feature Verification", 
                     fontsize=18, fontweight="bold", y=0.98)

        # ─────────────────────────────────────────────────────────────────
        # [Plot 1] Histogram & KDE (속도 및 가속도 확률밀도 대조)
        # ─────────────────────────────────────────────────────────────────
        ax1 = axes[0, 0]
        sns.kdeplot(data=df, x="a_mean", hue="label", fill=True, common_norm=False, 
                    palette=palette, alpha=0.4, linewidth=2, ax=ax1)
        ax1.set_title("Plot 1: Acceleration Probability Density (KDE)", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Normalized Acceleration Mean ($a_{mean}$)", fontsize=11)
        ax1.set_ylabel("Density", fontsize=11)

        # ─────────────────────────────────────────────────────────────────
        # [Plot 2] Violin Plot (속도 표준편차 및 방향 편차의 변산성 대조)
        # ─────────────────────────────────────────────────────────────────
        ax2 = axes[0, 1]
        ax2.axis('off') # 기존 격자 축은 투명하게 숨김 처리
        
        # Plot 2가 들어갈 자리에 좌/우 2개의 서브 Plot을 동적으로 삽입 (격자 안의 격자)
        gs = ax2.get_subplotspec().subgridspec(1, 2, wspace=0.3)
        ax2_left = fig.add_subplot(gs[0, 0])
        ax2_right = fig.add_subplot(gs[0, 1])
        
        # 왼쪽 서브 플롯: Speed Std (체급: 0 ~ 20)
        sns.violinplot(data=df, x="label", y="v_std", hue="label", legend=False,
                       palette=palette, inner="quart", ax=ax2_left)
        ax2_left.set_title("Speed Std ($v_{std}$)", fontsize=11, fontweight="bold")
        ax2_left.set_xlabel("")
        ax2_left.set_ylabel("Velocity Scale (BL/s)", fontsize=10)
        
        # 오른쪽 서브 플롯: Heading Change Ratio (체급: 0.0 ~ 0.5)
        sns.violinplot(data=df, x="label", y="heading_change_ratio", hue="label", legend=False,
                       palette=palette, inner="quart", ax=ax2_right)
        ax2_right.set_title("Heading Ratio ($h_{ratio}$)", fontsize=11, fontweight="bold")
        ax2_right.set_xlabel("")
        ax2_right.set_ylabel("Ratio (0.0 ~ 1.0)", fontsize=10)
        
        # 상위 타이틀은 부모 축의 타이틀로 통합 관리
        ax2.set_title("Plot 2: Kinematic Stability & Jitter Analysis", fontsize=13, fontweight="bold", y=1.05)
        
        # ─────────────────────────────────────────────────────────────────
        # [Plot 3] Maneuverability Scatter with Contour (기동성 계수 지형도 매핑)
        # ─────────────────────────────────────────────────────────────────
        ax3 = axes[1, 0]
        
        # 등고선(Contour)을 그리기 위한 가상의 Grid 데이터 연산 레이어
        grid_x, grid_y = np.meshgrid(np.linspace(0.0, 5.0, 100), np.linspace(0.0, 1.0, 100))
        # 파이프라인과 동일한 복합 기동성 수식 적용: sigma = v_scaled / h_scaled
        grid_z = grid_x / (grid_y + 1e-6)
        
        # 등고선 배경 채색 매핑 (Maneuverability 가 높을수록 밝은 녹색 계열)
        contour_filled = ax3.contourf(grid_x, grid_y, grid_z, levels=25, cmap="YlGn", alpha=0.15, vmax=15)
        contours = ax3.contour(grid_x, grid_y, grid_z, levels=[1.0, 3.0, 5.0, 8.0], colors="gray", linewidths=0.5)
        ax3.clabel(contours, inline=True, fontsize=8, fmt=lambda x: f"$\sigma$={x:.1f}")
        
        # 실제 데이터 1,200개 산점도 플로팅
        sns.scatterplot(data=df, x="v_mean", y="heading_change_ratio", hue="label", 
                        palette=palette, alpha=0.7, edgecolor="w", s=40, ax=ax3)
        ax3.set_title("Plot 3: 2D Maneuverability Space & Contour Map", fontsize=13, fontweight="bold")
        ax3.set_xlabel("Normalized Speed Mean ($v_{mean}$)", fontsize=11)
        ax3.set_ylabel("Heading Change Ratio ($h_{ratio}$)", fontsize=11)
        ax3.set_xlim(0.0, 5.0)  # Phase 2 V_MAX 스케일 동기화
        ax3.set_ylim(0.0, 1.0)  # Shared Schema 스케일 동기화

        # ─────────────────────────────────────────────────────────────────
        # [Plot 4] Correlation Matrix Heatmap (다중공선성 및 독립성 검증)
        # ─────────────────────────────────────────────────────────────────
        ax4 = axes[1, 1]
        feature_cols = ["v_mean", "v_std", "a_mean", "heading_change_ratio", "maneuverability_sigma"]
        corr_matrix = df[feature_cols].corr(method="pearson")
        
        # 피어슨 상관계수 매트릭스 그리기 (수치 텍스트 표시 설정)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0,
                    linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax4)
        ax4.set_title("Plot 4: Pearson Correlation Matrix Matrix", fontsize=13, fontweight="bold")
        ax4.set_xticklabels(feature_cols, rotation=25, ha="right")
        ax4.set_yticklabels(feature_cols, rotation=0)

        # 3. 레이아웃 여백 자동 최적화 조율 및 더블 익스포트 (PNG/PDF)
        plt.tight_layout()
        
        # 300 DPI 이상 고해상도 이미지 배포 발행
        fig.savefig(self.output_png, dpi=300, bbox_inches="tight")
        # 논문 인서트용 백터 포맷 발행
        fig.savefig(self.output_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)

        print("=" * 65)
        print(f"[Phase 4 완료] 학술 분석 리포트 대시보드 시각화 성공.")
        print(f"고해상도 발표용 PNG: {self.output_png}")
        print(f"논문 인쇄 인서트용 PDF: {self.output_pdf}")
        print("=" * 65)
        return self.plots_dir

if __name__ == "__main__":
    # 단독 유닛 테스트 가동용 모듈 경로 제어
    current_dir = os.path.dirname(os.path.abspath(__file__))
    visualizer = AcademicVisualizer(
        data_dir=os.path.join(current_dir, "data"),
        plots_dir=os.path.join(current_dir, "plots")
    )
    visualizer.generate_dashboard()