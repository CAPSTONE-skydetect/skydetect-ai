"""Part C: 평가 결과를 한 장짜리 카드 이미지로 그린다.

GitHub 이슈에 실험 기록을 남길 때 숫자 표 대신 붙이는 그림이다. RF 와 MiniRocket 이
같은 함수를 써서 실험끼리 같은 모양으로 비교되게 한다.

카드 한 줄 = 학습 구성(arm) 하나:
    [혼동행렬 히트맵] [종합 지표 + 95% CI] [클래스별 정밀도·재현율]
추이 그림 = 실험 순서대로 macro-F1(95% CI) 과 드론 recall.

입력 지표 dict 형식 (없는 값은 생략 가능):
    {"name": str, "cm": [[TN-bird, bird→drone], [drone→bird, TP-drone]],
     "accuracy", "balanced_accuracy", "macro_f1", "roc_auc": float,
     "precision": {"bird", "drone"}, "recall": {"bird", "drone"},
     "ci": {"macro_f1": [lo, hi], "balanced_accuracy": [lo, hi]},
     "coverage": float, "n_label": str}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

LABELS = ["bird", "drone"]
LABEL_KO = {"bird": "새", "drone": "드론"}

# dataviz 기준 팔레트. 새 = 파랑(slot 1), 드론 = 주황(slot 2).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BIRD = "#2a78d6"
DRONE = "#eb6834"
METRIC = "#2a78d6"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
REAL_BAND = "#f0efec"

_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
_KOREAN_FONTS = ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR")


def _setup_font() -> None:
    try:  # pip install koreanize-matplotlib 이 있으면 NanumGothic 을 등록해 준다.
        import koreanize_matplotlib  # noqa: F401
    except ImportError:
        pass
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _KOREAN_FONTS:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
    })


def _clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)


def _confusion(ax, cm, title: str) -> None:
    cm = np.asarray(cm, dtype=float)
    rows = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    ax.imshow(share, cmap=_CMAP, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            dark = share[i, j] > 0.55
            ax.text(j, i - 0.08, f"{int(cm[i, j]):,}", ha="center", va="center", fontsize=17,
                    fontweight="bold", color="#ffffff" if dark else INK)
            ax.text(j, i + 0.22, f"{share[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if dark else INK_2)
    ax.set_xticks([0, 1], [f"예측 {LABEL_KO[l]}" for l in LABELS], fontsize=10, color=INK_2)
    ax.set_yticks([0, 1], [f"정답 {LABEL_KO[l]}" for l in LABELS], fontsize=10, color=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    # 칸 사이 간격과 정답 칸(대각선) 테두리. 색만으로 읽지 않게 한다.
    ax.axhline(0.5, color=SURFACE, lw=4)
    ax.axvline(0.5, color=SURFACE, lw=4)
    for k in range(2):
        ax.add_patch(plt.Rectangle((k - 0.47, k - 0.47), 0.94, 0.94, fill=False, ec=INK, lw=1.6))
    ax.set_title(title, fontsize=11, color=INK_2, loc="left", pad=8)


def _summary(ax, m: dict) -> None:
    rows = [("macro-F1", "macro_f1"), ("Balanced acc", "balanced_accuracy"),
            ("Accuracy", "accuracy"), ("ROC-AUC", "roc_auc")]
    if m.get("coverage") is not None and m["coverage"] < 1:
        rows.append(("Coverage", "coverage"))
    rows = [(label, key) for label, key in rows if m.get(key) is not None]
    y = np.arange(len(rows))[::-1]
    values = [m[key] for _, key in rows]
    ax.barh(y, values, height=0.5, color=METRIC, alpha=0.9)
    ci = m.get("ci", {})
    for yy, (label, key), value in zip(y, rows, values):
        if key in ci:
            lo, hi = ci[key]
            ax.plot([lo, hi], [yy, yy], color=INK, lw=1.6, solid_capstyle="butt")
            ax.plot([lo, lo], [yy - 0.13, yy + 0.13], color=INK, lw=1.6)
            ax.plot([hi, hi], [yy - 0.13, yy + 0.13], color=INK, lw=1.6)
        ax.text(1.06, yy, f"{value:.3f}", va="center", ha="left", fontsize=11,
                transform=ax.get_yaxis_transform(),
                fontweight="bold" if key == "macro_f1" else "normal", color=INK)
    ax.set_yticks(y, [label for label, _ in rows], fontsize=10, color=INK_2)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1.0"], fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _clean(ax)
    note = "검은 선 = 95% 신뢰구간" if ci else ""
    ax.set_title(f"종합 지표   {note}", fontsize=11, color=INK_2, loc="left", pad=8)


def _per_class(ax, m: dict) -> None:
    groups = [("정밀도", m.get("precision")), ("재현율", m.get("recall"))]
    groups = [(name, values) for name, values in groups if values]
    width = 0.36
    x = np.arange(len(groups))
    for offset, label, color in ((-width / 2, "bird", BIRD), (width / 2, "drone", DRONE)):
        values = [values[label] for _, values in groups]
        bars = ax.bar(x + offset, values, width=width - 0.04, color=color, label=LABEL_KO[label])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x, [name for name, _ in groups], fontsize=10, color=INK_2)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1.0"], fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _clean(ax)
    ax.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(0, 1.13), ncol=2)
    ax.set_title("클래스별", fontsize=11, color=INK_2, loc="right", pad=8)


def experiment_card(title: str, subtitle: str, arms: list[dict], path: str | Path) -> Path:
    """실험 하나를 카드 이미지로 저장한다. arms 가 여러 개면 줄을 늘린다."""
    _setup_font()
    n = len(arms)
    head_in, row_in, foot_in = 1.25, 3.3, 0.35
    height = head_in + row_in * n + foot_in
    fig = plt.figure(figsize=(13, height))
    grid = fig.add_gridspec(n, 3, width_ratios=[1.0, 1.5, 1.2], wspace=0.42, hspace=0.55,
                            top=1 - head_in / height, bottom=foot_in / height,
                            left=0.06, right=0.95)
    fig.text(0.06, 1 - 0.38 / height, title, fontsize=17, fontweight="bold", va="center")
    fig.text(0.06, 1 - 0.72 / height, subtitle, fontsize=11, color=INK_2, va="center")
    for row, arm in enumerate(arms):
        head = arm["name"] + (f"  ·  {arm['n_label']}" if arm.get("n_label") else "")
        _confusion(fig.add_subplot(grid[row, 0]), arm["cm"], head)
        _summary(fig.add_subplot(grid[row, 1]), arm)
        _per_class(fig.add_subplot(grid[row, 2]), arm)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def trend_chart(title: str, points: list[dict], path: str | Path) -> Path:
    """실험 순서대로 macro-F1(95% CI) 과 드론 recall 을 그린다.

    points: [{"id", "label", "macro_f1", "ci": [lo, hi] | None, "drone_recall" | None, "real": bool}]
    실제 데이터 평가는 회색 띠로 구분한다. 합성 점수와 실제 점수는 같은 의미가 아니다.
    """
    _setup_font()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.2)
    x = np.arange(len(points))
    for i, p in enumerate(points):
        if p.get("real"):
            ax.axvspan(i - 0.45, i + 0.45, color=REAL_BAND, zorder=0)
            ax.text(i, 1.06, "실제 데이터", ha="center", fontsize=9, color=INK_2)
    f1 = [p.get("macro_f1") for p in points]
    has = [i for i, v in enumerate(f1) if v is not None]
    ax.plot([x[i] for i in has], [f1[i] for i in has], color=METRIC, lw=2, marker="o",
            markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, label="macro-F1 (95% CI)", zorder=3)
    for i in has:
        ci = points[i].get("ci")
        if ci:
            ax.plot([i, i], ci, color=METRIC, lw=1.4, alpha=0.6, zorder=2)
        ax.text(i, f1[i] + 0.07, f"{f1[i]:.3f}", ha="center", fontsize=11, fontweight="bold", color=INK)
    recall = [p.get("drone_recall") for p in points]
    has_r = [i for i, v in enumerate(recall) if v is not None]
    if has_r:
        ax.plot([x[i] for i in has_r], [recall[i] for i in has_r], color=DRONE, lw=2, ls="--",
                marker="s", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label="드론 recall", zorder=3)
    for i, p in enumerate(points):
        if p.get("macro_f1") is None and p.get("note"):
            ax.text(i, 0.5, p["note"], ha="center", va="center", fontsize=10, color=MUTED, wrap=True)
    ax.set_xticks(x, [f"{p['id']}\n{p['label']}" for p in points], fontsize=9.5, color=INK_2)
    ax.set_xlim(-0.6, len(points) - 0.4)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _clean(ax)
    ax.legend(frameon=False, fontsize=10, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    fig.text(0.07, 0.95, title, fontsize=15, fontweight="bold")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
