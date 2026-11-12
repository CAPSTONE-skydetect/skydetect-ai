"""물체 상자(w, h) 시계열을 MiniRocket 창과 같은 시점으로 뽑는다 (E4).

창마다 4채널 (N, 4, 60):
  s  = log(상자 크기) - 창 안 중앙값   (크기의 상대 변화, 거리와 무관)
  a  = log(가로/세로)                  (모양, 거리와 무관)
  ds, da = 직전 시점 대비 변화량       (첫 칸 0)
시점은 trajectory_sequence.window_track 과 같은 30Hz 격자(window_start_s 부터 60점)에서
선형 보간한다. 짧은 끊김 증강으로 지운 프레임(recipes.csv 의 dropped_frame_indices)은 똑같이 빼고 보간한다.
위치 증강(location_error)은 중심 좌표만 흔들므로 상자 값은 원본과 같다.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .trajectory_sequence import SequenceConfig

CONFIG = SequenceConfig()


@lru_cache(maxsize=None)
def _track(path: str):
    track = json.loads(Path(path).read_text(encoding="utf-8"))
    width, height = float(track["processed_width"]), float(track["processed_height"])
    rows = np.array([[p["frame_index"], p["timestamp_ms"] / 1000.0, p["w"] * width, p["h"] * height]
                     for p in track["history"]], dtype=np.float64)
    return rows


def box_window(rows: np.ndarray, start: float, dropped: set[int], jitter_px: float = 0.0,
               rng: np.random.Generator | None = None) -> np.ndarray:
    keep = ~np.isin(rows[:, 0], list(dropped)) if dropped else np.ones(len(rows), bool)
    time, w, h = rows[keep, 1], rows[keep, 2].copy(), rows[keep, 3].copy()
    if jitter_px:
        w = w + rng.normal(0, jitter_px, len(w))
        h = h + rng.normal(0, jitter_px, len(h))
    w, h = np.maximum(w, 1.0), np.maximum(h, 1.0)
    grid = start + np.arange(CONFIG.samples) / CONFIG.fps
    wi, hi = np.interp(grid, time, w), np.interp(grid, time, h)
    size = 0.5 * np.log(wi * hi)
    s = size - np.median(size)
    a = np.log(wi / hi)
    return np.stack([s, a, np.diff(s, prepend=s[0]), np.diff(a, prepend=a[0])]).astype(np.float32)


def package_box_windows(package_dir: Path, sources: pd.DataFrame, sample_ids, jitter_px: float = 0.0,
                        seed: int = 0) -> np.ndarray:
    """package 의 창 순서(sample_ids)대로 상자 채널을 만든다. sources: sources.csv (parent_track_id 인덱스)."""
    meta = pd.read_csv(package_dir / "minirocket_windows.csv", encoding="utf-8-sig").set_index("window_id")
    recipes = pd.read_csv(package_dir / "recipes.csv", encoding="utf-8-sig").set_index("sample_id")
    dropped = {sid: set(json.loads(v)) for sid, v in recipes.dropped_frame_indices.items()}
    rng = np.random.default_rng(seed)
    out = np.empty((len(sample_ids), 4, CONFIG.samples), dtype=np.float32)
    for i, wid in enumerate(sample_ids):
        row = meta.loc[wid]
        rows = _track(sources.loc[row.parent_track_id].source_path)
        out[i] = box_window(rows, float(row.window_start_s), dropped.get(row.sample_id, set()), jitter_px, rng)
    if not np.isfinite(out).all():
        raise ValueError(f"Non-finite box channels in {package_dir}")
    return out
