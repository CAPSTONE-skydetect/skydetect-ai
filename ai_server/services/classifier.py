"""Part C: MiniRocket + Ridge 분류기 추론 모듈.

TrackSequence 하나를 B의 trajectory_sequence.window_track 으로 2초 창들로 자르고,
창별 Ridge margin 을 평균해 판정한다. 학습 패키지를 만든 함수와 같은 함수를 쓰므로
학습용·서비스용 전처리가 갈리지 않는다.

joblib 은 실행 가능한 직렬화이므로 신뢰하는 출처의 모델만 로드한다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from ai_server.schemas import TrackSequence
from ai_server.services.sequence_model import (
    DEFAULT_MODEL_PATH,
    check_bundle,
    decision_scores,
)
from research.trajectory_sequence import SequenceConfig, window_track


@dataclass
class SequencePrediction:
    label: str
    decision_score: float | None
    abstain_reason: str | None = None
    abstain_detail: str | None = None
    window_scores: list[float] = field(default_factory=list)
    window_rejections: dict[str, int] = field(default_factory=dict)


class MiniRocketClassifier:
    """학습된 MiniRocket+scaler+Ridge 번들을 로드하고 TrackSequence 를 분류한다."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {path}\n"
                "먼저 python -m ai_server.services.train 을 실행하세요."
            )
        self._model = joblib.load(path)
        check_bundle(self._model)
        self._config = SequenceConfig()
        # MiniRocket 은 numba JIT 이라 첫 호출이 느리다. 첫 요청이 그 비용을 떠안지 않게 미리 돌린다.
        decision_scores(self._model, np.zeros((1, 4, self._config.samples), dtype=np.float32))

    @property
    def model(self) -> dict:
        return self._model

    @property
    def version(self) -> str:
        m = self._model
        return f"{m['contract_version']}/{m['contract_id']}/{m['arm']}/alpha={m['alpha']:g}"

    def predict(
        self,
        track: TrackSequence,
        *,
        margin_threshold: float = 0.0,
        min_windows: int = 1,
    ) -> SequencePrediction:
        try:
            windows, rejected = window_track(track.model_dump(mode="json"), self._config)
        except (ValueError, KeyError, TypeError) as exc:
            return SequencePrediction("uncertain", None, "invalid_input", str(exc))

        rejections = dict(Counter(r["reason"] for r in rejected))
        if len(windows) < min_windows:
            return SequencePrediction(
                "uncertain", None, "insufficient_observation",
                f"유효 2초 창 {len(windows)}개 < 최소 {min_windows}개",
                window_rejections=rejections,
            )

        scores = decision_scores(self._model, np.stack([w["X"] for w in windows]))
        # 창 점수를 평균해 긴 track 이 창 개수만큼 가산점을 받지 않게 한다.
        mean = float(scores.mean())
        result = SequencePrediction(
            "drone" if mean >= 0 else "bird", mean,
            window_scores=[float(s) for s in scores], window_rejections=rejections,
        )
        if abs(mean) < margin_threshold:
            result.label = "uncertain"
            result.abstain_reason = "low_separation"
            result.abstain_detail = f"|margin| {abs(mean):.3f} < {margin_threshold:g}"
        return result
