"""Part C: MultiRocket Stack 추론을 응답 스키마로 묶는 판정 절차 (MiniRocket 과 같은 정책)."""

from __future__ import annotations

import time
from functools import lru_cache

from ai_server.classifiers.multirocket.classifier import MultiRocketStackClassifier
from ai_server.classifiers.multirocket.contracts import ClassifyRequest, PredictionResult


@lru_cache(maxsize=1)
def get_classifier() -> MultiRocketStackClassifier:
    # 모델 로딩과 numba 준비는 비싸다. 프로세스당 한 번만 한다.
    return MultiRocketStackClassifier()


def classify_track_sequence(request: ClassifyRequest) -> PredictionResult:
    """TrackSequence 를 받아 최종 판정(bird / drone / uncertain)을 만든다."""
    start = time.perf_counter()
    classifier = get_classifier()
    track = request.track_sequence
    result = classifier.predict(
        track,
        margin_threshold=request.margin_threshold,
        min_windows=request.min_windows,
    )
    return PredictionResult(
        track_id=track.track_id,
        label=result.label,
        decision_score=result.decision_score,
        abstain_reason=result.abstain_reason,
        abstain_detail=result.abstain_detail,
        windows_used=len(result.window_scores),
        window_scores=result.window_scores,
        window_rejections=result.window_rejections,
        window_starts_s=result.window_starts_s,
        model_version=classifier.version,
        quality=track.quality,
        processing_time_ms=int((time.perf_counter() - start) * 1000),
    )
