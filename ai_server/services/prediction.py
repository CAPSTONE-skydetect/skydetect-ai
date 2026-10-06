"""Part C: MiniRocket 추론을 응답 스키마로 묶는 판정 절차.

라우터마다 모델 호출·보류 처리를 각자 적어두면 정책이 바뀔 때 한쪽만 고쳐져
판정이 갈린다. 그래서 절차를 여기 한 곳에 둔다.
"""

from __future__ import annotations

import time
from functools import lru_cache

from ai_server.schemas import ClassifyRequest, PredictionResult
from ai_server.services.classifier import MiniRocketClassifier


@lru_cache(maxsize=1)
def get_classifier() -> MiniRocketClassifier:
    # 모델 로딩과 numba 준비는 비싸다. 프로세스당 한 번만 한다.
    return MiniRocketClassifier()


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
