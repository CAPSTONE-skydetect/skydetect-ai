"""Part C: RuleFilter + RF 추론을 묶은 판정 절차.

routers/classify.py 와 routers/ui.py 가 같은 절차를 써야 한다. 라우터마다 필터를
걸고 모델을 부르는 순서를 각자 적어두면, 임계값이나 탈락 처리 방식이 바뀔 때 한쪽만
고쳐져 두 경로의 판정이 갈린다. 그래서 절차를 여기 한 곳에 둔다.
"""

from __future__ import annotations

import time

from ai_server.schemas import (
    FeatureVector,
    PredictionResult,
    ResponseQuality,
)
from ai_server.services.classifier import RFClassifier
from ai_server.services.rule_filter import RuleFilter
from ai_server.utils.explain import get_top_features

# 모델 로딩은 수백 ms가 걸린다. 요청마다 반복하지 않도록 모듈 수준에서 한 번만 만든다.
_rule_filter = RuleFilter()
_classifier = RFClassifier()


def get_rule_filter() -> RuleFilter:
    return _rule_filter


def get_classifier() -> RFClassifier:
    return _classifier


def classify_feature_vector(
    fv: FeatureVector,
    *,
    min_track_length: int | None = None,
    min_mean_conf: float | None = None,
    max_missing_ratio: float | None = None,
) -> PredictionResult:
    """FeatureVector를 받아 최종 판정(bird / drone / uncertain)을 만든다.

    필터에서 탈락하면 모델을 아예 부르지 않고 "uncertain"으로 끝낸다. 품질이
    미달인 트랙에 모델을 돌리면 근거 없는 확신값이 나오기 때문이다.
    """
    start = time.perf_counter()

    filter_result = _rule_filter.apply(
        fv,
        min_track_length=min_track_length,
        min_mean_conf=min_mean_conf,
        max_missing_ratio=max_missing_ratio,
    )

    quality = ResponseQuality(
        num_points=fv.quality.num_points if fv.quality else 0,
        mean_conf=fv.quality.mean_conf if fv.quality else 0.0,
        track_stability=fv.quality.track_stability if fv.quality else "poor",
        feature_status=fv.feature_status,
    )

    if not filter_result.passed:
        return PredictionResult(
            track_id=fv.track_id,
            label="uncertain",
            confidence=0.0,
            rule_filter=filter_result,
            top_features={},
            quality=quality,
            processing_time_ms=_elapsed_ms(start),
        )

    label, confidence = _classifier.predict(fv)

    return PredictionResult(
        track_id=fv.track_id,
        label=label,
        confidence=confidence,
        rule_filter=filter_result,
        top_features=get_top_features(_classifier.clf, _classifier.feature_names),
        quality=quality,
        processing_time_ms=_elapsed_ms(start),
    )


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
