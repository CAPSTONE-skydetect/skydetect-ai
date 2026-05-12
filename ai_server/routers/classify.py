import time

from fastapi import APIRouter

from ai_server.schemas import (
    ClassifyRequest,
    PredictionResult,
    ResponseQuality,
    RuleFilterResult,
)
from ai_server.services.classifier import RFClassifier
from ai_server.services.rule_filter import RuleFilter
from ai_server.utils.explain import get_top_features

router = APIRouter(tags=["classify"])

_rule_filter = RuleFilter()
_classifier = RFClassifier()


@router.post("/classify/rule-filter", response_model=RuleFilterResult)
def rule_filter(payload: ClassifyRequest) -> RuleFilterResult:
    return _rule_filter.apply(
        payload.feature_vector,
        min_track_length=payload.min_track_length,
        min_mean_conf=payload.min_mean_conf,
        max_missing_ratio=payload.max_missing_ratio,
    )


@router.post("/classify", response_model=PredictionResult)
def classify(payload: ClassifyRequest) -> PredictionResult:
    start = time.perf_counter()
    fv = payload.feature_vector

    filter_result = _rule_filter.apply(
        fv,
        min_track_length=payload.min_track_length,
        min_mean_conf=payload.min_mean_conf,
        max_missing_ratio=payload.max_missing_ratio,
    )

    quality = ResponseQuality(
        num_points=fv.quality.num_points if fv.quality else 0,
        track_stability=fv.quality.track_stability if fv.quality else "poor",
        feature_status=fv.feature_status,
    )

    if not filter_result.passed:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return PredictionResult(
            track_id=fv.track_id,
            label="uncertain",
            confidence=0.0,
            rule_filter=filter_result,
            top_features={},
            quality=quality,
            processing_time_ms=elapsed_ms,
        )

    label, confidence = _classifier.predict(fv)
    top_features = get_top_features(_classifier.clf, _classifier.feature_names)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return PredictionResult(
        track_id=fv.track_id,
        label=label,
        confidence=confidence,
        rule_filter=filter_result,
        top_features=top_features,
        quality=quality,
        processing_time_ms=elapsed_ms,
    )
