from fastapi import APIRouter

from ai_server.schemas import ClassifyRequest, PredictionResult, RuleFilterResult
from ai_server.services.prediction import classify_feature_vector, get_rule_filter

router = APIRouter(tags=["classify"])


@router.post("/classify/rule-filter", response_model=RuleFilterResult)
def rule_filter(payload: ClassifyRequest) -> RuleFilterResult:
    return get_rule_filter().apply(
        payload.feature_vector,
        min_track_length=payload.min_track_length,
        min_mean_conf=payload.min_mean_conf,
        max_missing_ratio=payload.max_missing_ratio,
    )


@router.post("/classify", response_model=PredictionResult)
def classify(payload: ClassifyRequest) -> PredictionResult:
    return classify_feature_vector(
        payload.feature_vector,
        min_track_length=payload.min_track_length,
        min_mean_conf=payload.min_mean_conf,
        max_missing_ratio=payload.max_missing_ratio,
    )
