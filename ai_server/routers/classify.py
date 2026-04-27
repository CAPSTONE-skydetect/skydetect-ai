from fastapi import APIRouter

from ai_server.schemas import ClassifyRequest, RuleFilterResult
from ai_server.services.rule_filter import RuleFilter

router = APIRouter(tags=["classify"])

_rule_filter = RuleFilter()


@router.post("/classify/rule-filter", response_model=RuleFilterResult)
def rule_filter(payload: ClassifyRequest) -> RuleFilterResult:
    return _rule_filter.apply(
        payload.feature_vector,
        min_track_length=payload.min_track_length,
        min_mean_conf=payload.min_mean_conf,
        max_missing_ratio=payload.max_missing_ratio,
    )
