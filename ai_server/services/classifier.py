"""Classifier stubs for part C."""

from ai_server.schemas import ClassifyRequest, PredictionResult, ResponseQuality, RuleFilterResult

RF_FEATURE_ORDER = [
    "v_mean",
    "v_std",
    "a_mean",
    "heading_change_ratio",
    "maneuverability_sigma",
    "curvature_mean",
    "curvature_cv",
    "turning_angle_mean",
    "bbox_area_mean",
    "bbox_area_std",
    "glcm_corr",
]


def classify_feature_vector(request: ClassifyRequest) -> PredictionResult:
    """Return a placeholder prediction until the RF pipeline is connected."""

    fv = request.feature_vector
    return PredictionResult(
        track_id=fv.track_id,
        label="uncertain",
        confidence=0.0,
        rule_filter=RuleFilterResult(passed=False, reject_reason="feature_error"),
        top_features={},
        quality=ResponseQuality(
            num_points=fv.quality.num_points if fv.quality else 0,
            track_stability=fv.quality.track_stability if fv.quality else "poor",
            feature_status=fv.feature_status,
        ),
        processing_time_ms=0,
    )
