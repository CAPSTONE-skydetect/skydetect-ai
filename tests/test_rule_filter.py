import pytest

from ai_server.schemas import FeatureVector, TrackFeatures, TrackQuality
from ai_server.services.rule_filter import RuleFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# feature v4 9종의 정상 범위 기본값. 시뮬레이터 4.0.0 표본의 중앙값을 사용한다.
# 개별 테스트는 검사하려는 필드만 키워드로 덮어쓴다. 피처가 늘거나 줄어도
# 이 사전 한 곳만 고치면 되도록 **overrides 형태로 받는다.
_DEFAULT_FEATURES: dict[str, float] = {
    "speed_median": 74.67,
    "speed_cv": 0.21,
    "acceleration_median": 27.89,
    "acceleration_p95": 76.84,
    "turn_rate_median": 0.25,
    "turn_rate_p95": 0.94,
    "curvature_cv": 0.93,
    "tortuosity": 1.01,
    "heading_change_ratio": 0.19,
}


def _make_features(**overrides: float) -> TrackFeatures:
    unknown = set(overrides) - set(_DEFAULT_FEATURES)
    assert not unknown, f"알 수 없는 피처 이름: {sorted(unknown)}"
    return TrackFeatures(**{**_DEFAULT_FEATURES, **overrides})


def _make_quality(
    num_points: int = 10,
    mean_conf: float = 0.85,
    missing_ratio: float = 0.05,
) -> TrackQuality:
    if num_points >= 4 and mean_conf >= 0.85:
        stability = "good"
    elif num_points >= 3 and mean_conf >= 0.70:
        stability = "fair"
    else:
        stability = "poor"
    return TrackQuality(
        num_points=num_points,
        mean_conf=mean_conf,
        missing_ratio=missing_ratio,
        track_stability=stability,
    )


def _make_fv(
    feature_status: str = "ok",
    num_points: int = 10,
    mean_conf: float = 0.85,
    missing_ratio: float = 0.05,
    include_quality: bool = True,
    **feature_overrides: float,
) -> FeatureVector:
    features = None if feature_status == "failed" else _make_features(**feature_overrides)
    quality = _make_quality(num_points, mean_conf, missing_ratio) if include_quality else None
    return FeatureVector(
        track_id=1,
        features=features,
        quality=quality,
        feature_status=feature_status,
    )


# ---------------------------------------------------------------------------
# Tests: pass
# ---------------------------------------------------------------------------

class TestPass:
    def test_good_track_passes(self):
        result = RuleFilter().apply(_make_fv())
        assert result.passed is True
        assert result.reject_reason is None

    def test_partial_status_passes_when_quality_ok(self):
        fv = FeatureVector(
            track_id=1,
            features=_make_features(),
            quality=_make_quality(),
            feature_status="partial",
            imputed_fields=["speed_cv"],
        )
        result = RuleFilter().apply(fv)
        assert result.passed is True

    def test_quality_none_is_rejected_as_feature_error(self):
        fv = _make_fv(include_quality=False)
        result = RuleFilter().apply(fv)
        assert result.passed is False
        assert result.reject_reason == "feature_error"


# ---------------------------------------------------------------------------
# Tests: feature_error
# ---------------------------------------------------------------------------

class TestFeatureError:
    def test_failed_status_is_rejected(self):
        result = RuleFilter().apply(_make_fv(feature_status="failed"))
        assert result.passed is False
        assert result.reject_reason == "feature_error"

    def test_feature_error_takes_priority_over_short_track(self):
        result = RuleFilter().apply(_make_fv(feature_status="failed", num_points=1))
        assert result.reject_reason == "feature_error"


# ---------------------------------------------------------------------------
# Tests: short_track
# ---------------------------------------------------------------------------

class TestShortTrack:
    def test_below_min_length_is_rejected(self):
        result = RuleFilter().apply(_make_fv(num_points=3))
        assert result.passed is False
        assert result.reject_reason == "short_track"

    def test_exact_min_length_passes(self):
        result = RuleFilter(min_track_length=5).apply(_make_fv(num_points=5))
        assert result.passed is True

    def test_short_track_takes_priority_over_low_confidence(self):
        result = RuleFilter().apply(_make_fv(num_points=2, mean_conf=0.1))
        assert result.reject_reason == "short_track"

    def test_override_min_track_length(self):
        fv = _make_fv(num_points=3)
        result = RuleFilter().apply(fv, min_track_length=3)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Tests: low_confidence
# ---------------------------------------------------------------------------

class TestLowConfidence:
    def test_below_min_conf_is_rejected(self):
        result = RuleFilter().apply(_make_fv(mean_conf=0.3))
        assert result.passed is False
        assert result.reject_reason == "low_confidence"

    def test_exact_min_conf_passes(self):
        result = RuleFilter(min_mean_conf=0.4).apply(_make_fv(mean_conf=0.4))
        assert result.passed is True

    def test_override_min_mean_conf(self):
        fv = _make_fv(mean_conf=0.3)
        result = RuleFilter().apply(fv, min_mean_conf=0.3)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Tests: high_noise — missing ratio
# ---------------------------------------------------------------------------

class TestHighNoiseMissingRatio:
    def test_high_missing_ratio_is_rejected(self):
        result = RuleFilter().apply(_make_fv(missing_ratio=0.6))
        assert result.passed is False
        assert result.reject_reason == "high_noise"

    def test_exact_max_missing_ratio_passes(self):
        result = RuleFilter(max_missing_ratio=0.5).apply(_make_fv(missing_ratio=0.5))
        assert result.passed is True

    def test_override_max_missing_ratio(self):
        fv = _make_fv(missing_ratio=0.6)
        result = RuleFilter().apply(fv, max_missing_ratio=0.7)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Tests: high_noise — feature-based
# ---------------------------------------------------------------------------

class TestHighNoiseFeatures:
    def test_high_speed_cv_is_rejected(self):
        # 기본 임계값 2.0 초과
        result = RuleFilter().apply(_make_fv(speed_cv=2.1))
        assert result.passed is False
        assert result.reject_reason == "high_noise"

    def test_speed_cv_at_threshold_passes(self):
        # 경계값은 통과한다 (strictly greater 비교)
        result = RuleFilter().apply(_make_fv(speed_cv=2.0))
        assert result.passed is True

    def test_observed_maximum_speed_cv_passes(self):
        # 시뮬레이터 4.0.0 표본의 speed_cv 최댓값 1.187. 정상 궤적은 걸러지면 안 된다.
        result = RuleFilter().apply(_make_fv(speed_cv=1.187))
        assert result.passed is True

    def test_custom_max_speed_cv(self):
        result = RuleFilter(max_speed_cv=0.5).apply(_make_fv(speed_cv=0.6))
        assert result.passed is False
        assert result.reject_reason == "high_noise"

    def test_high_turn_rate_p95_is_rejected(self):
        # 기본 임계값 20.0 rad/s 초과. 추적 실패나 ID 교체를 시사한다.
        result = RuleFilter().apply(_make_fv(turn_rate_p95=20.1))
        assert result.passed is False
        assert result.reject_reason == "high_noise"

    def test_turn_rate_p95_at_threshold_passes(self):
        result = RuleFilter().apply(_make_fv(turn_rate_p95=20.0))
        assert result.passed is True

    def test_observed_maximum_turn_rate_p95_passes(self):
        # 시뮬레이터 4.0.0 표본의 turn_rate_p95 최댓값 11.010.
        result = RuleFilter().apply(_make_fv(turn_rate_p95=11.010))
        assert result.passed is True

    def test_custom_max_turn_rate_p95(self):
        result = RuleFilter(max_turn_rate_p95=10.0).apply(_make_fv(turn_rate_p95=11.0))
        assert result.passed is False
        assert result.reject_reason == "high_noise"

    def test_quality_none_rejected_before_feature_noise_check(self):
        fv = _make_fv(include_quality=False)
        result = RuleFilter().apply(fv)
        assert result.passed is False
        assert result.reject_reason == "feature_error"
