import pytest

from ai_server.schemas import FeatureVector, TrackFeatures, TrackQuality
from ai_server.services.rule_filter import RuleFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(
    v_mean: float = 10.0,
    v_std: float = 1.0,
    a_mean: float = 0.5,
    heading_change_ratio: float = 0.2,
    maneuverability_sigma: float = 5.0,
) -> TrackFeatures:
    return TrackFeatures(
        v_mean=v_mean,
        v_std=v_std,
        a_mean=a_mean,
        heading_change_ratio=heading_change_ratio,
        maneuverability_sigma=maneuverability_sigma,
    )


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
    v_mean: float = 10.0,
    v_std: float = 1.0,
    a_mean: float = 0.5,
    heading_change_ratio: float = 0.2,
    maneuverability_sigma: float = 5.0,
    include_quality: bool = True,
) -> FeatureVector:
    features = (
        None
        if feature_status == "failed"
        else _make_features(v_mean, v_std, a_mean, heading_change_ratio, maneuverability_sigma)
    )
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
            imputed_fields=["v_std"],
        )
        result = RuleFilter().apply(fv)
        assert result.passed is True

    def test_quality_none_skips_quality_checks(self):
        fv = _make_fv(include_quality=False)
        result = RuleFilter().apply(fv)
        assert result.passed is True


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
    def test_high_velocity_cv_is_rejected(self):
        # v_std / v_mean = 35.0 / 10.0 = 3.5  > default 3.0
        result = RuleFilter().apply(_make_fv(v_mean=10.0, v_std=35.0))
        assert result.passed is False
        assert result.reject_reason == "high_noise"

    def test_velocity_cv_at_threshold_passes(self):
        # v_std / v_mean = 30.0 / 10.0 = 3.0  == default 3.0 (not strictly greater)
        result = RuleFilter().apply(_make_fv(v_mean=10.0, v_std=30.0))
        assert result.passed is True

    def test_zero_v_mean_skips_cv_check(self):
        result = RuleFilter().apply(_make_fv(v_mean=0.0, v_std=100.0))
        assert result.passed is True

    def test_high_maneuverability_sigma_is_rejected(self):
        result = RuleFilter().apply(_make_fv(maneuverability_sigma=31.0))
        assert result.passed is False
        assert result.reject_reason == "high_noise"

    def test_maneuverability_sigma_at_threshold_passes(self):
        result = RuleFilter().apply(_make_fv(maneuverability_sigma=30.0))
        assert result.passed is True

    def test_custom_max_maneuverability_sigma(self):
        result = RuleFilter(max_maneuverability_sigma=10.0).apply(_make_fv(maneuverability_sigma=11.0))
        assert result.passed is False
        assert result.reject_reason == "high_noise"

    def test_feature_noise_skipped_when_quality_none_and_features_clean(self):
        fv = _make_fv(include_quality=False)
        assert RuleFilter().apply(fv).passed is True
