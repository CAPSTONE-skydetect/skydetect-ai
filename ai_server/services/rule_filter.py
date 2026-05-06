"""Part C: Rule-based pre-filter for FeatureVectors."""

from ai_server.schemas import FeatureVector, RuleFilterResult

_MIN_TRACK_LENGTH: int = 5
_MIN_MEAN_CONF: float = 0.4
_MAX_MISSING_RATIO: float = 0.5
_MAX_VELOCITY_CV: float = 3.0       # v_std / v_mean
_MAX_MANEUVERABILITY_SIGMA: float = 30.0


class RuleFilter:
    """RF 분류기 진입 전 노이즈·저품질 FeatureVector를 걸러내는 규칙 기반 필터."""

    def __init__(
        self,
        min_track_length: int = _MIN_TRACK_LENGTH,
        min_mean_conf: float = _MIN_MEAN_CONF,
        max_missing_ratio: float = _MAX_MISSING_RATIO,
        max_velocity_cv: float = _MAX_VELOCITY_CV,
        max_maneuverability_sigma: float = _MAX_MANEUVERABILITY_SIGMA,
    ) -> None:
        self.min_track_length = min_track_length
        self.min_mean_conf = min_mean_conf
        self.max_missing_ratio = max_missing_ratio
        self.max_velocity_cv = max_velocity_cv
        self.max_maneuverability_sigma = max_maneuverability_sigma

    def apply(
        self,
        fv: FeatureVector,
        *,
        min_track_length: int | None = None,
        min_mean_conf: float | None = None,
        max_missing_ratio: float | None = None,
    ) -> RuleFilterResult:
        """FeatureVector에 규칙을 순서대로 적용해 RuleFilterResult를 반환한다.

        검사 순서:
            1. feature_error  — feature 계산 자체 실패
            2. short_track    — 트랙 길이 부족
            3. low_confidence — 평균 탐지 신뢰도 부족
            4. high_noise     — 누락 비율 초과 또는 신호 노이즈 과다
        """
        eff_min_length = min_track_length if min_track_length is not None else self.min_track_length
        eff_min_conf = min_mean_conf if min_mean_conf is not None else self.min_mean_conf
        eff_max_missing = max_missing_ratio if max_missing_ratio is not None else self.max_missing_ratio

        if fv.feature_status == "failed":
            return RuleFilterResult(passed=False, reject_reason="feature_error")

        if fv.quality is None:
            return RuleFilterResult(passed=False, reject_reason="feature_error")

        if fv.quality.num_points < eff_min_length:
            return RuleFilterResult(passed=False, reject_reason="short_track")
        if fv.quality.mean_conf < eff_min_conf:
            return RuleFilterResult(passed=False, reject_reason="low_confidence")
        if fv.quality.missing_ratio > eff_max_missing:
            return RuleFilterResult(passed=False, reject_reason="high_noise")

        if fv.features is not None and self._features_are_noisy(fv.features):
            return RuleFilterResult(passed=False, reject_reason="high_noise")

        return RuleFilterResult(passed=True)

    def _features_are_noisy(self, features) -> bool:
        if features.v_mean > 1e-6 and (features.v_std / features.v_mean) > self.max_velocity_cv:
            return True
        if features.maneuverability_sigma > self.max_maneuverability_sigma:
            return True
        return False
