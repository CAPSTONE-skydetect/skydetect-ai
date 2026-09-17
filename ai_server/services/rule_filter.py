"""Part C: Rule-based pre-filter for FeatureVectors."""

from ai_server.schemas import FeatureVector, RuleFilterResult

_MIN_TRACK_LENGTH: int = 5
_MIN_MEAN_CONF: float = 0.4
_MAX_MISSING_RATIO: float = 0.5

# feature v4 기준 노이즈 임계값.
#
# 이 필터의 목적은 클래스를 가르는 것이 아니라, 추적이 실패했거나 ID가 뒤바뀐
# 병리적 트랙을 RF 진입 전에 거르는 것이다. 따라서 정상 궤적 분포의 최댓값
# 위쪽에 여유를 두고 잡는다.
#
# 근거: 시뮬레이터 4.0.0 / 피처 4.0.0 표본 100건(ideal 49, noisy 51)의 분포.
#   speed_cv      p99 0.805, max 1.187  -> 2.0 (최댓값의 약 1.7배)
#   turn_rate_p95 p99 8.404, max 11.010 -> 20.0 (rad/s. 초당 3회전 이상은 비물리적)
#
# 주의: 표본 100건은 임계값을 확정하기에 작다. 전체 학습 데이터셋이 피처 4.0.0으로
# 재생성되면 같은 분위수 기준으로 재산정해야 한다.
#
# speed_cv 는 기존 v_std / v_mean 규칙을 그대로 대체한다. 정의가 같아 별도
# 0 나눗셈 방어가 필요 없다. 다만 기존 임계값 3.0은 v4 분포에서 한 번도
# 발동하지 않으므로(최댓값 1.187) 그대로 쓸 수 없다.
_MAX_SPEED_CV: float = 2.0
_MAX_TURN_RATE_P95: float = 20.0


class RuleFilter:
    """RF 분류기 진입 전 노이즈·저품질 FeatureVector를 걸러내는 규칙 기반 필터."""

    def __init__(
        self,
        min_track_length: int = _MIN_TRACK_LENGTH,
        min_mean_conf: float = _MIN_MEAN_CONF,
        max_missing_ratio: float = _MAX_MISSING_RATIO,
        max_speed_cv: float = _MAX_SPEED_CV,
        max_turn_rate_p95: float = _MAX_TURN_RATE_P95,
    ) -> None:
        self.min_track_length = min_track_length
        self.min_mean_conf = min_mean_conf
        self.max_missing_ratio = max_missing_ratio
        self.max_speed_cv = max_speed_cv
        self.max_turn_rate_p95 = max_turn_rate_p95

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
        if features.speed_cv > self.max_speed_cv:
            return True
        if features.turn_rate_p95 > self.max_turn_rate_p95:
            return True
        return False
