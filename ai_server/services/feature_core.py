"""Part B core feature extraction.

A가 만든 TrackSequence를 C가 소비하는 FeatureVector로 변환한다.

계산식은 여기서 다시 구현하지 않고 research.features.extract_features 를 그대로
호출한다. B파트 연구 코드와 서버가 같은 공식을 쓰도록 강제하기 위해서다. 과거에
학습 피처 목록을 추론 쪽에 따로 나열해 두었다가 두 곳이 어긋나 추론 시점에야
터진 적이 있다(classifier._to_array 주석 참고). 같은 실수를 계산식 층에서
반복하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_server.schemas import (
    FeatureStatus,
    FeatureVector,
    TrackFeatures,
    TrackQuality,
    TrackSequence,
)
from ai_server.utils.quality import build_track_quality
from research.features import FeatureConfig, extract_features

# research 쪽 상태값과 스키마 쪽 FeatureStatus 는 어휘가 다르다.
# research 는 "표본을 학습에 채택할 것인가"를, 스키마는 "피처가 쓸 수 있는가"를
# 뜻한다. partial 은 research 가 만들지 않는다. 일부만 계산되는 상태를 두지 않고
# 샘플 전체를 거부하는 설계이기 때문이다.
_STATUS_MAP: dict[str, FeatureStatus] = {
    "accepted": "ok",
    "rejected": "failed",
}


@dataclass
class FeatureExtractionResult:
    """FeatureVector와 함께, 거부 사유 등 진단 정보를 같이 돌려준다.

    FeatureVector 스키마에는 거부 사유를 담을 자리가 없다. 그렇다고 사유를 버리면
    UI에서 "판정 불가"만 뜨고 왜 그런지 알 수 없다. 그래서 래퍼로 함께 전달한다.
    """

    feature_vector: FeatureVector
    feature_version: str
    feature_config_id: str
    reasons: list[str] = field(default_factory=list)
    raw_quality: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.feature_vector.feature_status == "ok"


def build_feature_vector(
    track: TrackSequence,
    *,
    fps: float | None = None,
    config: FeatureConfig | None = None,
) -> FeatureExtractionResult:
    """TrackSequence에서 feature v4 9종을 계산해 FeatureVector로 포장한다.

    Args:
        track: A파트 추적 결과.
        fps: timestamp_ms가 전부 비어 있을 때만 쓰는 명시적 대체값.
            TrackPoint는 timestamp_ms를 필수로 요구하므로 보통은 쓰이지 않는다.
        config: 피처 계산 설정. 생략하면 research 기본값을 쓴다.

    Raises:
        아무것도 올리지 않는다. 계산 실패는 feature_status="failed" 와 reasons로
        표현한다. 추적은 됐는데 피처만 못 뽑은 상황은 예외 상황이 아니라 정상적인
        분석 결과 중 하나이고, 호출부가 그대로 사용자에게 보여줘야 하기 때문이다.
    """
    history = [
        {
            "frame_index": point.frame_index,
            "timestamp_ms": point.timestamp_ms,
            "cx": point.cx,
            "cy": point.cy,
            "w": point.w,
            "h": point.h,
            "conf": point.conf,
        }
        for point in track.history
    ]

    result = extract_features(
        history,
        image_width=track.processed_width,
        image_height=track.processed_height,
        fps=fps,
        config=config,
    )

    status = _STATUS_MAP.get(result["feature_status"], "failed")
    features = (
        TrackFeatures(**_snap_boundaries(result["features"]))
        if status == "ok" and result["features"]
        else None
    )

    return FeatureExtractionResult(
        feature_vector=FeatureVector(
            track_id=track.track_id,
            features=features,
            quality=_resolve_quality(track, result["quality"]),
            feature_status=status,
        ),
        feature_version=result["feature_version"],
        feature_config_id=result["feature_config_id"],
        reasons=list(result["reasons"]),
        raw_quality=dict(result["quality"]),
    )


def _snap_boundaries(features: dict[str, Any]) -> dict[str, Any]:
    """정의상 하한에 딱 걸리는 값의 부동소수점 오차를 하한으로 붙인다.

    tortuosity는 이동거리/변위라 수학적으로 항상 1 이상이다. 그런데 완전한 직선
    궤적에서는 두 값이 같아져 0.9999999999999999 처럼 1 ULP 아래로 떨어지고,
    ge=1.0인 스키마가 이를 거부한다. 실제로 드론의 직선 비행에서 재현된다.

    오차 한계를 1e-9로 잡았다. 이보다 크게 벗어난 값은 표현 오차가 아니라 계산
    오류이므로 덮지 않고 그대로 검증에서 터지게 둔다.
    """
    tortuosity = features.get("tortuosity")
    if tortuosity is not None and 1.0 - 1e-9 < tortuosity < 1.0:
        return {**features, "tortuosity": 1.0}
    return features


def _resolve_quality(
    track: TrackSequence,
    research_quality: dict[str, Any],
) -> TrackQuality | None:
    """C의 RuleFilter가 보는 품질 정보를 고른다.

    A가 계산한 quality를 우선한다. A는 시도한 프레임 수를 알고 있어서 결측률을
    정확히 내지만, research 쪽은 관측된 frame_index 범위만 보므로 추적이 중간에
    끊긴 구간을 결측으로 세지 못한다. A의 값이 없을 때만 research 값으로 메운다.
    """
    if track.quality is not None:
        return track.quality
    if track.history:
        return build_track_quality(track.history)
    if not research_quality:
        return None

    # history가 비어 있으면 build_track_quality가 거부하므로 여기까지 오지 않는다.
    return None
