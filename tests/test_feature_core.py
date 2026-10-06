"""A → B → C 연결 지점 테스트.

피처 계산식 자체는 research/tests/test_features.py 가 검증한다. 여기서는 그 결과를
TrackSequence에서 뽑아 FeatureVector로 옮기는 변환만 본다.
"""

import numpy as np
import pytest

from ai_server.schemas import TrackPoint, TrackQuality, TrackSequence
from ai_server.services.feature_core import build_feature_vector
from research.features import FEATURE_COLUMNS as FEATURE_NAMES
from research.features import FeatureConfig

FPS = 30


def _track(
    point_fn,
    *,
    count=90,
    track_id=1,
    fps=FPS,
    resolution=(1280, 720),
):
    history = [
        TrackPoint(
            frame_index=i,
            timestamp_ms=round(i * 1000 / fps),
            conf=0.9,
            w=0.03,
            h=0.03,
            **point_fn(i),
        )
        for i in range(count)
    ]
    return TrackSequence(
        track_id=track_id,
        source_video_id="test",
        processed_width=resolution[0],
        processed_height=resolution[1],
        history=history,
    )


def _straight(i):
    return {"cx": 0.2 + 0.006 * i, "cy": 0.5 + 0.0005 * i}


def test_extracts_all_nine_features_the_model_expects():
    result = build_feature_vector(_track(_straight))

    assert result.accepted
    features = result.feature_vector.features
    assert features is not None
    # 학습 피처 목록을 단일 진실 공급원으로 삼아 비교한다. 목록이 바뀌면 여기서 잡힌다.
    for name in FEATURE_NAMES:
        assert getattr(features, name) is not None


def test_perfectly_straight_track_does_not_fail_tortuosity_lower_bound():
    """직선 궤적의 tortuosity는 1 미만으로 내려가면 안 된다.

    이동거리/변위라 수학적으로 항상 1 이상인데, 완전한 직선에서는 부동소수점
    오차로 0.9999999999999999 가 나와 ge=1.0 스키마 검증에서 터졌다.
    """
    result = build_feature_vector(_track(lambda i: {"cx": 0.2 + 0.006 * i, "cy": 0.5}))

    assert result.accepted, result.reasons
    assert result.feature_vector.features.tortuosity >= 1.0


def test_short_track_is_rejected_with_a_reason_instead_of_raising():
    result = build_feature_vector(_track(_straight, count=4))

    assert not result.accepted
    assert result.feature_vector.feature_status == "failed"
    assert result.feature_vector.features is None
    # 사유가 없으면 UI가 "판정 불가"만 띄우고 원인을 설명할 수 없다.
    assert "insufficient_points" in result.reasons


def test_track_id_and_quality_carry_through_to_the_classifier():
    track = _track(_straight, track_id=7)
    result = build_feature_vector(track)

    assert result.feature_vector.track_id == 7
    assert result.feature_vector.quality is not None
    assert result.feature_vector.quality.num_points == len(track.history)


@pytest.mark.parametrize(
    "resolution",
    [(640, 360), (1280, 720), (1920, 1080), (3840, 2160)],
)
def test_runtime_features_use_full_hd_canonical_coordinates(resolution):
    result = build_feature_vector(_track(_straight, resolution=resolution))
    reference = build_feature_vector(
        _track(_straight, resolution=(1920, 1080))
    )

    assert result.accepted
    assert result.coordinate_policy == "fhd_width_1920_v1"
    assert result.timebase_policy == "timestamp_ms_priority_30hz_resample_v1"
    assert result.canonical_width == 1920.0
    assert result.canonical_height == 1080.0
    assert result.coordinate_scale == pytest.approx(1920 / resolution[0])
    assert result.raw_quality["aspect_ratio_matches_training"] is True
    assert result.raw_quality["target_feature_fps"] == 30.0
    assert result.feature_vector.provenance is not None
    assert result.feature_vector.provenance.coordinate_policy == result.coordinate_policy
    assert result.feature_vector.provenance.timebase_policy == result.timebase_policy
    assert result.feature_vector.provenance.feature_config_id == result.feature_config_id
    assert (
        result.feature_vector.provenance.feature_contract_id
        == result.feature_contract_id
        == result.raw_quality["feature_contract_id"]
    )
    assert result.feature_vector.features.model_dump() == pytest.approx(
        reference.feature_vector.features.model_dump(),
        rel=1e-10,
        abs=1e-10,
    )
    assert result.feature_vector.features.speed_median == pytest.approx(
        346.0, rel=0.002
    )


def test_non_16_by_9_input_is_warning_only_and_preserves_uniform_scale():
    result = build_feature_vector(_track(_straight, resolution=(320, 200)))

    assert result.accepted
    assert result.canonical_width == 1920.0
    assert result.canonical_height == 1200.0
    assert result.feature_vector.provenance is not None
    assert result.feature_vector.provenance.aspect_ratio_matches_training is False


def test_feature_contract_id_covers_timebase_policy_not_only_formula_config():
    at_30 = build_feature_vector(_track(_straight), config=FeatureConfig(target_fps=30))
    at_60 = build_feature_vector(_track(_straight), config=FeatureConfig(target_fps=60))

    assert at_30.feature_config_id != at_60.feature_config_id
    assert at_30.timebase_policy == "timestamp_ms_priority_30hz_resample_v1"
    assert at_60.timebase_policy == "timestamp_ms_priority_60hz_resample_v1"
    assert (
        at_30.feature_vector.provenance.feature_contract_id
        != at_60.feature_vector.provenance.feature_contract_id
    )


def test_roi_start_offset_does_not_create_pre_roi_or_post_track_motion():
    baseline = _track(_straight)
    shifted = baseline.model_copy(
        update={
            "history": [
                point.model_copy(
                    update={
                        "frame_index": point.frame_index + 120,
                        "timestamp_ms": point.timestamp_ms + 4000,
                    }
                )
                for point in baseline.history
            ]
        }
    )

    baseline_result = build_feature_vector(baseline)
    shifted_result = build_feature_vector(shifted)

    assert shifted_result.accepted
    assert shifted_result.feature_vector.features.model_dump() == pytest.approx(
        baseline_result.feature_vector.features.model_dump()
    )
    assert shifted_result.raw_quality["duration_seconds"] == pytest.approx(
        baseline_result.raw_quality["duration_seconds"]
    )


def test_long_lost_gap_is_split_instead_of_interpolated():
    track = _track(_straight, count=120)
    track = track.model_copy(
        update={
            "history": [
                point
                for point in track.history
                if not 40 <= point.frame_index < 65
            ]
        }
    )

    result = build_feature_vector(track)

    assert result.accepted
    assert result.raw_quality["long_gap_count"] == 1
    assert result.raw_quality["usable_segments"] == 2
    assert result.raw_quality["retained_duration_seconds"] < 3.3


def test_a_track_quality_has_priority_over_b_fallback_quality():
    quality = TrackQuality(
        num_points=90,
        mean_conf=0.77,
        missing_ratio=0.25,
        track_stability="fair",
    )
    track = _track(_straight).model_copy(update={"quality": quality})

    result = build_feature_vector(track)

    assert result.feature_vector.quality == quality


def test_30_and_60_fps_curved_tracks_produce_equivalent_features():
    at_30 = build_feature_vector(_curved_track(30))
    at_60 = build_feature_vector(_curved_track(60))
    assert at_30.accepted and at_60.accepted

    features_30 = at_30.feature_vector.features
    features_60 = at_60.feature_vector.features
    assert features_60.speed_median == pytest.approx(
        features_30.speed_median, rel=0.01
    )
    assert features_60.speed_cv == pytest.approx(features_30.speed_cv, rel=0.02)
    assert features_60.acceleration_median == pytest.approx(
        features_30.acceleration_median, rel=0.04
    )
    assert features_60.acceleration_p95 == pytest.approx(
        features_30.acceleration_p95, rel=0.10
    )
    assert features_60.turn_rate_median == pytest.approx(
        features_30.turn_rate_median, abs=0.03
    )
    assert features_60.turn_rate_p95 == pytest.approx(
        features_30.turn_rate_p95, rel=0.02
    )
    assert features_60.curvature_cv == pytest.approx(
        features_30.curvature_cv, rel=0.05
    )
    assert features_60.tortuosity == pytest.approx(
        features_30.tortuosity, rel=0.01
    )
    assert features_60.heading_change_ratio == pytest.approx(
        features_30.heading_change_ratio, abs=0.03
    )
    assert at_30.raw_quality["clock"] == "timestamp_ms"
    assert at_60.raw_quality["clock"] == "timestamp_ms"


def _curved_track(fps: int, duration: float = 5.0) -> TrackSequence:
    history = []
    for frame_index in range(round(duration * fps) + 1):
        time_seconds = frame_index / fps
        history.append(
            TrackPoint(
                frame_index=frame_index,
                timestamp_ms=round(time_seconds * 1000),
                cx=(420 + 95 * time_seconds + 40 * time_seconds**2) / 1920,
                cy=(500 + 150 * np.sin(0.8 * time_seconds)) / 1080,
                w=0.03,
                h=0.03,
                conf=0.9,
            )
        )
    return TrackSequence(
        track_id=1,
        source_video_id=f"curved-{fps}",
        processed_width=1920,
        processed_height=1080,
        history=history,
    )
