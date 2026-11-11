"""C파트 MiniRocket 분류기 테스트.

A TrackSequence → 2초 창 → MiniRocket + Ridge → PredictionResult 경로와
학습/평가 공통 모듈의 누수·계약 검사를 확인한다.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai_server.main import create_app
from ai_server.schemas import (
    ClassifyRequest,
    PredictionResult,
    StabilizationInfo,
    TrackPoint,
    TrackSequence,
)
from ai_server.services import sequence_model as sm
from ai_server.services.prediction import classify_track_sequence, get_classifier

FPS = 30


def _track(seconds=5.0, *, track_id=1, stabilized=True, resolution=(1280, 720), drop=None):
    history = []
    for i in range(int(seconds * FPS)):
        if drop and drop[0] <= i < drop[1]:
            continue
        t = i / FPS
        history.append(TrackPoint(
            frame_index=i, timestamp_ms=round(t * 1000), conf=0.9, w=0.02, h=0.02,
            cx=0.3 + 0.05 * t, cy=0.5 + 0.02 * np.sin(4 * t),
        ))
    return TrackSequence(
        track_id=track_id, source_video_id="test",
        processed_width=resolution[0], processed_height=resolution[1],
        history=history,
        stabilization=StabilizationInfo(applied=stabilized, method="opencv_feature_cmc" if stabilized else "none"),
    )


def _classify(track, **kwargs) -> PredictionResult:
    return classify_track_sequence(ClassifyRequest(track_sequence=track, **kwargs))


def test_model_bundle_matches_preprocessing_contract():
    model = get_classifier().model
    sm.check_bundle(model)
    assert model["classes"] == ["bird", "drone"]
    assert model["score_is_probability"] is False
    assert model["validation_used_for_fit"] is False


def test_long_track_is_classified_with_margin_not_probability():
    result = _classify(_track(5.0))
    assert result.label in {"bird", "drone"}
    assert result.abstain_reason is None
    assert result.score_is_probability is False
    # 150프레임 = 관측 4.967초 → 시작 0,1,2초의 2초 창 3개. 모자란 구간을 늘여 채우지 않는다.
    assert result.windows_used == 3
    assert result.decision_score == pytest.approx(np.mean(result.window_scores))
    assert (result.decision_score >= 0) == (result.label == "drone")


def test_short_track_abstains_instead_of_stretching():
    result = _classify(_track(1.5))
    assert result.label == "uncertain"
    assert result.abstain_reason == "insufficient_observation"
    assert result.decision_score is None
    assert result.window_rejections == {"short_track": 1}


def test_unstabilized_track_is_rejected_as_invalid_input():
    result = _classify(_track(5.0, stabilized=False))
    assert result.label == "uncertain"
    assert result.abstain_reason == "invalid_input"


def test_long_gap_windows_are_excluded_not_zero_filled():
    full = _classify(_track(6.0))
    gapped = _classify(_track(6.0, drop=(60, 75)))  # 0.5초 연속 누락
    assert gapped.windows_used < full.windows_used
    assert gapped.window_rejections.get("long_gap", 0) > 0


def test_window_start_times_follow_scores_and_skip_rejected_windows():
    full = _classify(_track(5.0))
    assert full.window_starts_s == [0.0, 1.0, 2.0]

    full6 = _classify(_track(6.0))
    gapped = _classify(_track(6.0, drop=(60, 75)))
    assert len(gapped.window_starts_s) == len(gapped.window_scores)
    # 제외된 창의 시각은 빠지고, 남은 창의 시각만 원래 자리 그대로 남는다.
    assert set(gapped.window_starts_s) < set(full6.window_starts_s)


def test_margin_threshold_turns_weak_decision_into_low_separation():
    base = _classify(_track(5.0))
    gated = _classify(_track(5.0), margin_threshold=abs(base.decision_score) + 1.0)
    assert gated.label == "uncertain"
    assert gated.abstain_reason == "low_separation"
    assert gated.decision_score == pytest.approx(base.decision_score)


def test_min_windows_requires_enough_observation():
    result = _classify(_track(3.0), min_windows=5)
    assert result.abstain_reason == "insufficient_observation"


def test_prediction_is_invariant_to_resolution_with_same_aspect():
    scores = [_classify(_track(5.0, resolution=r)).decision_score
              for r in [(640, 360), (1280, 720), (1920, 1080)]]
    assert scores == pytest.approx([scores[0]] * len(scores), abs=1e-4)


def test_classify_endpoint_round_trip():
    client = TestClient(create_app())
    payload = ClassifyRequest(track_sequence=_track(4.0, track_id=9)).model_dump(mode="json")
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["track_id"] == 9
    assert body["score_type"] == "ridge_margin_mean"
    assert body["score_is_probability"] is False


def test_prediction_result_rejects_inconsistent_abstain_state():
    with pytest.raises(ValueError):
        PredictionResult(track_id=1, label="uncertain", model_version="x")
    with pytest.raises(ValueError):
        PredictionResult(track_id=1, label="bird", abstain_reason="low_separation", model_version="x")


def test_validate_windows_rejects_contract_violations():
    x = np.zeros((2, 4, 60), dtype=np.float32)
    sm.validate_windows(x)
    with pytest.raises(ValueError):
        sm.validate_windows(x.astype(np.float64))
    with pytest.raises(ValueError):
        sm.validate_windows(np.zeros((2, 4, 120), dtype=np.float32))
    broken = x.copy()
    broken[0, 2, 5] = 1.0  # 변위 채널이 좌표 차분과 다름
    with pytest.raises(ValueError):
        sm.validate_windows(broken)


def test_aggregation_averages_windows_then_tracks_per_group():
    data = dict(
        sample_id=np.array(["track-a:w0000", "track-a:w0001", "track-a:w0002", "track-b:w0000"]),
        group_id=np.array(["g1", "g1", "g1", "g1"]),
        y=np.array(["drone"] * 4),
    )
    _, tracks, groups = sm.aggregate_scores(data, np.array([3.0, 3.0, 3.0, -1.0]))
    assert sorted(tracks.decision) == [-1.0, 3.0]
    # 창이 3개인 track-a 가 그룹 점수를 지배하지 않는다: (3 + -1) / 2
    assert groups.decision.iloc[0] == pytest.approx(1.0)


def test_balanced_group_weights_equalize_class_and_group_mass():
    y = np.array(["bird", "bird", "bird", "drone"])
    groups = np.array(["g1", "g1", "g2", "g3"])
    w = sm.balanced_group_weights(y, groups)
    assert w.sum() == pytest.approx(len(y))
    assert w[y == "bird"].sum() == pytest.approx(w[y == "drone"].sum())
    assert w[groups == "g1"].sum() == pytest.approx(w[groups == "g2"].sum())


def test_abstain_metrics_keep_abstentions_in_recall_denominator():
    import pandas as pd

    table = pd.DataFrame(dict(label=["bird", "bird", "drone", "drone"], decision=[-2.0, -0.1, 2.0, 0.1]))
    result = sm.metrics(table, margin_threshold=0.5)
    assert result["abstain"]["coverage"] == pytest.approx(0.5)
    assert result["abstain"]["decided_accuracy"] == pytest.approx(1.0)
    assert result["abstain"]["recall_with_abstain"] == {"bird": 0.5, "drone": 0.5}


@pytest.mark.skipif(not sm.DEFAULT_PACKAGE.exists(), reason="B data package not unpacked")
def test_package_loader_enforces_no_train_validation_overlap():
    _, train, validation = sm.load_package()
    for data in train.values():
        assert not set(data["group_id"]) & set(validation["group_id"])
