"""MultiRocket Stack 서빙: 번들 검사, 창 점수 계산 경로, 판정 정책."""
import numpy as np
import pytest

from ai_server.classifiers.multirocket.classifier import (
    MultiRocketStackClassifier,
    check_bundle,
    scale_windows,
    window_scores,
)
from ai_server.classifiers.minirocket.contracts import TrackSequence
from research.minirocket_stack_training import scale_windows as research_scale_windows
from research.trajectory_sequence import SequenceConfig, window_track
from tests.model_parity_probe import make_track


@pytest.fixture(scope="module")
def classifier():
    return MultiRocketStackClassifier()


def test_bundle_contract(classifier):
    check_bundle(classifier.model)
    assert classifier.model["transform"] == "multirocket"
    assert classifier.model["held_out_data"] is None
    assert "multirocket-stack" in classifier.version


def test_scale_windows_matches_research():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(3, 2, 60)).astype(np.float32)
    x = np.concatenate([q, np.diff(q, axis=2, prepend=q[:, :, :1])], axis=1).astype(np.float32)
    scales = np.array([0.001, 0.01, 0.1])
    np.testing.assert_array_equal(scale_windows(x, scales, 400.0), research_scale_windows(x, scales))


def test_prediction_is_mean_of_stack_window_scores(classifier):
    track = TrackSequence.model_validate(make_track())
    result = classifier.predict(track)
    windows, _ = window_track(track.model_dump(mode="json"), SequenceConfig())
    expected = window_scores(classifier.model, np.stack([w["X"] for w in windows]),
                             np.array([w["normalization_scale"] for w in windows]))
    np.testing.assert_allclose(result.window_scores, expected)
    assert result.decision_score == pytest.approx(float(expected.mean()))
    assert result.label == ("drone" if expected.mean() >= 0 else "bird")


def test_non_contiguous_windows_give_same_scores(classifier):
    track = TrackSequence.model_validate(make_track())
    windows, _ = window_track(track.model_dump(mode="json"), SequenceConfig())
    x = np.stack([w["X"] for w in windows])
    scales = np.array([w["normalization_scale"] for w in windows])
    np.testing.assert_allclose(window_scores(classifier.model, np.asfortranarray(x), scales),
                               window_scores(classifier.model, np.ascontiguousarray(x), scales))


def test_short_track_abstains(classifier):
    result = classifier.predict(TrackSequence.model_validate(make_track(30)))
    assert result.label == "uncertain"
    assert result.abstain_reason == "insufficient_observation"


def test_rejects_wrong_bundle(classifier):
    bad = dict(classifier.model, bundle_version="other")
    with pytest.raises(ValueError):
        check_bundle(bad)
