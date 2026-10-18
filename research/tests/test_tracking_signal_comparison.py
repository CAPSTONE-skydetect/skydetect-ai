import copy

import numpy as np
import pytest

from ai_server.schemas import TrackPoint, TrackSequence
from research.features import FeatureConfig, extract_features
from research.tracking_signal_comparison import compare_signals


def _track(indices, *, bias=0., jitter=0.):
    return TrackSequence(
        track_id=0, processed_width=1280, processed_height=720,
        history=[TrackPoint(frame_index=i, timestamp_ms=2000+round(i/50*1000),
                            cx=(200+2*i+bias+jitter*(-1)**i)/1280,
                            cy=.3, w=.01, h=.01, conf=1.) for i in indices],
    )


def test_diagnostic_flag_preserves_all_existing_outputs():
    history = [p.model_dump() for p in _track(range(100), jitter=.8).history]
    original = copy.deepcopy(history)
    plain = extract_features(history, 1920, 1080)
    diagnostics = []
    traced = extract_features(history, 1920, 1080, diagnostics=diagnostics)
    assert diagnostics
    assert traced == plain
    assert history == original


def test_constant_bias_does_not_create_acceleration_error():
    result = compare_signals(_track(range(100), bias=5), _track(range(100)))
    assert result["summary"]["acceleration_error_rmse"] < 1e-8
    assert result["observed"][0]["error_x"] == pytest.approx(7.5)
    assert result["positions"][0]["time_seconds"] == 2.
    assert result["speed"][0]["time_seconds"] > 2.
    assert result["acceleration"][0]["time_seconds"] > result["speed"][0]["time_seconds"]


def test_signals_reproduce_b_acceleration_and_preserve_long_gaps():
    indices = list(range(50)) + list(range(75, 130))
    a = _track(indices, jitter=.8)
    result = compare_signals(a, _track(indices))
    assert {r["segment_id"] for r in result["acceleration"]} == {0, 1}
    assert not any(2.98 < r["time_seconds"] < 3.5 for r in result["positions"])
    diagnostics = []
    extract_features(
        [p.model_dump() for p in a.history], 1920, 1080,
        diagnostics=diagnostics,
    )
    for segment_id, segment in enumerate(diagnostics):
        speed = np.asarray(segment["speed_px_s"])
        expected = np.abs(np.diff(speed)) / segment["step_seconds"]
        actual = [row["a"] for row in result["acceleration"]
                  if row["segment_id"] == segment_id]
        np.testing.assert_allclose(expected, actual)
    assert result["summary"]["missing_frame_count"] == 25


def test_mismatched_frames_or_geometry_are_rejected():
    with pytest.raises(ValueError, match="exact A frames"):
        compare_signals(_track(range(50)), _track(range(1, 51)))
    gt = _track(range(50)).model_copy(update={"processed_width": 640})
    with pytest.raises(ValueError, match="geometry"):
        compare_signals(_track(range(50)), gt)


def test_short_gap_is_interpolated_only_in_b_not_observations():
    indices = [i for i in range(100) if i != 40]
    result = compare_signals(_track(indices, jitter=.4), _track(indices))
    assert len(result["observed"]) == 99
    assert not any(r["frame_index"] == 40 for r in result["observed"])
    assert len({r["segment_id"] for r in result["positions"]}) == 1
    assert any(39 < r["source_frame_position"] < 41 for r in result["positions"])


def test_rejected_trace_does_not_expose_partial_signals():
    diagnostics = []
    result = extract_features([p.model_dump() for p in _track(range(4)).history],
                              1920, 1080, diagnostics=diagnostics, config=FeatureConfig())
    assert result["feature_status"] == "rejected"
    assert diagnostics == []
