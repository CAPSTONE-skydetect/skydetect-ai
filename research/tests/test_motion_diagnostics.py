import numpy as np
import pytest

from research.measure_train_motion import train_tracks
from research.motion_diagnostics import MotionConfig, analyze_segment, resample_segments


def timeline(seconds=8):
    return np.arange(int(seconds*30)+1)/30


def test_straight_flight_does_not_create_turn_or_reversal():
    t = timeline()
    rng = np.random.default_rng(12)
    points = np.column_stack((.1+.03*t, np.full(len(t), .2)))+rng.normal(0, .1/1920, (len(t), 2))
    result = analyze_segment(t, points)
    assert not result["events"]
    assert not result["summary"]["unresolved"]


def test_stop_then_reverse_has_a_supported_pause_and_reversal():
    t = timeline()
    x = .1+.03*np.minimum(t, 3)-.03*np.maximum(t-4, 0)
    result = analyze_segment(t, np.column_stack((x, np.full(len(t), .2))))
    low = [e for e in result["events"] if e["kind"] == "low_motion"]
    reverse = [e for e in result["events"] if e["kind"] == "reversal"]
    assert len(low) == 1 and .5 < low[0]["duration_s"] < 1.3
    assert len(reverse) == 1 and reverse[0]["change_angle_deg"] > 170


def test_sustained_circle_is_a_turn_not_many_instant_reversals():
    t = timeline()
    points = np.column_stack((.3+.1*np.cos(.5*t), .3+.1*np.sin(.5*t)))
    result = analyze_segment(t, points)
    turns = [e for e in result["events"] if e["kind"] == "turn"]
    assert len(turns) == 1 and turns[0]["duration_s"] > 6
    assert not [e for e in result["events"] if e["kind"] == "reversal"]


def test_unresolved_stationary_or_tiny_motion_is_not_labeled_as_real_hover():
    t = timeline()
    result = analyze_segment(t, np.full((len(t), 2), .2))
    assert result["summary"]["unresolved"] and not result["events"]
    assert np.isfinite(list(result["summary"].values())).all()


def test_long_missing_interval_is_not_connected_for_event_detection():
    frames = np.r_[np.arange(60), np.arange(90, 180)]
    track = dict(processed_width=1920, processed_height=1080,
                 stabilization={"applied": True}, history=[
                     dict(timestamp_ms=int(round(f*1000/30)), frame_index=int(f), cx=.2+.001*f, cy=.4)
                     for f in frames])
    segments = resample_segments(track)
    assert len(segments) == 2
    assert segments[0]["time"][-1] < 2
    assert segments[1]["time"][0] == pytest.approx(3)


def test_invalid_clock_and_config_fail_explicitly():
    with pytest.raises(ValueError, match="regularly sampled"):
        analyze_segment(np.arange(30)*.04, np.full((30, 2), .2))
    with pytest.raises(ValueError, match="thresholds"):
        MotionConfig(reverse_angle_deg=70)


def test_training_loader_never_opens_validation_or_test(tmp_path):
    forbidden = tmp_path/"does-not-exist.json"
    manifest = dict(sources=[dict(split=split, status="accepted", source_path=str(forbidden))
                             for split in ("validation", "test")])
    assert list(train_tracks(manifest)) == []
