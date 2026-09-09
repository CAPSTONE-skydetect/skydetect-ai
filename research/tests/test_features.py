import copy

import numpy as np
import pytest

from research.features import FeatureConfig, extract_features


def line(fps=30, duration=4., vx=20., vy=0., width=20., image=(1920, 1080)):
    return [dict(frame_index=i, timestamp_ms=i*1000/fps,
                 cx=(300+vx*i/fps)/image[0], cy=(300+vy*i/fps)/image[1],
                 w=width/image[0], h=10/image[1], conf=.9)
            for i in range(int(duration*fps)+1)]


def feature(history, **kwargs):
    result = extract_features(history, 1920, 1080, **kwargs)
    assert result["feature_status"] == "accepted", result
    return result["features"]


@pytest.mark.parametrize("fps", [15, 24, 30, 50, 60, 120])
def test_constant_velocity_analytic_and_fps_invariant(fps):
    f = feature(line(fps=fps, vx=12, vy=16))
    assert f["v_mean"] == pytest.approx(1., abs=1e-9)
    assert f["a_mean"] < 1e-8
    assert f["turn_rate_mean"] < 1e-8
    assert f["straightness"] == pytest.approx(1.)


def test_gap_interpolation_uses_elapsed_time():
    full = line()
    partial = [p for i, p in enumerate(full) if i % 8 not in (2, 3)]
    f = feature(partial)
    assert f["v_mean"] == pytest.approx(1., abs=1e-9)
    assert f["a_mean"] < 1e-8
    result = extract_features(partial, 1920, 1080)
    assert .2 < result["quality"]["missing_fraction"] < .3


def test_explicit_frame_clock_and_timestamp_priority():
    history = line()
    assert feature(history, fps=100)["v_mean"] == pytest.approx(1.)
    for p in history:
        del p["timestamp_ms"]
    assert feature(history, fps=30)["v_mean"] == pytest.approx(1.)
    assert extract_features(history, 1920, 1080)["feature_status"] == "rejected"


def test_long_gap_is_not_bridged():
    history = [p for p in line() if not 40 <= p["frame_index"] < 65]
    result = extract_features(history, 1920, 1080)
    assert result["quality"]["usable_segments"] == 2
    assert result["quality"]["long_gap_count"] == 1
    assert result["quality"]["retained_duration_seconds"] < 3.3
    assert result["features"]["v_mean"] == pytest.approx(1.)


def test_stationary_and_scale_only_dont_invent_acceleration():
    history = line(vx=0)
    for i, p in enumerate(history):
        p["w"] *= 1+i/len(history)
    f = feature(history)
    assert f["v_mean"] < 1e-8
    assert f["a_mean"] < 1e-8
    assert f["turn_rate_mean"] == 0
    assert f["stationary_ratio"] == 1


def test_vector_acceleration_captures_constant_speed_turn():
    history = line(fps=60)
    for p in history:
        t = p["timestamp_ms"]/1000
        p["cx"] = (500+100*np.cos(.5*t))/1920
        p["cy"] = (500+100*np.sin(.5*t))/1080
    f = feature(history)
    assert f["v_mean"] == pytest.approx(2.5, rel=.01)
    assert f["a_mean"] == pytest.approx(1.25, rel=.03)
    assert f["turn_rate_mean"] == pytest.approx(.5, rel=.03)


def test_bbox_change_does_not_differentiate_normalized_speed():
    history = line()
    for i, p in enumerate(history):
        p["w"] *= 1+i/len(history)
    assert feature(history)["a_mean"] < 1e-8


def test_heading_order_is_retained():
    def zigzag(switch_every):
        points = line(duration=8)
        x, y = 300., 300.
        for i, p in enumerate(points):
            x += 2
            y += 1 if (i//switch_every) % 2 else -1
            p.update(cx=x/1920, cy=y/1080)
        return feature(points, config=FeatureConfig(smoothing_seconds=0))
    assert zigzag(3)["turn_rate_mean"] > 20*zigzag(120)["turn_rate_mean"]


def test_pixel_aspect_ratio_and_resolution_invariance():
    a = feature(line(vx=0, vy=20))
    history = line(vx=0, vy=20)
    b = extract_features(history, 3840, 2160)["features"]
    assert a["v_mean"] == pytest.approx(1.)
    assert b == pytest.approx(a)


def test_downsampling_filters_high_frequency_jitter_before_aliasing():
    history = line(fps=120, duration=8, vx=0)
    for p in history:
        p["cx"] += 5*np.sin(2*np.pi*30.5*p["timestamp_ms"]/1000)/1920
    f = feature(history)
    assert f["v_mean"] < .08


def test_smoothing_reduces_stationary_jitter_not_claimed_as_ground_truth():
    points = line(vx=0)
    rng = np.random.default_rng(10)
    for p in points:
        p["cx"] += rng.normal(0, .35)/1920
        p["cy"] += rng.normal(0, .35)/1080
    raw = feature(points, config=FeatureConfig(smoothing_seconds=0))
    smoothed = feature(points)
    assert smoothed["a_mean"] < .35*raw["a_mean"]
    assert smoothed["v_mean"] < .5*raw["v_mean"]
    assert smoothed["heading_change_ratio"] < .05
    assert extract_features(points, 1920, 1080)["quality"]["heading_valid_fraction"] < .1


@pytest.mark.parametrize("mutation,reason", [
    (lambda x: x[4].update(frame_index=3), "frame_indices"),
    (lambda x: x[4].update(frame_index=3.5), "frame_indices"),
    (lambda x: x[4].update(cx=float("nan")), "nonfinite"),
    (lambda x: x[4].update(w=0), "invalid_normalized"),
    (lambda x: x[4].update(conf=1.5), "invalid_normalized"),
    (lambda x: x[4].update(cx=-.1), "invalid_normalized"),
    (lambda x: x[4].update(timestamp_ms=0), "strictly_increasing"),
    (lambda x: x[4].pop("timestamp_ms"), "partial_timestamps"),
])
def test_invalid_inputs_reject_explicitly(mutation, reason):
    points = line()
    mutation(points)
    result = extract_features(points, 1920, 1080)
    assert result["features"] is None
    assert result["feature_status"] == "rejected"
    assert reason in result["reasons"][0]


def test_sparse_short_huge_tracks_reject_without_allocation():
    assert extract_features(line()[:3], 1920, 1080)["reasons"] == ["insufficient_points"]
    assert extract_features(line()[::4], 1920, 1080)["reasons"] == ["excessive_missing_fraction"]
    huge = line()
    for p in huge:
        p["timestamp_ms"] *= 1e7
    assert extract_features(huge, 1920, 1080)["reasons"] == ["resampling_limit_exceeded"]


def test_input_not_mutated_and_config_fingerprinted():
    points = line()
    saved = copy.deepcopy(points)
    feature(points)
    assert points == saved
    assert FeatureConfig().fingerprint != FeatureConfig(smoothing_seconds=.5).fingerprint


@pytest.mark.parametrize("kwargs", [{"target_fps":0}, {"smoothing_seconds":-1}, {"max_missing_fraction":2}, {"stationary_speed":-1}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        FeatureConfig(**kwargs)
