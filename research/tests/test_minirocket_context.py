import numpy as np
import pandas as pd
import pytest

from research.minirocket_context import reconstruct, validate_context
from research.minirocket_orbit import orbit_features
from research.minirocket_focus import rotate_channels
from research.trajectory_sequence import normalize_window, SequenceConfig
from research.minirocket_framescale import scale_windows


def _package():
    t = np.arange(151) / 30
    points = np.column_stack([.2 + t * .005, .3 + .003*np.sin(t)])
    xs, meta = [], []
    for start in range(4):
        x, stats = normalize_window(points[start*30:start*30+60], SequenceConfig())
        xs.append(x)
        meta.append({"sample_id": f"parent:w{start}", "window_start_s": float(start),
                     "window_end_s": float(start+2), "label": "drone", "source_group_id": "g",
                     **stats})
    data = {"X": np.stack(xs), "sample_id": np.array([row["sample_id"] for row in meta]),
            "y": np.array(["drone"]*4), "group_id": np.array(["g"]*4)}
    return data, pd.DataFrame(meta), points


def test_long_windows_reconstruct_original_signal_and_preserve_source_group():
    data, metadata, points = _package()
    long, audit = reconstruct(data, metadata)
    assert long["X"].shape == (2, 4, 120)
    assert set(long["group_id"]) == {"g"}
    expected = normalize_window(points[:120], SequenceConfig(window_seconds=4))[0]
    np.testing.assert_allclose(long["X"][0], expected, atol=2e-6)
    validate_context(long["X"])
    assert audit["max_overlap_disagreement_width_normalized"] < 1e-6


def test_long_window_never_fills_missing_support_or_ignores_conflicting_overlap():
    data, metadata, _ = _package()
    broken = {key: value[[0, 3]] for key, value in data.items()}
    with pytest.raises(ValueError, match="fully supported"):
        reconstruct(broken, metadata)
    metadata.loc[1, "center_x"] += .01
    with pytest.raises(ValueError, match="disagree"):
        reconstruct(data, metadata)


def test_orbit_pool_is_quarter_turn_invariant():
    class Features:
        def transform(self, x):
            return np.column_stack([x[:, 0].max(axis=1), x[:, 1].mean(axis=1), (x[:, 2]**2).sum(axis=1)])
    x = _package()[0]["X"]
    base = orbit_features(Features(), x)
    for turns in range(4):
        np.testing.assert_allclose(orbit_features(Features(), rotate_channels(x, turns)), base, atol=1e-12)


def test_frame_scale_recovers_speed_information_erased_by_window_normalization():
    t = np.arange(60) / 30
    slow = np.column_stack([.2 + t * .005, .3 + t * .002])
    fast = slow[0] + (slow - slow[0]) * 10
    x1, m1 = normalize_window(slow, SequenceConfig())
    x2, m2 = normalize_window(fast, SequenceConfig())
    np.testing.assert_allclose(x1, x2, atol=1e-6)
    scaled = scale_windows(np.stack([x1, x2]), [m1["normalization_scale"], m2["normalization_scale"]])
    np.testing.assert_allclose(scaled[1], scaled[0] * 10, atol=2e-6)
    np.testing.assert_allclose(scaled[:, 2:, 1:], np.diff(scaled[:, :2], axis=2))


def test_stack_window_mean_equals_track_meta_prediction(monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from research import minirocket_stack as stack
    values = np.array([[-1., -2.], [-.5, -.4], [1., .3], [2., 2.]])
    scaler = StandardScaler().fit(values)
    meta = LogisticRegression().fit(scaler.transform(values), ["bird", "bird", "drone", "drone"])
    model = {"contract_version": stack.VERSION, "classes": ["bird", "drone"], "meta": meta, "meta_scaler": scaler}
    monkeypatch.setattr(stack, "member_margins", lambda *args: values)
    actual = stack.decision_margins(model, None, None).mean()
    expected = meta.decision_function(scaler.transform(values.mean(axis=0, keepdims=True)))[0]
    assert abs(actual - expected) < 1e-12
