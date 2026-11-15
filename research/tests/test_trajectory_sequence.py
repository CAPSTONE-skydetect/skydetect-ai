import copy
import json

import numpy as np
import pandas as pd
import pytest

from research.build_real_sequence_dataset import build, group_records, assign_splits
from research.trajectory_sequence import SequenceConfig, normalize_window, window_track


def track(fps=30, duration=4, phase=0):
    return dict(processed_width=1280, processed_height=720,
                stabilization=dict(applied=True), source_video_id=f"source-{phase}", track_id=1,
                history=[dict(frame_index=i, timestamp_ms=1000*i/fps,
                              cx=0.2+0.02*i/fps,
                              cy=0.5+0.02*np.sin(2*np.pi*2*i/fps+phase))
                         for i in range(int(fps*duration)+1)])


def test_two_second_contract_and_no_input_mutation():
    t = track()
    original = copy.deepcopy(t)
    windows, rejected = window_track(t)
    assert t == original and not rejected
    assert len(windows) == 3
    assert [w["relative_start_s"] for w in windows] == [0, 1, 2]
    for row in windows:
        x = row["X"]
        assert x.shape == (4, 60) and x.dtype == np.float32
        np.testing.assert_allclose(x[2:, 1:], np.diff(x[:2], axis=1), atol=2e-7)
        assert np.array_equal(x[2:, 0], [0, 0])
        assert np.isfinite(x).all()


def test_spatial_translation_and_uniform_scale_preserve_shape():
    cfg = SequenceConfig()
    p = np.column_stack((np.linspace(0, 1, 60), np.sin(np.linspace(0, 5, 60))))
    a, _ = normalize_window(p, cfg)
    b, _ = normalize_window(p*3 + [7, -9], cfg)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_aspect_ratio_preserved_not_independently_scaled():
    a = track()
    b = copy.deepcopy(a)
    b["processed_height"] = 960
    for point in b["history"]:
        point["cy"] *= 720/960
    wa, _ = window_track(a)
    wb, _ = window_track(b)
    np.testing.assert_allclose(wa[0]["X"], wb[0]["X"], atol=1e-6)


def test_stationary_does_not_amplify_noise_or_reject_hover():
    t = track()
    for point in t["history"]:
        point.update(cx=.5, cy=.5)
    windows, _ = window_track(t)
    assert windows[0]["scale_floor_applied"]
    assert np.count_nonzero(windows[0]["X"]) == 0


def test_short_track_not_stretched_or_padded():
    assert window_track(track(duration=1.99))[1][0]["reason"] == "short_track"
    assert len(window_track(track(duration=2))[0]) == 1


def test_long_gap_rejects_crossing_windows_only():
    t = track(duration=6)
    t["history"] = [p for p in t["history"] if not 65 <= p["frame_index"] < 90]
    windows, rejects = window_track(t)
    assert {w["relative_start_s"] for w in windows} == {0, 3, 4}
    assert len(rejects) == 2 and all(r["reason"] == "long_gap" for r in rejects)


def test_short_gap_budget():
    t = track(duration=2)
    t["history"] = [p for p in t["history"] if p["frame_index"] not in (20, 21)]
    windows, rejects = window_track(t)
    assert not rejects and windows[0]["missing_fraction"] == pytest.approx(1/30)
    assert "short_gap_interpolated" in windows[0]["quality_flags"]
    t = track(duration=2)
    t["history"] = [p for p in t["history"] if p["frame_index"] % 5 != 3]
    assert window_track(t)[1][0]["reason"] == "missing_fraction"


@pytest.mark.parametrize("kind", ["duplicate_time", "backward_frame", "nan", "dimensions", "raw"])
def test_bad_input_fails_explicitly(kind):
    t = track()
    if kind == "duplicate_time":
        t["history"][1]["timestamp_ms"] = 0
    elif kind == "backward_frame":
        t["history"][1]["frame_index"] = 0
    elif kind == "nan":
        t["history"][1]["cx"] = float("nan")
    elif kind == "dimensions":
        t["processed_width"] = 0
    else:
        t["stabilization"]["applied"] = False
    with pytest.raises(ValueError):
        window_track(t)


def test_high_fps_anti_alias_suppresses_out_of_band_signal():
    low, high = track(fps=60, duration=6), track(fps=60, duration=6)
    for t, frequency in ((low, 3), (high, 23)):
        for p in t["history"]:
            p["cy"] = .5+.02*np.sin(2*np.pi*frequency*p["timestamp_ms"]/1000)
    lw, _ = window_track(low)
    hw, _ = window_track(high)
    assert hw[1]["anti_alias_applied"]
    assert np.std(hw[1]["X"][1]) < .15*np.std(lw[1]["X"][1])


@pytest.mark.parametrize("fps", [30, 29.97, 60, 59.94])
def test_integer_millisecond_clock_does_not_bias_native_fps_or_filter_choice(fps):
    t = track(fps=fps, duration=5)
    for p in t["history"]:
        p["timestamp_ms"] = int(round(p["timestamp_ms"]))
    t["history"] = [p for p in t["history"] if p["frame_index"] not in (20, 21)]
    windows, _ = window_track(t)
    assert windows and windows[0]["source_fps"] == pytest.approx(fps, rel=2e-4)
    assert windows[0]["anti_alias_applied"] == (fps > 30.3)


def test_upsampling_preserves_seconds_not_signal_claim():
    windows, _ = window_track(track(fps=15))
    assert len(windows) == 3 and windows[0]["X"].shape == (4, 60)
    assert "upsampled_no_new_information" in windows[0]["quality_flags"]


def test_group_union_uses_source_video_and_session_transitively():
    records = [dict(parent_track_id=str(i), label="bird", source_video_id=f"s{i}",
                    center_hash=f"h{i}", video_sha256="", file_name=f"bird{i}.json")
               for i in range(6)]
    records[1]["source_video_id"] = records[0]["source_video_id"]
    records[2]["video_sha256"] = records[1]["video_sha256"] = "same-video"
    links = group_records(records, {"2": "same-session", "3": "same-session"})
    assign_splits(records, 42)
    assert len({r["source_group_id"] for r in records[:4]}) == 1
    assert len({r["split"] for r in records[:4]}) == 1
    assert {r["evidence"] for r in links} >= {"source_id", "video_bytes", "reviewed_session"}


def test_end_to_end_reproducible_group_safe_and_pickle_free(tmp_path):
    root = tmp_path/"tracks"
    for label, offset in (("bird", 0), ("drone", 10)):
        (root/label).mkdir(parents=True)
        for i in range(6):
            t = track(duration=12, phase=(i+offset)*.17)
            (root/label/f"{i}.json").write_text(json.dumps(t))
    duplicate = root/"bird"/"copy.json"
    duplicate.write_bytes((root/"bird"/"0.json").read_bytes())
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    a, b = tmp_path/"a", tmp_path/"b"
    result = build(root, a)
    build(root, b)
    assert result["split_group_overlap"] == 0
    assert any(r["status"] == "duplicate" for r in result["sources"])
    groups = []
    for split in ("train", "validation", "test"):
        x = np.load(a/f"{split}.npz", allow_pickle=False)
        y = np.load(b/f"{split}.npz", allow_pickle=False)
        for name in x.files:
            np.testing.assert_array_equal(x[name], y[name])
        assert x["X"].shape[1:] == (4, 60)
        assert set(x["y"]) == {"bird", "drone"}
        groups.append(set(x["group_id"]))
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    meta = pd.read_csv(a/"metadata.csv")
    assert meta[meta.split == "train"].groupby("source_group_id").size().max() <= 8
    for split, rows in meta.groupby("split"):
        x = np.load(a/f"{split}.npz", allow_pickle=False)
        assert rows.sample_id.tolist() == x["sample_id"].tolist()
        assert rows.npz_row.tolist() == list(range(len(rows)))
    assert all(path.read_bytes() == contents for path, contents in before.items())
    with pytest.raises(ValueError, match="empty"):
        build(root, a)
