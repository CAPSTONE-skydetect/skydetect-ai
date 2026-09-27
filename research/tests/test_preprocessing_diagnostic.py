import copy
import json

import numpy as np
import pandas as pd
import pytest

from research.diagnose_real_preprocessing import (
    artifact_index, assign_provisional_groups, read_raw_match, run,
    spectral_diagnostic, track_identity,
)
from research.features import FeatureConfig, extract_features


def track(fps=30, duration=4., frequency=5.):
    return dict(track_id=1, source_video_id="source-a", processed_width=1280,
                processed_height=720, stabilization=dict(applied=True, method="opencv_feature_cmc"),
                history=[dict(frame_index=i, timestamp_ms=1000*i/fps,
                              cx=(300+20*i/fps)/1920,
                              cy=(400+3*np.sin(2*np.pi*frequency*i/fps))/1080,
                              w=.02, h=.02, conf=.9)
                         for i in range(int(duration*fps)+1)])


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_diagnostics_preserve_exact_features_and_do_not_mutate_input(fps):
    t = track(fps)
    original = copy.deepcopy(t)
    baseline = extract_features(t["history"], 1920, 1080)
    traces = []
    diagnosed = extract_features(t["history"], 1920, 1080, diagnostics=traces)
    assert diagnosed == baseline
    assert t == original
    assert len(traces) == 1
    assert len(traces[0]["speed_px_s"]) == len(traces[0]["time_seconds"])-1
    assert len(traces[0]["valid_turn"]) == len(traces[0]["time_seconds"])-2
    traces[0]["smoothed_xy_px"][:] = -1
    assert t == original


def test_long_gaps_remain_separate_and_rejections_have_no_trace():
    t = track()
    t["history"] = [p for p in t["history"] if not 35 <= p["frame_index"] < 60]
    traces = []
    result = extract_features(t["history"], 1920, 1080, diagnostics=traces)
    assert result["feature_status"] == "accepted"
    assert len(traces) == 2
    assert traces[0]["time_seconds"][-1] < traces[1]["time_seconds"][0]-.7
    traces = []
    result = extract_features(t["history"][:2], 1920, 1080, diagnostics=traces)
    assert result["feature_status"] == "rejected"
    assert traces == []


def test_five_hz_signal_has_expected_sg_power_attenuation():
    spectral = spectral_diagnostic(track(duration=8.), FeatureConfig())
    assert spectral["band_retention"] == pytest.approx(.29**2, abs=.002)


def test_spectrum_does_not_bridge_missing_frames_or_invent_high_frequency_support():
    t = track(duration=5.)
    t["history"] = [p for p in t["history"] if p["frame_index"] % 25 != 0]
    assert spectral_diagnostic(t, FeatureConfig()) is None
    assert spectral_diagnostic(track(fps=15), FeatureConfig()) is None


def test_raw_match_checks_exported_coordinates_and_frame_alignment(tmp_path):
    t = track()
    folder = tmp_path/"run-a"
    folder.mkdir()
    (folder/"track_sequence.json").write_text(json.dumps(t))
    rows = [dict(frame_index=p["frame_index"], timestamp_ms=p["timestamp_ms"],
                 raw_x=p["cx"]*1280-4, raw_y=p["cy"]*720,
                 compensated_x=p["cx"]*1280, compensated_y=p["cy"]*720)
            for p in t["history"]]
    pd.DataFrame(rows).to_csv(folder/"trajectory.csv", index=False)
    index, errors = artifact_index(tmp_path)
    assert not errors
    raw, paths, errors = read_raw_match(t, index[track_identity(t)])
    assert not errors and paths
    assert np.allclose(raw["compensated"][:, 0]-raw["raw"][:, 0], 6.)
    rows[0]["compensated_x"] += 1
    pd.DataFrame(rows).to_csv(folder/"trajectory.csv", index=False)
    raw, paths, errors = read_raw_match(t, index[track_identity(t)])
    assert raw is None and errors


def test_grouping_unions_reuploads_and_marks_filename_hints_provisional():
    records = [dict(sample_id=str(i), source_video_id=f"source-{i}", history_hash=str(i),
                    label="bird", file_name=name) for i, name in enumerate(
                        ["bird36(1)_tracksequence.json", "bird36(2)_tracksequence.json", "other.json", "last.json"])]
    records[1]["video_sha256"] = records[2]["video_sha256"] = "same-video-bytes"
    assign_provisional_groups(records)
    assert len({r["candidate_group_id"] for r in records[:3]}) == 1
    assert records[-1]["candidate_group_id"] != records[0]["candidate_group_id"]
    assert all(r["split"] == "unassigned_development" for r in records)


def test_report_smoke_without_raw_artifacts_or_model(tmp_path):
    inputs = tmp_path/"inputs"
    for label in ("bird", "drone"):
        (inputs/label).mkdir(parents=True)
        t = track(frequency=5. if label == "bird" else 1.)
        t["source_video_id"] = label
        (inputs/label/"one.json").write_text(json.dumps(t))
    output = tmp_path/"output"
    table = run(inputs, tmp_path/"absent", output)
    assert len(table) == 2
    assert not table.raw_match.any()
    assert len(list((output/"plots").glob("*.png"))) == 2
    assert (output/"report.md").is_file()
    manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    assert all(x["raw_match"] is False for x in manifest)
