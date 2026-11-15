from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from research.camera import Camera
from research.diagnose_sequence_gap import common_windows, optical_at_observed_frames, source_evidence
from research.observation import NoiseConfig, ObservationModel
from research.refine_motion_residual import cv_gate, residual_gain


def test_optical_pair_preserves_gaps_without_mutating_observations():
    record = dict(track=dict(history=[dict(frame_index=i, cx=.51, cy=.49) for i in (0, 1, 4)]),
                  optical_truth=[dict(frame_index=i, cx=.5, cy=.5) for i in range(5)])
    before = deepcopy(record)
    pair = optical_at_observed_frames(record)
    assert record == before
    assert [p["frame_index"] for p in pair["track"]["history"]] == [0, 1, 4]
    assert all(p["cx"] == .5 for p in pair["track"]["history"])


def test_common_windows_aligns_by_identity_and_excludes_unpaired_rows():
    left = (np.array([[1], [2], [3]]), pd.DataFrame(dict(sample_id=["a", "b", "c"])))
    right = (np.array([[30], [10]]), pd.DataFrame(dict(sample_id=["c", "a"])))
    paired = common_windows(dict(left=left, right=right))
    assert paired["left"][0].ravel().tolist() == [1, 3]
    assert paired["right"][0].ravel().tolist() == [10, 30]


def test_source_audit_detects_cross_split_upload_without_reading_test_file():
    records = [dict(parent_track_id=str(i), label="bird", file_name=f"bird{i}.json",
                    source_video_id="same-upload", center_hash=str(i), video_sha256="",
                    split=split, status="accepted", source_path="does-not-exist.json")
               for i, split in enumerate(("train", "test"))]
    result = source_evidence(dict(sources=records))
    assert len(result["cross_split_evidence_links"]) == 1
    assert result["validation_test_coordinates_opened"] is False
    assert "source_group_id" not in records[0]


def test_motion_residual_scales_with_pixel_motion_without_label():
    cfg = NoiseConfig(jitter_px=.02, jitter_motion_fraction=.1, jitter_difficulty_gain=0,
                      dropout_rate=0, burst_probability=0, drift_probability=0, camera_probability=0)
    residuals = []
    for step in (0., 2.5):
        truth = [dict(cx=.15+step*i/1920, cy=.5, w=.02, h=.02, conf=1.,
                      frame_index=i, timestamp_ms=round(i*1000/30)) for i in range(400)]
        history, _ = ObservationModel(Camera(), np.random.default_rng(12), config=cfg).apply(truth, 30)
        residuals.append(np.array([(p["cx"]-truth[p["frame_index"]]["cx"])*1920 for p in history])[30:])
    assert residuals[1].std() > 8*residuals[0].std()
    assert residuals[0].std() == pytest.approx(.02, rel=.25)
    with pytest.raises(ValueError, match="Motion residual"):
        NoiseConfig(jitter_motion_fraction=-.1)


def test_cv_gate_rejects_mean_improvement_when_one_fold_regresses():
    rows = [dict(fold=i, variant=variant, distance=d, objective=d+.1, usable=.95)
            for variant, distances in (("control", [1., 1., 1.]), ("motion_residual", [.5, .5, 1.1]))
            for i, d in enumerate(distances)]
    gate = cv_gate(pd.DataFrame(rows))
    assert gate["checks"]["mean_distance_improved"]
    assert not gate["passed"]
    rows[-1]["distance"], rows[-1]["objective"] = .9, 1.
    assert cv_gate(pd.DataFrame(rows))["passed"]
    rows[-1]["distance"] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        cv_gate(pd.DataFrame(rows))


def test_residual_filter_gain_agrees_with_stationary_ar1_simulation():
    from scipy.signal import savgol_filter
    rng = np.random.default_rng(18)
    x = np.zeros(20000)
    for i in range(1, len(x)):
        x[i] = .55*x[i-1]+np.sqrt(1-.55**2)*rng.normal()
    residual = x-savgol_filter(x, 9, 2)
    assert residual[100:-100].std() == pytest.approx(residual_gain(), rel=.03)
