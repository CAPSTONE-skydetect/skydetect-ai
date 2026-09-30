"""Guard the source-group and sequence contract of anchor augmentation."""
import numpy as np
import pandas as pd
import pytest

from research.build_anchor_training_dataset import quotas_for_groups, select_balanced
from research.evaluate_anchor_augmentation import augment_anchors, donor_residual
from research.trajectory_sequence import SequenceConfig, normalize_window


def _fixtures():
    t = np.linspace(0, 1, 60)
    anchor = np.column_stack((0.05 * t, 0.01 * np.sin(2 * np.pi * t)))
    x, info = normalize_window(anchor, SequenceConfig())
    real = dict(X=np.stack([x, x]), y=np.array(["bird", "drone"]),
                group_id=np.array(["video-b", "video-d"]),
                sample_id=np.array(["bird:w0000", "drone:w0000"]))
    metadata = pd.DataFrame([
        dict(sample_id=real["sample_id"][i], source_group_id=real["group_id"][i],
             parent_track_id=f"track-{i}", center_x=0.4, center_y=0.3,
             normalization_scale=info["normalization_scale"])
        for i in range(2)
    ])
    donor = np.column_stack((0.15 * t, 0.03 * np.sin(6 * np.pi * t)))
    donor_x, _ = normalize_window(donor, SequenceConfig())
    donors = dict(X=np.stack([donor_x, donor_x]), y=real["y"],
                  sample_id=np.array(["donor-b", "donor-d"]))
    return real, metadata, {"track-0": 0.6, "track-1": 0.6}, donors


def test_residual_removes_endpoint_trend():
    _, _, _, donors = _fixtures()
    residual = donor_residual(donors["X"][0])
    np.testing.assert_allclose(residual[0], residual[-1], atol=1e-6)
    assert np.max(np.linalg.norm(residual, axis=1)) > 0


def test_augmented_windows_keep_real_parent_label_group_and_channels():
    real, metadata, dimensions, donors = _fixtures()
    before = real["X"].copy()
    generated, provenance, rejected = augment_anchors(
        real, metadata, dimensions, donors, .04, copies=2)
    assert rejected == 0
    assert len(generated["X"]) == 4
    assert set(generated["group_id"]) == set(real["group_id"])
    assert set(generated["y"]) == set(real["y"])
    assert provenance.groupby("parent_track_id").donor_id.nunique().max() == 1
    assert np.isfinite(generated["X"]).all()
    np.testing.assert_allclose(generated["X"][:, 2:, 1:],
                               np.diff(generated["X"][:, :2], axis=2), atol=1e-6)
    np.testing.assert_array_equal(real["X"], before)
    repeated, _, _ = augment_anchors(real, metadata, dimensions, donors, .04, copies=2)
    np.testing.assert_array_equal(generated["X"], repeated["X"])


def test_misaligned_metadata_is_rejected():
    real, metadata, dimensions, donors = _fixtures()
    with pytest.raises(ValueError, match="misaligned"):
        augment_anchors(real, metadata.iloc[::-1].reset_index(drop=True),
                        dimensions, donors, .04)


def test_out_of_frame_candidate_is_rejected():
    real, metadata, dimensions, donors = _fixtures()
    metadata["center_x"] = 0.999
    metadata["normalization_scale"] = 1.0
    with pytest.raises(ValueError, match="All augmented windows rejected"):
        augment_anchors(real, metadata, dimensions, donors, .1, copies=1)


def test_bulk_selection_balances_classes_and_original_groups():
    real, metadata, dimensions, donors = _fixtures()
    generated, provenance, _ = augment_anchors(
        real, metadata, dimensions, donors, .04, copies=6)
    for index in range(len(generated["X"])):
        generated["X"][index, 0, 10] += index * .001
    quotas = quotas_for_groups(real["y"], real["group_id"], 8)
    selected, selected_metadata = select_balanced(generated, provenance, quotas, 8)
    assert quotas == {"video-b": 4, "video-d": 4}
    assert list(selected_metadata.sample_id) == list(selected["sample_id"])
    assert dict(zip(*np.unique(selected["y"], return_counts=True))) == {"bird": 4, "drone": 4}
    assert set(selected["group_id"]) == set(real["group_id"])
    assert len({row.tobytes() for row in selected["X"]}) == 8
    with pytest.raises(ValueError, match="even"):
        quotas_for_groups(real["y"], real["group_id"], 7)
