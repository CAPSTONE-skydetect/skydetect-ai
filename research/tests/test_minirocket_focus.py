import numpy as np

from research.minirocket_focus import aggregation_scores, rotate_channels
from research.minirocket_retrain import augment_training, run


def _data():
    q = np.array([[[0, 1, 3], [0, 2, 4]], [[0, 3, 2], [0, -1, 2]]], dtype=np.float32)
    x = np.concatenate([q, np.diff(q, axis=2, prepend=q[:, :, :1])], axis=1)
    return {"X": x, "y": np.array(["bird", "drone"]), "group_id": np.array(["g1", "g2"]),
            "sample_id": np.array(["t1:w0", "t2:w0"]), "sample_weight": np.array([.25, .75])}


def test_rotation_preserves_geometry_and_displacement_contract():
    original = _data()["X"]
    for turns in range(4):
        rotated = rotate_channels(original, turns)
        np.testing.assert_allclose(rotated[:, 2:], np.diff(rotated[:, :2], axis=2, prepend=rotated[:, :2, :1]))
        np.testing.assert_allclose(np.linalg.norm(rotated[:, :2], axis=1), np.linalg.norm(original[:, :2], axis=1))
        np.testing.assert_array_equal(rotate_channels(rotated, 4-turns), original)


def test_train_augmentation_keeps_groups_and_parent_weight_mass():
    data = _data()
    before = {key: value.copy() for key, value in data.items()}
    augmented = augment_training(data, True)
    assert len(set(augmented["sample_id"])) == 8
    for group in data["group_id"]:
        np.testing.assert_allclose(augmented["sample_weight"][augmented["group_id"] == group].sum(),
                                   data["sample_weight"][data["group_id"] == group].sum())
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])


def test_negative_windows_are_not_repaired_by_median_or_majority():
    values = aggregation_scores(np.array([-.3, -.8, -.1]))
    assert all(value < 0 for value in values.values())


def test_missing_package_blocks_before_loading_or_fitting(tmp_path):
    result = run(tmp_path, tmp_path / "missing", tmp_path / "output")
    assert result["status"] == "blocked"
    assert result["fitting_performed"] is False
