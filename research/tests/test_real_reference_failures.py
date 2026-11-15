import numpy as np
import pandas as pd
import pytest

from research.analyze_real_reference_failures import checked_predictions, load_aligned_metadata
from research.evaluate_sequence_handoff import aggregate_scores


def example_arrays():
    return dict(X=np.zeros((3, 4, 60), np.float32),
                y=np.array(["bird", "bird", "drone"]),
                group_id=np.array(["g1", "g1", "g2"]),
                sample_id=np.array(["t1:w0000", "t1:w0001", "t2:w0000"]))


def write_predictions(folder, arrays):
    decisions = np.array([-.5, .2, .7])
    for level, table in zip(("window", "track", "group"), aggregate_scores(arrays, decisions)):
        table = table.copy()
        table["prediction"] = np.where(table.decision >= 0, "drone", "bird")
        table.to_csv(folder / f"real_only_validation_{level}.csv", index=False)


def test_checked_predictions_rebuilds_window_to_group_aggregation(tmp_path):
    arrays = example_arrays()
    write_predictions(tmp_path, arrays)
    tables = checked_predictions(tmp_path, "real_only", arrays)
    assert len(tables["window"]) == 3
    assert tables["group"].loc[tables["group"].group_id == "g1", "decision"].item() == pytest.approx(-.15)


def test_checked_predictions_rejects_changed_group_margin(tmp_path):
    arrays = example_arrays()
    write_predictions(tmp_path, arrays)
    path = tmp_path / "real_only_validation_group.csv"
    changed = pd.read_csv(path)
    changed.loc[0, "decision"] = -.9
    changed.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Aggregation changed"):
        checked_predictions(tmp_path, "real_only", arrays)


def test_metadata_alignment_rejects_reordered_samples(tmp_path):
    arrays = example_arrays()
    metadata = pd.DataFrame(dict(split=["validation"] * 3, npz_row=[0, 1, 2],
                                 sample_id=["t1:w0001", "t1:w0000", "t2:w0000"],
                                 source_group_id=arrays["group_id"], label=arrays["y"]))
    path = tmp_path / "metadata.csv"
    metadata.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not align"):
        load_aligned_metadata(path, arrays, "validation")
