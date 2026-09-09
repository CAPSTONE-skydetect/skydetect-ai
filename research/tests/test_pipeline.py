import json

import numpy as np
import pandas as pd
import pytest

from research.comparison import compare_domains
from research.evaluation import evaluate_dataset, score_predictions, validate_table
from research.features import FEATURE_COLUMNS
from research.io import read_jsonl, write_json
from research.pipeline import BatchRunner, CoreFeatureExtractor, assign_splits
from research.statistical_engine import describe_dataset
from research.verify_real_track import RealTrackVerifier, load_tracks


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    path = tmp_path_factory.mktemp("dataset")
    table = BatchRunner(path, seed=22).run(samples_per_subtype=5)
    return path, table


def test_dataset_roundtrip_split_disjoint_and_no_silent_retries(batch):
    path, table = batch
    assert len(table) == 4*5*7*2
    assert table.sample_id.nunique() == len(table)
    assert table.groupby("family_id").split.nunique().max() == 1
    assert set(table.split) == {"train", "validation", "test"}
    assert (table.groupby("scenario").split.nunique() == 3).all()
    assert set(table.subtype) == {"pigeon", "seagull", "falcon", "consumer_quad", "hover_quad", "racing_quad", "fixed_wing_drone"}
    parsed = pd.read_csv(path / "simulation_features_v3.csv")
    validate_table(parsed)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["calibration_status"] == "uncalibrated_no_real_A"
    samples = list(read_jsonl(path / "raw_trajectories_v3.jsonl"))
    assert len(samples) == len(table)
    for first, second in zip(samples[::2], samples[1::2]):
        assert first["world_truth"] == second["world_truth"]
        assert first["metadata"]["split"] == second["metadata"]["split"]
    reextracted = CoreFeatureExtractor().process(path / "raw_trajectories_v3.jsonl")
    for original, result in zip(samples, reextracted):
        assert result["features"] == original["feature_result"]["features"]


def test_random_labels_and_stratified_summary_are_diagnostic_only(batch):
    _, table = batch
    result = evaluate_dataset(table)
    assert result["status"] == "completed"
    assert set(result["ablations"]) == {"motion", "all", "bbox_only"}
    assert result["real_holdout"]["status"] == "not_measured_no_real_A"
    assert 0 <= result["ablations"]["all"]["balanced_accuracy"] <= 1
    summary = describe_dataset(table)
    assert summary["real_world_validity"] == "not_measured"
    assert summary["total"] == len(table)
    assert summary["observation_sensitivity"]["v_mean"]["pairs"] > 0
    json.dumps(summary, allow_nan=False)
    json.dumps(result, allow_nan=False)


def test_heldout_changes_dont_change_fitted_model(batch):
    _, table = batch
    before = evaluate_dataset(table)
    changed = table.copy()
    changed.loc[changed.split == "test", "v_mean"] *= 100
    after = evaluate_dataset(changed)
    assert before["ablations"]["all"]["feature_importance"] == after["ablations"]["all"]["feature_importance"]


@pytest.mark.parametrize("error", ["leak", "version", "config", "duplicate", "nan"])
def test_csv_contract_rejects_invalid_inputs(batch, error):
    _, original = batch
    table = original.copy()
    index = table.index[table.feature_status == "accepted"][0]
    if error == "leak":
        table.loc[index, "split"] = "calibration"
    elif error == "version":
        table.loc[index, "feature_version"] = "2.0.0"
    elif error == "config":
        table.loc[index, "feature_config_id"] = "other"
    elif error == "duplicate":
        table = pd.concat([table, table.iloc[:1]], ignore_index=True)
    else:
        table.loc[index, "v_mean"] = np.nan
    with pytest.raises(ValueError):
        validate_table(table)


def real_fixture(tmp_path):
    # This is an adapter contract test, NOT evidence from real footage.
    sample = BatchRunner(seed=7).simulate("baseline", "drone", "consumer_quad", 0, noisy=False, frame_count=120)
    track = sample["track"]
    track["stabilization"] = {"applied": True, "method": "opencv_feature_cmc"}
    write_json(tmp_path / "track.json", track)
    entry = dict(track_path="track.json", source_group_id="original-session-1", label="drone",
                 split="calibration", coordinate_space="post_cmc_residual_observation",
                 frame_width=1920, frame_height=1080, review_status="approved")
    write_json(tmp_path / "manifest.json", {"videos": [entry]})
    return sample, entry


def test_real_import_shares_feature_code_and_provenance(tmp_path):
    sample, _ = real_fixture(tmp_path)
    table, diagnostics = RealTrackVerifier().import_manifest(tmp_path / "manifest.json")
    assert diagnostics[0]["feature_result"] == sample["feature_result"]
    assert table.family_id.iloc[0] == "real:original-session-1"
    assert table.a_missing_ratio.iloc[0] == sample["track"]["quality"]["missing_ratio"]


@pytest.mark.parametrize("error", ["resolution", "unstabilized", "split", "duplicate"])
def test_real_import_rejects_ambiguous_contract(tmp_path, error):
    sample, entry = real_fixture(tmp_path)
    if error == "resolution":
        del entry["frame_width"]
    elif error == "unstabilized":
        sample["track"]["stabilization"]["applied"] = False
        write_json(tmp_path / "track.json", sample["track"])
    elif error == "split":
        entry["split"] = "train"
    entries = [entry, entry] if error == "duplicate" else [entry]
    write_json(tmp_path / "manifest.json", {"videos": entries})
    with pytest.raises((ValueError, KeyError)):
        RealTrackVerifier().import_manifest(tmp_path / "manifest.json")


def test_real_comparison_excludes_test_and_reports_missing_class(batch, tmp_path):
    _, synthetic = batch
    real_fixture(tmp_path)
    real, _ = RealTrackVerifier().import_manifest(tmp_path / "manifest.json")
    comparison = compare_domains(synthetic, real)
    assert comparison["classes"]["bird"]["status"] == "missing_class"
    assert comparison["classes"]["drone"]["real_groups"] == 1
    assert comparison["classes"]["drone"]["domain_classifier"]["status"] == "insufficient_groups_for_domain_cv"
    real["split"] = "test"
    assert compare_domains(synthetic, real)["status"] == "no_real_calibration_tracks_test_not_used"


def test_group_assignments_stable_to_input_order():
    families = [f"seed:scenario:{i}" for i in range(50)]
    assert assign_splits(families, 42) == assign_splits(list(reversed(families)), 42)


def test_grouped_comparison_detects_injected_shift_and_no_shift(batch):
    # Artificial copies exercise statistics, not sim-to-real evidence.
    _, synthetic = batch
    real = synthetic[(synthetic.split == "train") & (synthetic.feature_status == "accepted")].copy()
    real["split"] = "calibration"
    real["sample_id"] = "real:" + real.sample_id
    real["family_id"] = "real:" + real.family_id
    real["review_status"] = "approved"
    no_shift = compare_domains(synthetic, real)
    for label in ("bird", "drone"):
        assert no_shift["classes"][label]["metrics"]["v_mean"]["wasserstein"] == pytest.approx(0.)
        assert no_shift["classes"][label]["real_real_split_distance_q05_q50_q95"] is not None
    real[FEATURE_COLUMNS] += 1000
    shifted = compare_domains(synthetic, real)
    for label in ("bird", "drone"):
        result = shifted["classes"][label]
        assert result["metrics"]["v_mean"]["wasserstein"] == pytest.approx(1000.)
        assert min(result["domain_classifier"]["fold_balanced_accuracy"]) > .9


def test_real_holdout_requires_manual_review(batch):
    _, synthetic = batch
    real = synthetic[synthetic.split == "test"].copy()
    real["review_status"] = "unreviewed"
    result = evaluate_dataset(synthetic, real_table=real)
    assert result["real_holdout"]["status"] == "insufficient_labeled_real_test_classes"
    real["review_status"] = "approved"
    result = evaluate_dataset(synthetic, real_table=real)
    assert result["real_holdout"]["reviewed_test_rows"] > 0


def test_single_class_slice_does_not_claim_two_class_accuracy():
    result = score_predictions(["bird", "bird"], np.array(["bird", "bird"]), np.array([.1, .2]))
    assert result["balanced_accuracy"] is None
    assert result["macro_f1"] is None
    assert result["auroc"] is None
    assert result["recall"]["drone"] is None
    assert result["observed_accuracy"] == 1.
