from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from research.build_real_sequence_dataset import file_hash
from research.package_real_reference_dataset import package
from research.trajectory_sequence import CONTRACT_VERSION, SequenceConfig


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def save_windows(path, labels, groups, ids):
    np.savez_compressed(path, X=np.zeros((len(ids), 4, 60), np.float32),
                        y=np.asarray(labels), group_id=np.asarray(groups), sample_id=np.asarray(ids))


@pytest.fixture
def inputs(tmp_path):
    handoff, augmentation, evaluation = (tmp_path / name for name in ("handoff", "augmentation", "evaluation"))
    real = handoff / "real"
    for path in (real, augmentation, evaluation):
        path.mkdir(parents=True)
    save_windows(real / "train.npz", ["bird", "drone"], ["g1", "g2"], ["t1:w0", "t2:w0"])
    save_windows(real / "validation.npz", ["bird", "drone"], ["g3", "g4"], ["t3:w0", "t4:w0"])
    save_windows(handoff / "synthetic_train.npz", ["bird", "drone"],
                 ["sim:bird:101", "sim:drone:102"], ["sim:bird:101:w0", "sim:drone:102:w0"])
    save_windows(augmentation / "augmented_train.npz", ["bird", "drone"],
                 ["g1", "g2"], ["t1:w0:anchor1", "t2:w0:anchor1"])
    pd.DataFrame([
        dict(npz_row=i % 2, split="train" if i < 2 else "validation",
             sample_id=f"t{i + 1}:w0", parent_track_id=f"t{i + 1}",
             source_group_id=f"g{i + 1}", label="bird" if i % 2 == 0 else "drone",
             file_name="private.mp4", audit_sample_id="private") for i in range(4)
    ]).to_csv(real / "metadata.csv", index=False)
    pd.DataFrame([
        dict(sample_id=f"sim:{label}:{seed}:w0", group_id=f"sim:{label}:{seed}",
             label=label, seed=seed, subtype="example")
        for label, seed in (("bird", 101), ("drone", 102))
    ]).to_csv(handoff / "synthetic_train_metrics.csv", index=False)
    pd.DataFrame([
        dict(sample_id=f"t{i}:w0:anchor1", group_id=f"g{i}", y=label,
             parent_track_id=f"t{i}", donor_id=f"sim:{label}:101:w0")
        for i, label in ((1, "bird"), (2, "drone"))
    ]).to_csv(augmentation / "provenance.csv", index=False)
    write_json(handoff / "group_audit.json", dict(cross_split_groups={}, changed_groups=[],
               source_video_groups_verified=True, train_without_video_hash=0))
    write_json(handoff / "preparation_results.json", dict(status="experimental_downstream_ablation_only"))
    write_json(handoff / "selected_profile.json", dict(name="candidate"))
    write_json(evaluation / "results.json", dict(status="development_candidate_only",
               large_generation_ready=True))
    np.savez_compressed(evaluation / "donor_pool.npz", X=np.zeros((1, 4, 60), np.float32))
    config = SequenceConfig()
    sources = [dict(parent_track_id=f"t{i + 1}", source_group_id=f"g{i + 1}",
                    split="train" if i < 2 else "validation" if i < 4 else "test",
                    label="bird" if i % 2 == 0 else "drone", video_sha256="") for i in range(6)]
    real_manifest = dict(contract_version=CONTRACT_VERSION, contract_id=config.fingerprint,
                         channels=["q_x", "q_y", "d_x", "d_y"], config=asdict(config), sources=sources,
                         artifact_hashes={name: file_hash(real / name) for name in
                                          ("train.npz", "validation.npz", "metadata.csv")})
    write_json(real / "dataset_manifest.json", real_manifest)
    aug_manifest = dict(contract_version=CONTRACT_VERSION, contract_id=config.fingerprint,
                        source_manifest_sha256=file_hash(real / "dataset_manifest.json"),
                        evaluation_results_sha256=file_hash(evaluation / "results.json"),
                        donor_pool_sha256=file_hash(evaluation / "donor_pool.npz"),
                        status="development_training_only_not_independent_cases",
                        recommended_classifier_mass=.1,
                        hashes={name: file_hash(augmentation / name) for name in
                                ("augmented_train.npz", "provenance.csv")})
    write_json(augmentation / "dataset_manifest.json", aug_manifest)
    return handoff, augmentation, evaluation, tmp_path / "package"


def test_package_three_arms_one_real_validation_without_test(inputs):
    handoff, augmentation, evaluation, output = inputs
    manifest = package(handoff, augmentation, evaluation, output)
    assert manifest["status"] == "development_comparison_only"
    assert manifest["test_included"] is False
    assert manifest["real_split_source_groups"] == {"train": 2, "validation": 2, "test": 2}
    assert not (output / "test.npz").exists()
    with np.load(output / "train_real_plus_augmented.npz", allow_pickle=False) as data:
        assert data["X"].shape == (4, 4, 60)
        assert np.isclose(data["sample_weight"][2:].sum() / data["sample_weight"].sum(), .1)
        assert set(data["domain"]) == {"real", "real_anchor_augmented"}
    with np.load(output / "train_synthetic_only.npz", allow_pickle=False) as data:
        assert set(data["domain"]) == {"synthetic"}
    assert "private" not in (output / "real_metadata.csv").read_text(encoding="utf-8")
    assert manifest["file_hashes"]["validation_real.npz"] == file_hash(output / "validation_real.npz")


def test_reject_changed_augmentation_source(inputs):
    handoff, augmentation, evaluation, output = inputs
    with (augmentation / "provenance.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Augmented source changed"):
        package(handoff, augmentation, evaluation, output)
    assert not output.exists()


def test_reject_cross_split_group(inputs):
    handoff, augmentation, evaluation, output = inputs
    manifest_path = handoff / "real" / "dataset_manifest.json"
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    source["sources"][2]["source_group_id"] = "g1"
    write_json(manifest_path, source)
    aug_path = augmentation / "dataset_manifest.json"
    aug = json.loads(aug_path.read_text(encoding="utf-8"))
    aug["source_manifest_sha256"] = file_hash(manifest_path)
    write_json(aug_path, aug)
    with pytest.raises(ValueError, match="Source video crosses real splits"):
        package(handoff, augmentation, evaluation, output)
    assert not output.exists()


def test_reject_augmented_label_that_disagrees_with_parent(inputs):
    handoff, augmentation, evaluation, output = inputs
    path = augmentation / "augmented_train.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["y"] = arrays["y"][::-1]
    np.savez_compressed(path, **arrays)
    provenance = pd.read_csv(augmentation / "provenance.csv")
    provenance["y"] = arrays["y"]
    provenance.to_csv(augmentation / "provenance.csv", index=False)
    manifest_path = augmentation / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hashes"] = {name: file_hash(augmentation / name) for name in
                          ("augmented_train.npz", "provenance.csv")}
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="differs from its real parent"):
        package(handoff, augmentation, evaluation, output)
    assert not output.exists()
