import json

import numpy as np
import pytest

from research.build_real_sequence_dataset import file_hash
from research.evaluate_real_reference_comparison import fit_arm, load_package
from research.trajectory_sequence import CONTRACT_VERSION, SequenceConfig


def save(path, labels, groups, ids, domains=None, weights=None):
    arrays = dict(X=np.zeros((len(ids), 4, 60), np.float32), y=np.asarray(labels),
                  group_id=np.asarray(groups), sample_id=np.asarray(ids))
    if domains is not None:
        arrays.update(domain=np.asarray(domains), sample_weight=np.asarray(weights, np.float64))
    np.savez_compressed(path, **arrays)


@pytest.fixture
def package(tmp_path):
    files = dict(real_only="train_real_only.npz",
                 real_plus_augmentation="train_real_plus_augmented.npz",
                 synthetic_only="train_synthetic_only.npz")
    labels = ["bird", "drone"]
    save(tmp_path / files["real_only"], labels, ["r1", "r2"], ["r1:w0", "r2:w0"],
         ["real"] * 2, [1, 1])
    save(tmp_path / files["real_plus_augmentation"], labels * 2,
         ["r1", "r2"] * 2, ["r1:w0", "r2:w0", "aug1:w0", "aug2:w0"],
         ["real"] * 2 + ["real_anchor_augmented"] * 2, [.9, .9, .1, .1])
    save(tmp_path / files["synthetic_only"], labels, ["s1", "s2"],
         ["s1:w0", "s2:w0"], ["synthetic"] * 2, [1, 1])
    save(tmp_path / "validation_real.npz", labels, ["v1", "v2"], ["v1:w0", "v2:w0"])
    manifest = dict(schema="real-reference-comparison-1", contract_version=CONTRACT_VERSION,
                    contract_id=SequenceConfig().fingerprint, test_included=False,
                    source_group_overlap=0, arms=files, common_validation="validation_real.npz",
                    counts={name: dict(windows=2 if name != files["real_plus_augmentation"] else 4)
                            for name in (*files.values(), "validation_real.npz")},
                    augmentation_sample_weight_mass=.1)
    manifest["file_hashes"] = {p.name: file_hash(p) for p in tmp_path.glob("*.npz")}
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_load_package_accepts_three_arms_with_one_validation(package):
    manifest, train, validation = load_package(package)
    assert set(train) == set(manifest["arms"])
    assert set(validation["group_id"]) == {"v1", "v2"}


def test_load_package_rejects_changed_file(package):
    with (package / "validation_real.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Package file changed"):
        load_package(package)


def test_load_package_rejects_validation_group_leak(package):
    path = package / "validation_real.npz"
    save(path, ["bird", "drone"], ["r1", "v2"], ["v1:w0", "v2:w0"])
    manifest_path = package / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_hashes"][path.name] = file_hash(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Training/validation overlap"):
        load_package(package)


def test_fit_arm_uses_only_supplied_representation_cohort(monkeypatch):
    seen = []

    class FakeRocket:
        def fit_transform(self, x):
            seen.append(x.copy())
            return self.transform(x)

        def transform(self, x):
            return np.column_stack((x[:, 0, 0], x[:, 0, 0] ** 2 + 1))

    monkeypatch.setattr("research.evaluate_real_reference_comparison.transformer", FakeRocket)
    x = np.zeros((4, 4, 60), np.float32)
    x[:, 0, 0] = [0, 1, 2, 3]
    data = dict(X=x, y=np.array(["bird", "bird", "drone", "drone"]),
                sample_weight=np.ones(4))
    fit_on = dict(X=x[:2])
    model = fit_arm(data, fit_on, 1.)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], fit_on["X"])
    assert model["classes"] == ["bird", "drone"]
