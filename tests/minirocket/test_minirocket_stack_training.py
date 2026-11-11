import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from examples.stack_synthetic import generate
from research import minirocket_stack_training as stack
from research.minirocket_stack_data import collection, validate_lineage


def test_builder_preserves_clock_and_reserves_test(tmp_path):
    path = generate(tmp_path / "portable name")
    data, metadata, audit, digest = collection(path)
    assert set(data) == {"train", "validation"}
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert data["train"]["X"].shape == (16, 4, 60)
    assert sum(a["status"] == "reserved_test" for a in audit) == 2
    assert np.all(stack.scales_for(data["train"], metadata) > 0)
    assert len(stack.cv_folds(data["train"])) == 3

    manifest = json.loads(path.read_bytes())
    record = manifest["records"][0]
    track = path.parent / record["track_path"]
    raw = json.loads(track.read_bytes())
    raw["history"][15]["timestamp_ms"] += 20
    payload = json.dumps(raw).encode()
    track.write_bytes(payload)
    with pytest.raises(ValueError, match="hash changed"):
        collection(path)
    record["track_sha256"] = hashlib.sha256(payload).hexdigest()
    path.write_text(json.dumps(manifest))
    assert collection(path)[2][0]["status"] == "rejected_contract"


@pytest.mark.parametrize("key", ["source_sha256", "source_group", "session_group", "parent_id"])
def test_lineage_leakage(tmp_path, key):
    path = generate(tmp_path / "fixture")
    train, val = [json.loads(path.read_bytes())["records"][i] for i in (0, 8)]
    train["session_group"] = "capture-A"
    val["segment_id"] = "different-segment"
    val[key] = train[key]
    with pytest.raises(ValueError, match="leak"):
        validate_lineage([train, val])


def test_scales_and_output_guards(tmp_path):
    part = dict(sample_id=np.array(["a"]), y=np.array(["bird"]), group_id=np.array(["g"]))
    meta = pd.DataFrame(dict(sample_id=["a"], label=["bird"], source_group_id=["g"], normalization_scale=[.02]))
    assert stack.scales_for(part, meta).tolist() == [.02]
    with pytest.raises(ValueError, match="Missing"):
        stack.scales_for(part, meta.iloc[:0])
    with pytest.raises(ValueError, match="Duplicate"):
        stack.scales_for(part, pd.concat([meta, meta]))
    with pytest.raises(ValueError, match="invalid"):
        stack.scale_windows(np.zeros((1, 4, 60), dtype=np.float32), [0])
    with pytest.raises(FileExistsError):
        stack.fit({}, None, "unused", tmp_path)


def test_ineligible_folds_fail_before_fitting(tmp_path):
    data, _, _, _ = collection(generate(tmp_path / "fixture"))
    train = data["train"]
    keep = np.isin(train["group_id"], ["fixture-0", "fixture-1", "fixture-2", "fixture-4"])
    with pytest.raises(ValueError, match="fold missing"):
        stack.cv_folds({k: v[keep] for k, v in train.items()})


def test_scope_train_only_inherited_transforms(tmp_path, monkeypatch):
    data, meta, _, digest = collection(generate(tmp_path / "fixture"))
    val_arrays = (data["validation"]["X"],
                  stack.scale_windows(data["validation"]["X"], stack.scales_for(data["validation"], meta)))
    calls = []
    def no_fit(x):
        calls.append(x)
        assert all(not any(np.array_equal(a, b) for a in x for b in val) for val in val_arrays)
        raise RuntimeError("test intercepted first train transform before fitting")
    monkeypatch.setattr(stack.seq, "fit_representation", no_fit)
    with pytest.raises(RuntimeError, match="intercepted"):
        stack.fit(data, meta, digest, tmp_path / "never-fitted")
    assert len(calls) == 1
