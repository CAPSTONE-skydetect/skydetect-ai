import pandas as pd
import pytest

from research.build_dataset import assign_train_test, build_dataset, select_balanced


def candidates():
    rows = []
    for scenario in ("a", "b"):
        for subtype, label in (("x", "bird"), ("y", "drone")):
            for profile in ("ideal", "noisy"):
                for i in range(20):
                    rows.append(dict(sample_id=f"{scenario}:{subtype}:{profile}:{i}", family_id=f"{scenario}:{i}",
                                     scenario=scenario, subtype=subtype, label=label,
                                     observation_profile=profile, split="train", feature_status="accepted"))
    return pd.DataFrame(rows)


def test_exact_selection_balances_class_and_strata():
    result = select_balanced(candidates(), 120, 3, "train")
    assert len(result) == 120
    assert result.label.value_counts().to_dict() == {"bird": 60, "drone": 60}
    assert result.groupby(["label", "scenario", "observation_profile"]).size().nunique() == 1
    assert result.sample_id.nunique() == 120


def test_family_split_is_scenario_stratified_and_deterministic():
    families = {s: [f"{s}:{i}" for i in range(10)] for s in ("a", "b")}
    first = assign_train_test(families, 4)
    second = assign_train_test({s: list(reversed(v)) for s, v in families.items()}, 4)
    assert first == second
    for scenario in families:
        values = [first[f] for f in families[scenario]]
        assert values.count("test") == 2
        assert values.count("train") == 8


@pytest.mark.parametrize("train_count,test_count", [(11, 4), (10, 3), (0, 4)])
def test_invalid_target_counts_fail(train_count, test_count, tmp_path):
    with pytest.raises(ValueError):
        build_dataset(tmp_path, train_count, test_count)


def test_small_end_to_end_exact_dataset(tmp_path):
    train, test, ledger, manifest = build_dataset(
        tmp_path, train_count=32, test_count=8, seed=5, families_per_scenario=4,
        preview_per_combination=0)
    assert len(train) == 32 and len(test) == 8
    assert not (set(train.family_id) & set(test.family_id))
    assert manifest["family_leakage_count"] == 0
    assert len(ledger) == 4*4*7*2
    assert (tmp_path/"candidate_ledger_v3.csv").exists()
    assert (tmp_path/"dataset_manifest.json").exists()


def test_selected_csvs_retain_declared_splits(tmp_path):
    build_dataset(tmp_path, train_count=32, test_count=8, seed=7,
                  families_per_scenario=4, preview_per_combination=0)
    assert set(pd.read_csv(tmp_path/"train_features_v3.csv").split) == {"train"}
    assert set(pd.read_csv(tmp_path/"test_features_v3.csv").split) == {"test"}
