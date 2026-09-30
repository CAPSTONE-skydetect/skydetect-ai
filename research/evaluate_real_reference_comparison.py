"""Compare three training cohorts on the same previously used real validation set."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from .build_real_sequence_dataset import file_hash
from .evaluate_sequence_handoff import (
    aggregate_scores, choose_alpha, load_arrays, metrics, predict, transformer,
)
from .io import write_json
from .trajectory_sequence import CONTRACT_VERSION, SequenceConfig


ARMS = ("real_only", "real_plus_augmentation", "synthetic_only")


def load_package(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "dataset_manifest.json").read_text(encoding="utf-8"))
    if (manifest["schema"] != "real-reference-comparison-1" or
            manifest["contract_version"] != CONTRACT_VERSION or
            manifest["contract_id"] != SequenceConfig().fingerprint or
            manifest["test_included"] or manifest["source_group_overlap"]):
        raise ValueError("Unexpected dataset contract or split")
    for name, checksum in manifest["file_hashes"].items():
        if file_hash(folder / name) != checksum:
            raise ValueError(f"Package file changed: {name}")
    train = {}
    for arm in ARMS:
        path = folder / manifest["arms"][arm]
        arrays = load_arrays(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays["sample_weight"] = archive["sample_weight"].copy()
            arrays["domain"] = archive["domain"].copy()
        weights = arrays["sample_weight"]
        if (weights.shape != (len(arrays["y"]),) or not np.isfinite(weights).all() or
                np.any(weights <= 0) or len(arrays["domain"]) != len(weights)):
            raise ValueError(f"Invalid training weights or domains: {arm}")
        if manifest["counts"][path.name]["windows"] != len(weights):
            raise ValueError(f"Training count changed: {arm}")
        train[arm] = arrays
    validation = load_arrays(folder / manifest["common_validation"])
    if manifest["counts"][manifest["common_validation"]]["windows"] != len(validation["y"]):
        raise ValueError("Validation count changed")
    real, augmented, synthetic = (train[arm] for arm in ARMS)
    n = len(real["y"])
    if (not np.array_equal(real["sample_id"], augmented["sample_id"][:n]) or
            not np.array_equal(real["X"], augmented["X"][:n]) or
            not np.array_equal(real["y"], augmented["y"][:n]) or
            not np.array_equal(real["group_id"], augmented["group_id"][:n]) or
            set(augmented["domain"][:n]) != {"real"} or
            set(augmented["domain"][n:]) != {"real_anchor_augmented"} or
            set(synthetic["domain"]) != {"synthetic"}):
        raise ValueError("Training cohort lineage changed")
    if (set(augmented["group_id"][n:]) - set(real["group_id"]) or
            set(synthetic["group_id"]) & set(real["group_id"])):
        raise ValueError("Augmented parent or synthetic group lineage changed")
    all_ids = [set(data["sample_id"]) for data in (real, validation, synthetic)]
    if (any(all_ids[i] & all_ids[j] for i in range(3) for j in range(i + 1, 3)) or
            set(augmented["sample_id"][n:]) & set.union(*all_ids) or
            any(set(data["group_id"]) & set(validation["group_id"]) for data in train.values())):
        raise ValueError("Training/validation overlap")
    aug_mass = float(augmented["sample_weight"][n:].sum() / augmented["sample_weight"].sum())
    if not np.isclose(aug_mass, manifest["augmentation_sample_weight_mass"]):
        raise ValueError("Augmentation weight mass changed")
    return manifest, train, validation


def fit_arm(data, fit_representation_on, alpha):
    rocket = transformer()
    scaler = StandardScaler(with_mean=False)
    scaler.fit(rocket.fit_transform(fit_representation_on["X"]))
    features = scaler.transform(rocket.transform(data["X"])).astype(np.float64)
    classifier = RidgeClassifier(alpha=alpha).fit(
        features, data["y"], sample_weight=data["sample_weight"])
    return dict(rocket=rocket, scaler=scaler, classifier=classifier,
                alpha=alpha, classes=classifier.classes_.tolist(),
                contract_version=CONTRACT_VERSION, contract_id=SequenceConfig().fingerprint)


def evaluate(folder, output):
    folder, output = Path(folder), Path(output)
    if output.exists():
        raise ValueError("Output exists; use a new run directory")
    manifest, train, validation = load_package(folder)
    real_alpha, real_cv, real_folds = choose_alpha(train["real_only"])
    synthetic_alpha, synthetic_cv, synthetic_folds = choose_alpha(train["synthetic_only"])
    output.mkdir(parents=True)
    real_cv.to_csv(output / "real_alpha_cv.csv", index=False)
    synthetic_cv.to_csv(output / "synthetic_alpha_cv.csv", index=False)
    write_json(output / "cv_groups.json", dict(real=real_folds, synthetic=synthetic_folds))
    protocol = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        package_manifest_sha256=file_hash(folder / "dataset_manifest.json"),
        contract_version=CONTRACT_VERSION, contract_id=SequenceConfig().fingerprint,
        arms={arm: manifest["arms"][arm] for arm in ARMS},
        common_validation=manifest["common_validation"],
        representation="MiniRocket(n_kernels=10000, random_state=20260928, n_jobs=2)",
        scaler="StandardScaler(with_mean=False); fit on representation cohort only",
        classifier="RidgeClassifier; float64 features; packaged sample_weight",
        alpha_selection="Grouped 3-fold train-only CV; real alpha shared by real arms; synthetic alpha chosen on synthetic train",
        representation_fit={"real_only": "real train", "real_plus_augmentation": "real train only",
                            "synthetic_only": "synthetic train only"},
        alpha={"real_only": real_alpha, "real_plus_augmentation": real_alpha,
               "synthetic_only": synthetic_alpha},
        aggregation="Mean window margin per track, then equal-object mean per source video",
        validation_status="Repeatedly used development set; not an independent holdout",
        selection_status="Exploratory comparison only; no final model or generalization claim",
        test_opened=False,
        versions={name: importlib.metadata.version(name) for name in
                  ("aeon", "numpy", "pandas", "scikit-learn", "scipy", "numba", "joblib")},
        evaluator_sha256=file_hash(__file__),
    )
    write_json(output / "protocol.json", protocol)
    results = {}
    for arm in ARMS:
        fit_data = train["synthetic_only" if arm == "synthetic_only" else "real_only"]
        model = fit_arm(train[arm], fit_data, protocol["alpha"][arm])
        model_path = output / f"{arm}.joblib"
        joblib.dump(model, model_path, compress=3)
        tables = aggregate_scores(validation, predict(model, validation["X"]))
        levels = ("window", "track", "group")
        results[arm] = dict(train_windows=len(train[arm]["y"]),
                            train_groups=len(set(train[arm]["group_id"])),
                            alpha=protocol["alpha"][arm],
                            validation={level: metrics(table) for level, table in zip(levels, tables)},
                            model_sha256=file_hash(model_path))
        for level, table in zip(levels, tables):
            table = table.copy()
            table["prediction"] = np.where(table.decision >= 0, "drone", "bird")
            table.to_csv(output / f"{arm}_validation_{level}.csv", index=False)
        print(f"{arm}: group macro-F1={results[arm]['validation']['group']['macro_f1']:.4f}", flush=True)
    report = dict(status="development_comparison_only", results=results,
                  validation_reused=manifest["validation_previously_reused"],
                  independent_real_world_evaluation_ready=False,
                  test_opened=False, package_manifest_sha256=protocol["package_manifest_sha256"])
    write_json(output / "results.json", report)
    lines = ["# B-3: Three Training Cohorts on Real A Validation", "",
             "All three runs use the same MiniRocket/Ridge implementation and the same real A validation. ",
             "This validation was previously reused; the numbers are development evidence, not an independent test.", "",
             "| Training cohort | Windows | Groups | Alpha | Window macro-F1 | Track macro-F1 | Video-group macro-F1 | Bird recall | Drone recall |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        row = results[arm]
        val = row["validation"]
        lines.append(f"| {arm} | {row['train_windows']} | {row['train_groups']} | {row['alpha']:g} | "
                     f"{val['window']['macro_f1']:.4f} | {val['track']['macro_f1']:.4f} | "
                     f"{val['group']['macro_f1']:.4f} | {val['group']['recall']['bird']:.4f} | "
                     f"{val['group']['recall']['drone']:.4f} |")
    lines += ["", "MiniRocket and scaler were fit on real train for the two real arms, and on synthetic train",
              "for synthetic-only. The augmented arm used packaged 10% augmentation weight mass.",
              "Alpha was selected without validation data. See protocol.json and the per-group CSV files.",
              "No new untouched real test exists; do not describe the best row as final real-world accuracy.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path("research/output/real_reference_comparison_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.package, args.output)
