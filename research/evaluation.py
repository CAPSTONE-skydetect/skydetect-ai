"""Group-held-out classification: synthetic accuracy is NOT real-world validity."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from . import FEATURE_VERSION
from .features import BBOX_COLUMNS, FEATURE_COLUMNS, MOTION_COLUMNS
from .io import write_json


def validate_table(table):
    required = set(FEATURE_COLUMNS + ["sample_id", "family_id", "label", "split", "feature_status",
                                      "feature_version", "feature_config_id"])
    if not required.issubset(table.columns):
        raise ValueError(f"Missing columns: {sorted(required-set(table.columns))}")
    if table.empty or set(table.feature_version.astype(str)) != {FEATURE_VERSION}:
        raise ValueError("Only feature version 3.0.0 is supported")
    if table.feature_config_id.isna().any() or table.feature_config_id.nunique() != 1:
        raise ValueError("Feature configurations cannot be mixed")
    if table.sample_id.duplicated().any() or table.family_id.isna().any():
        raise ValueError("Duplicate sample IDs or missing family IDs")
    if not set(table.label).issubset({"bird", "drone"}):
        raise ValueError("Labels must be bird or drone")
    if not set(table.split).issubset({"train", "validation", "test", "calibration"}):
        raise ValueError("Unknown split")
    if (table.groupby("family_id").split.nunique() > 1).any():
        raise ValueError("Family/video leakage across splits")
    valid = table[table.feature_status == "accepted"].copy()
    if not np.isfinite(valid[FEATURE_COLUMNS].to_numpy(float)).all():
        raise ValueError("Accepted samples contain nonfinite features")
    return valid


def make_model(seed):
    return RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=3,
                                  class_weight="balanced", random_state=seed, n_jobs=1)


def score_predictions(y, prediction, probability):
    labels = ["bird", "drone"]
    matrix = confusion_matrix(y, prediction, labels=labels)
    both_classes = len(set(y)) == 2
    return dict(balanced_accuracy=float(balanced_accuracy_score(y, prediction)) if both_classes else None,
                macro_f1=float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)) if both_classes else None,
                observed_accuracy=float(np.mean(np.asarray(y) == prediction)),
                auroc=float(roc_auc_score(np.asarray(y) == "drone", probability)) if len(set(y)) == 2 else None,
                labels=labels, confusion_matrix=matrix.tolist(),
                recall={label: float(matrix[i, i]/matrix[i].sum()) if matrix[i].sum() else None
                        for i, label in enumerate(labels)})


def cluster_interval(table, predictions, probabilities, seed, repetitions=300):
    groups = table.family_id.unique()
    if len(groups) < 5:
        return dict(status="insufficient_independent_groups", groups=len(groups))
    rng = np.random.default_rng(seed)
    indices = [np.flatnonzero(table.family_id.to_numpy() == g) for g in groups]
    values = []
    y = table.label.to_numpy()
    for _ in range(repetitions):
        pick = np.concatenate([indices[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(set(y[pick])) == 2:
            values.append(balanced_accuracy_score(y[pick], predictions[pick]))
    return dict(status="approximate_cluster_bootstrap", groups=len(groups), repetitions=len(values),
                balanced_accuracy_95_percentile=np.quantile(values, [.025, .975]).tolist() if values else None)


def evaluate_dataset(table, seed=42, real_table=None):
    valid = validate_table(table)
    train, test = valid[valid.split == "train"], valid[valid.split == "test"]
    if min(train.label.nunique(), test.label.nunique()) < 2:
        return dict(status="insufficient_train_test_classes", scope="synthetic_only")
    report = dict(status="completed", scope="synthetic_only", feature_version=FEATURE_VERSION,
                  feature_config_id=str(valid.feature_config_id.iloc[0]),
                  train_groups=train.family_id.nunique(), test_groups=test.family_id.nunique(),
                  train_rows=len(train), test_rows=len(test),
                  warning="Synthetic holdout measures simulator discrimination, not sim-to-real validity.",
                  selection="Fixed hyperparameters; validation and test are never used to fit or tune.", ablations={})
    trained = None
    for name, columns in (("motion", MOTION_COLUMNS), ("bbox_only", BBOX_COLUMNS), ("all", FEATURE_COLUMNS)):
        model = make_model(seed).fit(train[columns], train.label)
        prediction = model.predict(test[columns])
        probability = model.predict_proba(test[columns])[:, list(model.classes_).index("drone")]
        metrics = score_predictions(test.label, prediction, probability)
        metrics["interval"] = cluster_interval(test, prediction, probability, seed)
        metrics["feature_importance"] = dict(zip(columns, model.feature_importances_.tolist()))
        metrics["subtype_recall"] = {str(k): float(np.mean(prediction[test.subtype.to_numpy() == k]
                                                          == test.label.to_numpy()[test.subtype.to_numpy() == k]))
                                     for k in test.subtype.unique()} if "subtype" in test else {}
        metrics["slices"] = {}
        for dimension in ("scenario", "observation_profile", "training_length_group"):
            if dimension in test:
                metrics["slices"][dimension] = {}
                for value in test[dimension].dropna().unique():
                    mask = test[dimension].to_numpy() == value
                    metrics["slices"][dimension][str(value)] = dict(rows=int(mask.sum()),
                        **score_predictions(test.label.to_numpy()[mask], prediction[mask], probability[mask]))
        report["ablations"][name] = metrics
        if name == "all":
            trained = model
    # Stable random labels per underlying trajectory, shared between paired variants.
    from .pipeline import stable_seed
    def random_labels(frame):
        keys = frame.sample_id.str.rsplit(":", n=1).str[0]
        return np.array(["bird" if stable_seed(seed, key, "null") % 2 else "drone" for key in keys])
    null_train, null_test = random_labels(train), random_labels(test)
    null_model = make_model(seed).fit(train[FEATURE_COLUMNS], null_train)
    report["random_label_sanity"] = dict(
        balanced_accuracy=float(balanced_accuracy_score(null_test, null_model.predict(test[FEATURE_COLUMNS]))),
        interpretation="Single deterministic null run, diagnostic only; not a significance test.")
    if real_table is not None:
        real = validate_table(real_table)
        if real_table.feature_config_id.iloc[0] != train.feature_config_id.iloc[0]:
            raise ValueError("Real and synthetic feature configurations differ")
        if "review_status" not in real:
            raise ValueError("Real holdout requires explicit manual review_status")
        real = real[(real.split == "test") & (real.review_status == "approved")]
        if real.label.nunique() < 2:
            report["real_holdout"] = dict(status="insufficient_labeled_real_test_classes")
        else:
            predicted = trained.predict(real[FEATURE_COLUMNS])
            probability = trained.predict_proba(real[FEATURE_COLUMNS])[:, list(trained.classes_).index("drone")]
            report["real_holdout"] = score_predictions(real.label, predicted, probability)
            report["real_holdout"]["interval"] = cluster_interval(real, predicted, probability, seed)
            report["real_holdout"]["warning"] = "Do not calibrate or select models using these test videos."
            report["real_holdout"]["reviewed_test_rows"] = len(real)
    else:
        report["real_holdout"] = dict(status="not_measured_no_real_A")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--test-csv", type=Path,
                        help="Optional separate test CSV; rows are combined before grouped evaluation")
    parser.add_argument("--real-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.csv)
    if args.test_csv:
        table = pd.concat([table, pd.read_csv(args.test_csv)], ignore_index=True)
    report = evaluate_dataset(table, real_table=pd.read_csv(args.real_csv) if args.real_csv else None)
    write_json(args.output, report)
    print(report["status"])


if __name__ == "__main__":
    main()
