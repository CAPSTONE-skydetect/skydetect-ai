"""Fit RF and MiniRocket on the same reviewed real tracks and two-second windows.

Run using the MiniRocket runtime environment. An explicit reviewed group split
is required; this module never invents labels or assigns a split from filenames.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ai_server.schemas import TrackSequence
from ai_server.services.feature_core import build_feature_vector
from research.common_c_dataset import _sha256
from research.trajectory_sequence import CONTRACT_VERSION, SequenceConfig, window_track


def validate_manifest(path: Path) -> tuple[dict, list[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    seen_ids = set()
    groups = defaultdict(set)
    classes = defaultdict(set)
    hashes = defaultdict(set)
    records = manifest.get("records", [])
    if not records:
        errors.append("records are empty")
    for record in records:
        sample = record["sample_id"]
        if sample in seen_ids:
            errors.append(f"duplicate sample_id: {sample}")
        seen_ids.add(sample)
        split, label = record.get("partition"), record.get("label")
        group = record.get("source_group_id")
        if not group:
            errors.append(f"{sample}: source_group_id is required")
        groups[group].add(split)
        if split not in {"train", "validation", "test"}:
            errors.append(f"{sample}: explicit train/validation/test partition required")
        if label not in {"bird", "drone"}:
            errors.append(f"{sample}: label must be bird or drone")
        classes[split].add(label)
        if record.get("review_status") != "approved":
            errors.append(f"{sample}: human review_status=approved required")
        if record.get("domain") != "real":
            errors.append(f"{sample}: this real-only experiment requires domain=real")
        for kind in ("gt", "a"):
            value = record.get(f"{kind}_track")
            if not value:
                errors.append(f"{sample}: {kind}_track is required")
                continue
            track_path = (path.parent / value).resolve()
            if not track_path.is_file():
                errors.append(f"{sample}: missing {track_path}")
                continue
            digest = _sha256(track_path)
            if digest != record.get(f"{kind}_sha256"):
                errors.append(f"{sample}: {kind} track hash mismatch")
            hashes[digest].add(group)
            try:
                TrackSequence.model_validate_json(track_path.read_text(encoding="utf-8"))
            except ValueError as error:
                errors.append(f"{sample}: invalid {kind} TrackSequence: {error}")
    for group, splits in groups.items():
        if len(splits) > 1:
            errors.append(f"group leakage: {group} occurs in {sorted(map(str, splits))}")
    for split in ("train", "validation", "test"):
        if classes[split] != {"bird", "drone"}:
            errors.append(f"{split}: both classes are required")
    if any(len(value) > 1 for value in hashes.values()):
        errors.append("identical track files occur in different source groups")
    return manifest, sorted(set(errors))


def _cohorts(path: Path, manifest: dict, kind: str, include_test: bool) -> tuple[dict, list]:
    cohorts = defaultdict(list)
    rejected = []
    for record in manifest["records"]:
        split = record["partition"]
        if split == "test" and not include_test:
            continue
        track = TrackSequence.model_validate_json(
            (path.parent / record[f"{kind}_track"]).read_text(encoding="utf-8")
        )
        try:
            windows, mr_rejections = window_track(track.model_dump(mode="json"))
        except ValueError as error:
            rejected.append({"sample_id": record["sample_id"], "reason": str(error)})
            continue
        rejected.extend({"sample_id": record["sample_id"], **row} for row in mr_rejections)
        for window in windows:
            history = [p for p in track.history if window["window_start_s"] <= p.timestamp_ms / 1000 < window["window_end_s"]]
            if len(history) < 5:
                rejected.append({"sample_id": record["sample_id"], "reason": "too_few_RF_points"})
                continue
            cropped = track.model_copy(update={"history": history, "quality": None})
            runtime = build_feature_vector(cropped)
            features = runtime.feature_vector.features
            if not runtime.accepted or features is None:
                rejected.append({"sample_id": record["sample_id"], "reason": "RF_feature_rejected", "details": runtime.reasons})
                continue
            cohorts[split].append({
                "X": window["X"], "features": features.model_dump(),
                "sample_id": record["sample_id"], "group": record["source_group_id"],
                "label": record["label"], "window_start_s": window["window_start_s"],
                "window_end_s": window["window_end_s"],
            })
    return dict(cohorts), rejected


def _metrics(truth: list[str], labels: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    covered = [i for i, label in enumerate(labels) if label != "uncertain"]
    return {
        "n": len(truth), "accuracy_with_abstentions_as_errors": float(accuracy_score(truth, labels)),
        "macro_f1": float(f1_score(truth, labels, labels=["bird", "drone"], average="macro", zero_division=0)),
        "coverage": len(covered) / len(truth), "abstention_rate": 1 - len(covered) / len(truth),
        "covered_accuracy": float(np.mean([truth[i] == labels[i] for i in covered])) if covered else None,
        "drone_recall": float(recall_score(truth, labels, labels=["drone"], average="macro", zero_division=0)),
        "bird_false_drone_rate": sum(t == "bird" and p == "drone" for t, p in zip(truth, labels)) / max(1, truth.count("bird")),
    }


def run(path: Path, output: Path, *, kind: str = "a", evaluate_test: bool = False) -> dict:
    manifest, errors = validate_manifest(path)
    output.mkdir(parents=True, exist_ok=False)
    if errors:
        result = {"status": "blocked", "errors": errors, "manifest_sha256": _sha256(path)}
        (output / "status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    cohorts, rejected = _cohorts(path, manifest, kind, evaluate_test)
    for split in ("train", "validation") + (("test",) if evaluate_test else ()):
        if {r["label"] for r in cohorts.get(split, [])} != {"bird", "drone"}:
            raise ValueError(f"{split} loses a class after common quality gates")
    from sklearn.ensemble import RandomForestClassifier
    from research.evaluate_sequence_handoff import transformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifier
    import joblib

    train = cohorts["train"]
    feature_names = sorted(train[0]["features"])
    counts = Counter(r["group"] for r in train)
    weights = np.array([1.0 / counts[r["group"]] for r in train])
    y = [r["label"] for r in train]
    rf = RandomForestClassifier(n_estimators=300, random_state=20261018, n_jobs=2)
    rf.fit([[r["features"][f] for f in feature_names] for r in train], y, sample_weight=weights)
    rocket = transformer()
    scaler = StandardScaler(with_mean=False)
    train_z = scaler.fit_transform(rocket.fit_transform(np.stack([r["X"] for r in train]))).astype(np.float64)
    ridge = RidgeClassifier(alpha=1.0).fit(train_z, y, sample_weight=weights)
    results = {}
    for split, rows in cohorts.items():
        if split == "train":
            continue
        probabilities = rf.predict_proba([[r["features"][f] for f in feature_names] for r in rows])
        margins = ridge.decision_function(scaler.transform(rocket.transform(np.stack([r["X"] for r in rows]))).astype(np.float64))
        grouped = defaultdict(list)
        for row, probability, margin in zip(rows, probabilities, margins):
            grouped[row["sample_id"]].append((row, probability, float(margin)))
        predictions = []
        for sample_id, values in sorted(grouped.items()):
            probability = np.mean([v[1] for v in values], axis=0)
            margin = float(np.mean([v[2] for v in values]))
            winner = int(np.argmax(probability))
            binary = str(rf.classes_[winner])
            predictions.append({
                "sample_id": sample_id, "truth": values[0][0]["label"],
                "rf_binary": binary, "rf_service": binary if probability[winner] >= 0.6 else "uncertain",
                "rf_probability": float(probability[winner]), "mr_margin": margin,
                "minirocket": "drone" if margin >= 0 else "bird", "common_windows": len(values),
            })
        truth = [row["truth"] for row in predictions]
        results[split] = {
            "predictions": predictions,
            "metrics": {model: _metrics(truth, [r[model] for r in predictions]) for model in ("rf_binary", "rf_service", "minirocket")},
            "input_track_count": sum(r["partition"] == split for r in manifest["records"]),
            "delivered_track_count": len(predictions),
        }
    joblib.dump({"model": rf, "feature_names": feature_names}, output / "rf.joblib")
    joblib.dump({"rocket": rocket, "scaler": scaler, "classifier": ridge,
                 "classes": ridge.classes_.tolist(), "contract_version": CONTRACT_VERSION,
                 "contract_id": SequenceConfig().fingerprint, "arm": "real_only", "alpha": 1.0,
                 "manifest_sha256": _sha256(path)}, output / "minirocket.joblib")
    result = {
        "status": "completed", "input_kind": kind, "manifest_sha256": _sha256(path),
        "fixed_parameters": {"rf_trees": 300, "ridge_alpha": 1.0, "group_equal_train_weight": True},
        "test_evaluated": evaluate_test, "results": results, "rejections": rejected,
        "window_manifest": [{k: v for k, v in row.items() if k not in {"X", "features"}} | {"partition": split}
                            for split, rows in cohorts.items() for row in rows],
    }
    (output / "status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("a", "gt"), default="a")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    result = run(args.manifest, args.output, kind=args.kind, evaluate_test=args.evaluate_test)
    print(json.dumps({key: value for key, value in result.items() if key != "window_manifest"}, indent=2))


if __name__ == "__main__":
    main()
