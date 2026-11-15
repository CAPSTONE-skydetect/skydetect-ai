"""Audit B-3 errors without fitting or selecting another model."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_real_sequence_dataset import file_hash
from .evaluate_sequence_handoff import aggregate_scores, load_arrays, metrics
from .io import write_json
from .sequence_comparison import sequence_metrics


ARMS = ("real_only", "real_plus_augmentation", "synthetic_only")
DIAGNOSTICS = ("screen_span", "screen_path", "straightness", "turn_radians_mean",
               "reversal_fraction", "low_motion_fraction", "step_cv", "band_3_10_ratio")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_aligned_metadata(path, arrays, split):
    frame = pd.read_csv(path, keep_default_na=False)
    frame = frame.loc[frame.split == split].sort_values("npz_row").reset_index(drop=True)
    if (len(frame) != len(arrays["y"]) or
            not np.array_equal(frame.sample_id.to_numpy(dtype=str), arrays["sample_id"]) or
            not np.array_equal(frame.source_group_id.to_numpy(dtype=str), arrays["group_id"]) or
            not np.array_equal(frame.label.to_numpy(dtype=str), arrays["y"])):
        raise ValueError(f"Metadata does not align with {split} arrays")
    return frame


def diagnostic_windows(arrays, metadata):
    rows = []
    for x, row in zip(arrays["X"], metadata.itertuples()):
        values = sequence_metrics(x, float(row.normalization_scale))
        rows.append(dict(sample_id=row.sample_id, group_id=row.source_group_id,
                         label=row.label, parent_track_id=row.parent_track_id,
                         missing_fraction=float(row.missing_fraction),
                         max_missing_seconds=float(row.max_missing_seconds),
                         normalization_scale=float(row.normalization_scale),
                         quality_flags=row.quality_flags, **values))
    return pd.DataFrame(rows)


def checked_predictions(evaluation, arm, arrays):
    tables = {}
    for level in ("window", "track", "group"):
        table = pd.read_csv(evaluation / f"{arm}_validation_{level}.csv", keep_default_na=False)
        if not {"group_id", "label", "decision", "prediction"}.issubset(table.columns):
            raise ValueError(f"Missing prediction columns: {arm} {level}")
        expected = np.where(table.decision.to_numpy(dtype=float) >= 0, "drone", "bird")
        if not np.array_equal(expected, table.prediction.to_numpy(dtype=str)):
            raise ValueError(f"Score/prediction mismatch: {arm} {level}")
        tables[level] = table
    window = tables["window"]
    if (len(window) != len(arrays["y"]) or
            not np.array_equal(window.sample_id.to_numpy(dtype=str), arrays["sample_id"]) or
            not np.array_equal(window.group_id.to_numpy(dtype=str), arrays["group_id"]) or
            not np.array_equal(window.label.to_numpy(dtype=str), arrays["y"])):
        raise ValueError(f"Window predictions do not match validation: {arm}")
    rebuilt = aggregate_scores(arrays, window.decision.to_numpy(dtype=float))
    for level, table, calculated in zip(("window", "track", "group"),
                                         tables.values(), rebuilt):
        keys = ["group_id", "label"]
        if level == "track":
            keys.append("parent_track_id")
        if level == "window":
            keys.append("sample_id")
        if (not table[keys].equals(calculated[keys]) or
                not np.allclose(table.decision, calculated.decision, atol=1e-9)):
            raise ValueError(f"Aggregation changed: {arm} {level}")
    return tables


def audit(package, evaluation, source_audit, output):
    package, evaluation, source_audit, output = map(Path, (package, evaluation, source_audit, output))
    if output.exists():
        raise ValueError("Output exists; choose a new directory")
    manifest, protocol, results = (read_json(path) for path in
                                   (package / "dataset_manifest.json", evaluation / "protocol.json",
                                    evaluation / "results.json"))
    if (results["status"] != "development_comparison_only" or
            results["package_manifest_sha256"] != file_hash(package / "dataset_manifest.json") or
            protocol["package_manifest_sha256"] != results["package_manifest_sha256"] or
            results["test_opened"] or protocol["test_opened"]):
        raise ValueError("B-3 provenance or scope mismatch")
    for name, checksum in manifest["file_hashes"].items():
        if file_hash(package / name) != checksum:
            raise ValueError(f"Package file changed: {name}")
    validation = load_arrays(package / manifest["common_validation"])
    real_train = load_arrays(package / manifest["arms"]["real_only"])
    metadata_path = package / "real_metadata.csv"
    train_meta = load_aligned_metadata(metadata_path, real_train, "train")
    val_meta = load_aligned_metadata(metadata_path, validation, "validation")
    diagnostic = diagnostic_windows(validation, val_meta)
    train_diagnostic = diagnostic_windows(real_train, train_meta)
    source = pd.read_csv(source_audit, keep_default_na=False)
    source = source.loc[source.split == "validation", ["source_group_id", "video_name", "video_path",
                                                        "track_file", "frame_ok", "clock_ok", "hash_match"]]
    if (set(source.source_group_id) != set(validation["group_id"]) or
            (source.groupby("source_group_id")[["video_name", "video_path"]].nunique() > 1).any().any()):
        raise ValueError("Video audit does not identify each validation video")
    if (not source[["frame_ok", "clock_ok"]].astype(str).apply(
            lambda col: col.str.lower().eq("true")).all().all() or
            not source.hash_match.astype(str).str.lower().isin(("true", "unknown")).all()):
        raise ValueError("Source video correspondence is not sufficiently checked")
    source = source.groupby("source_group_id", as_index=False).agg(
        video_name=("video_name", "first"), video_path=("video_path", "first"),
        track_files=("track_file", lambda values: " | ".join(sorted(values))),
        video_hash_status=("hash_match", lambda values: "verified" if all(
            str(value).lower() == "true" for value in values) else "metadata_only"))
    predictions = {arm: checked_predictions(evaluation, arm, validation) for arm in ARMS}
    for arm, tables in predictions.items():
        for level in ("window", "track", "group"):
            recorded = results["results"][arm]["validation"][level]
            actual = metrics(tables[level])
            if recorded["n"] != actual["n"] or not np.isclose(recorded["macro_f1"], actual["macro_f1"]):
                raise ValueError(f"B-3 result mismatch: {arm} {level}")

    group = (diagnostic.groupby(["group_id", "label"], as_index=False)
             .agg(n_windows=("sample_id", "size"), n_tracks=("parent_track_id", "nunique"),
                  missing_fraction_median=("missing_fraction", "median"),
                  max_missing_seconds=("max_missing_seconds", "max"),
                  normalization_scale_median=("normalization_scale", "median"),
                  **{f"{name}_median": (name, "median") for name in DIAGNOSTICS}))
    group = group.merge(source, left_on="group_id", right_on="source_group_id", validate="one_to_one")
    for arm, tables in predictions.items():
        part = tables["group"][["group_id", "label", "decision", "prediction"]].rename(
            columns={"decision": f"{arm}_margin", "prediction": f"{arm}_prediction"})
        group = group.merge(part, on=["group_id", "label"], validate="one_to_one")
        group[f"{arm}_correct"] = group[f"{arm}_prediction"] == group.label
        counts = tables["window"].assign(correct=lambda t: t.prediction == t.label).groupby("group_id").correct.agg(
            ["sum", "count"]).rename(columns={"sum": f"{arm}_correct_windows", "count": f"{arm}_windows"})
        group = group.merge(counts, left_on="group_id", right_index=True, validate="one_to_one")
    group["aug_minus_real_margin"] = group.real_plus_augmentation_margin - group.real_only_margin
    group["common_real_error"] = ~group.real_only_correct & ~group.real_plus_augmentation_correct
    group["synthetic_additional_error"] = group.real_only_correct & ~group.synthetic_only_correct
    if len(group) != len(set(validation["group_id"])):
        raise ValueError("Lost validation group")

    train_group = (train_diagnostic.groupby(["group_id", "label"], as_index=False)
                   .agg(**{f"{name}_median": (name, "median") for name in DIAGNOSTICS}))
    synthetic = pd.read_csv(package / "synthetic_metadata.csv", keep_default_na=False)
    if not {"group_id", "label", *DIAGNOSTICS}.issubset(synthetic.columns):
        raise ValueError("Synthetic diagnostic metadata is incomplete")
    synthetic_group = synthetic.groupby(["group_id", "label"], as_index=False)[list(DIAGNOSTICS)].median()
    for name in DIAGNOSTICS:
        field = f"{name}_median"
        group[f"{name}_train_class_percentile"] = group.apply(
            lambda row: float(100 * np.mean(
                train_group.loc[train_group.label == row.label, field].to_numpy() <= row[field])), axis=1)
        group[f"{name}_synthetic_class_percentile"] = group.apply(
            lambda row: float(100 * np.mean(
                synthetic_group.loc[synthetic_group.label == row.label, name].to_numpy() <= row[field])), axis=1)
    output.mkdir(parents=True)
    group.sort_values(["label", "group_id"]).to_csv(output / "validation_group_diagnostics.csv", index=False)
    diagnostic.to_csv(output / "validation_window_diagnostics.csv", index=False)
    train_group.to_csv(output / "real_train_group_reference.csv", index=False)
    synthetic_group.to_csv(output / "synthetic_train_group_reference.csv", index=False)
    summary = dict(
        status="descriptive_failure_analysis_not_model_selection",
        created_at=datetime.now(timezone.utc).isoformat(),
        b3_results_sha256=file_hash(evaluation / "results.json"),
        package_manifest_sha256=file_hash(package / "dataset_manifest.json"),
        video_audit_sha256=file_hash(source_audit),
        validation_groups=len(group), validation_tracks=sum(group.n_tracks),
        validation_windows=sum(group.n_windows),
        class_groups={label: int(sum(group.label == label)) for label in ("bird", "drone")},
        errors={arm: {label: int(sum((group.label == label) & ~group[f"{arm}_correct"]))
                      for label in ("bird", "drone")} for arm in ARMS},
        common_real_error_groups=group.loc[group.common_real_error, "group_id"].tolist(),
        synthetic_additional_error_groups=group.loc[group.synthetic_additional_error, "group_id"].tolist(),
        paired_group_prediction_changes=int(sum(group.real_only_prediction != group.real_plus_augmentation_prediction)),
        test_opened=False,
        interpretation="Margins are Ridge scores, not calibrated probabilities; percentiles are descriptive against small real train groups",
    )
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path("research/output/real_reference_comparison_v1"))
    parser.add_argument("--evaluation", type=Path, default=Path("research/output/real_reference_evaluation_b3_v2"))
    parser.add_argument("--source-audit", type=Path,
                        default=Path("research/output/original_video_audit_v1/video_track_matches.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.package, args.evaluation, args.source_audit, args.output), indent=2))
