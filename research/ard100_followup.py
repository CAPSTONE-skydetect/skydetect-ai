"""Prepare fixed development subsets and compare localization, not just labels."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def prepare(baseline, output, scope, variant):
    config = json.loads((baseline / "protocol.json").read_text())
    samples = config["samples"]
    if scope == "dev12":
        eligible = [s for s in samples if s["split"] == "train" and s["status"] == "ready"]
        samples = eligible[::max(1, len(eligible)//12)][:12]
    else:
        samples = [s for s in samples if s["split"] == scope]
    config.update(samples=samples, variant=variant, baseline=str(baseline.resolve()),
                  selection="lexicographic stride on ready train IDs only; no GT error or C outcome used" if scope == "dev12" else "all original split videos, including preflight failures",
                  development_scope=scope)
    config["d0_tuning"].update(target_scale_klt=variant not in {"AR3", "AR4"})
    if variant in {"AR3", "AR4"}:
        config["d0_tuning"].update(motion_lock_audit=True)
    if variant == "AR4":
        config["d0_tuning"].update(motion_audit_min_score=.38)
    if variant == "AR2":
        config["d0_tuning"].update(contrast_recentering_enabled=False)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(json.dumps(dict(variant=variant, scope=scope, samples=[s["sample_id"] for s in samples]), indent=2))


def compare(baseline, output):
    config = json.loads((output / "protocol.json").read_text())
    rows = []
    for sample in config["samples"]:
        for label, root in (("D0", baseline), (config["variant"], output)):
            path = root / "samples" / sample["sample_id"] / "result.json"
            result = json.loads(path.read_text()) if path.exists() else {}
            report = result.get("tracking_evaluation", {})
            ref = report.get("reference_metrics", {})
            success = report.get("localization", {}).get("normalized_threshold_success", {}).get("within_0_5_bbox_diagonal", {})
            row = dict(sample_id=sample["sample_id"], variant=label,
                       status=result.get("status", sample["status"]),
                       correct_location_requested_ratio=success.get("full_gt_ratio", 0.),
                       observation_requested_ratio=report.get("coverage", {}).get("full_gt_observation_ratio", 0.),
                       median_error_px_1280=ref.get("error_px", {}).get("median"),
                       p95_error_px_1280=ref.get("error_px", {}).get("p95"),
                       median_error_bbox_width=ref.get("error_bbox_width", {}).get("median"),
                       jitter_median_px_1280=ref.get("error_change_px", {}).get("median"))
            rows.append(row)
    aggregate = []
    for label in ("D0", config["variant"]):
        selected = [r for r in rows if r["variant"] == label]
        item = dict(variant=label, requested=len(selected),
                    mean_correct_localization_ratio=mean(r["correct_location_requested_ratio"] for r in selected),
                    mean_observation_ratio=mean(r["observation_requested_ratio"] for r in selected))
        for key in ("median_error_px_1280", "p95_error_px_1280", "median_error_bbox_width", "jitter_median_px_1280"):
            values = [r[key] for r in selected if r[key] is not None]
            item[key] = median(values) if values else None
        aggregate.append(item)
    old, new = aggregate
    gate = dict(localization_improved=new["mean_correct_localization_ratio"] > old["mean_correct_localization_ratio"],
                observation_drop_at_most_2pp=new["mean_observation_ratio"] >= old["mean_observation_ratio"] - .02,
                median_position_not_worse=(new["median_error_px_1280"] is not None
                                           and old["median_error_px_1280"] is not None
                                           and new["median_error_px_1280"] <= old["median_error_px_1280"]))
    result = dict(scope=config["development_scope"], summaries=aggregate, gate=gate,
                  passes_all=all(gate.values()),
                  missing_samples_count_as_zero_localization=True,
                  meaning="error <= half GT bbox diagonal over all requested GT frames; no dropped-frame reward")
    with (output / "tracking_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "tracking_comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "compare"])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=["dev12", "train", "test"], default="dev12")
    parser.add_argument("--variant", choices=["AR1", "AR2", "AR3", "AR4"], default="AR1")
    a = parser.parse_args()
    if a.command == "prepare":
        prepare(a.baseline.resolve(), a.output.resolve(), a.scope, a.variant)
    else:
        compare(a.baseline.resolve(), a.output.resolve())
