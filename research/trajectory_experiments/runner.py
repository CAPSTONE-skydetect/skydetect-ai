from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

from ai_server.schemas import TrackPoint, TrackSequence
from ai_server.services.classifier import RFClassifier
from ai_server.services.feature_core import build_feature_vector
from ai_server.services.manual_roi_tracker import process_manual_roi_video
from ai_server.services.rule_filter import RuleFilter
from ai_server.tracking_schemas import TrackingTuning
from research.io import write_json
from research.tracking_evaluation import evaluate_dataset, write_comparison_video
from research.trajectory_experiments.manifest import load_and_validate, sha256
from research.trajectory_refinement import refine_track


VARIANTS: dict[str, dict[str, Any]] = {
    "D0": {},
    "D0_repeat": {},
    "N1": {"contrast_recentering_enabled": False},
    "N2": {"contrast_recentering_strength": 0.05},
    "N3": {"contrast_recentering_max_step_px": 0.3},
    "N4": {"contrast_correction_alpha": 0.5},
    "N4_native": {
        "contrast_correction_alpha": 0.5,
        "native_resolution": True,
        "cmc_analysis_width": 1280,
    },
    "N5": {"motion_temporal_alpha": 0.5},
    "N6": {"contrast_local_centroid": True},
    "N7": {"klt_patch_registration": True, "contrast_recentering_enabled": False},
    "N8": {"klt_patch_registration": True},
    "N9": {"klt_patch_registration": True, "contrast_local_centroid": True},
    "E1": {"native_resolution": True, "cmc_analysis_width": 1280},
    "E2a": {"appearance_local_refine": True},
    "E2b": {"appearance_local_refine": True, "appearance_subpixel": True},
    "E2c": {
        "appearance_local_refine": True,
        "appearance_subpixel": True,
        "contrast_subpixel": True,
    },
    "E12": {
        "native_resolution": True,
        "appearance_local_refine": True,
        "appearance_subpixel": True,
        "contrast_subpixel": True,
        "cmc_analysis_width": 1280,
    },
    "E4": {"reacquisition_confirmation_frames": 2},
    "E5": {"temporal_evidence_frames": 3},
}


def run_worker(job_path: str | Path) -> dict[str, Any]:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    sample = job["sample"]
    options = VARIANTS[job["variant"]]
    tuning_data = dict(sample["tracking_tuning"])
    tuning_data.update({key: value for key, value in options.items() if key != "native_resolution"})
    code_root = Path(__file__).parents[2]
    source_hashes = {
        str(path.relative_to(code_root)): sha256(path)
        for package in ("ai_server", "research")
        for path in sorted((code_root / package).rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    started = time.monotonic()
    result = process_manual_roi_video(
        sample["video"],
        source_video_id=f"{sample['sample_id']}-{job['variant']}",
        target_bbox=tuple(sample["target_bbox_source"]),
        init_frame_index=sample["init_frame_index"],
        max_seconds=sample["max_seconds"],
        stabilize=sample["stabilize"],
        tuning=TrackingTuning(**tuning_data),
        output_dir=job["output_dir"],
        resize_width=None if options.get("native_resolution") else sample["resize_width"],
        write_overlay=bool(job.get("write_overlay", False)),
    )
    payload = {
        "status": "completed",
        "sample_id": sample["sample_id"],
        "variant": job["variant"],
        "elapsed_seconds": time.monotonic() - started,
        "artifacts": result.artifacts,
        "metadata": result.metadata,
        "metrics": result.metrics,
        "source_hashes": source_hashes,
    }
    write_json(Path(result.artifacts["run_dir"]) / "source_manifest.json", {
        "source_hashes": source_hashes, "job": job, "python": sys.version,
    })
    write_json(job["result_path"], payload)
    return payload


def run_experiment(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    variants: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest = load_and_validate(manifest_path)
    selected = variants or list(manifest["variants"])
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    root = (Path(output_root) / manifest["experiment_id"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(manifest_path)
    write_json(root / "manifest.resolved.json", manifest)
    write_json(root / "provenance.json", provenance)
    preflight = _preflight(manifest, selected)
    write_json(root / "preflight.json", preflight)
    if dry_run:
        return {"status": "planned", "root": str(root), "preflight": preflight}

    timeout = int(manifest.get("limits", {}).get("sample_timeout_seconds", 600))
    records = _collect_worker_records(root, exclude_variants=set(selected))
    for variant in selected:
        variant_root = root / "variants" / variant
        for sample in manifest["samples"]:
            sample_root = variant_root / sample["sample_id"]
            result_path = sample_root / "worker_result.json"
            sample_root.mkdir(parents=True, exist_ok=True)
            job_path = sample_root / "job.json"
            job = {
                "sample": sample,
                "variant": variant,
                "output_dir": str(sample_root / "runs"),
                "result_path": str(result_path),
                "write_overlay": variant in {"D0", "N1", "N2", "N3", "N4", "N5"},
            }
            write_json(job_path, job)
            command = [
                sys.executable, "-m", "research.trajectory_experiments",
                "worker", "--job", str(job_path),
            ]
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    command, cwd=Path(__file__).parents[2], capture_output=True,
                    text=True, timeout=timeout, check=False,
                )
                if completed.returncode != 0 or not result_path.exists():
                    record = {
                        "status": "failed", "sample_id": sample["sample_id"],
                        "variant": variant, "returncode": completed.returncode,
                        "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                else:
                    record = json.loads(result_path.read_text(encoding="utf-8"))
            except subprocess.TimeoutExpired as error:
                record = {
                    "status": "failed", "sample_id": sample["sample_id"],
                    "variant": variant, "reason": "sample_timeout",
                    "stdout": (error.stdout or "")[-4000:] if isinstance(error.stdout, str) else "",
                    "stderr": (error.stderr or "")[-4000:] if isinstance(error.stderr, str) else "",
                    "elapsed_seconds": time.monotonic() - started,
                }
            records.append(record)
            write_json(root / "status.json", {"status": "running", "records": records})
        _evaluate_variant(manifest, root, variant, records)

    if "D0" in selected:
        _run_postprocessing(manifest, root)
    decision = build_decision(root, selected)
    status = "completed" if all(row["status"] == "completed" for row in records) else "needs_review"
    final = {"status": status, "records": records, "decision": decision}
    write_json(root / "status.json", final)
    return final


def _evaluate_variant(
    manifest: dict[str, Any], root: Path, variant: str, records: list[dict[str, Any]],
) -> None:
    current = {
        row["sample_id"]: row for row in records
        if row["variant"] == variant and row["status"] == "completed"
    }
    input_root = root / "evaluation_inputs" / variant
    comparison_root = root / "evaluation" / variant
    for sample in manifest["samples"]:
        result = current.get(sample["sample_id"])
        if result is None:
            continue
        artifacts = result["artifacts"]
        sample_dir = input_root / sample["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        comparison_video = comparison_root / "comparison_videos" / f"{sample['sample_id']}.mp4"
        payload = {
            "sample_id": sample["sample_id"],
            "provenance_note": f"{variant} / fixed ROI and attempted window",
            "ground_truth": {"path": sample["ground_truth"], "track_ids": sample["track_ids"]},
            "prediction": {
                "trajectory": artifacts["trajectory"],
                "metadata": artifacts["metadata"],
            },
            "alignment": {
                "prediction_frame_offset": sample["prediction_frame_offset"],
                "source_is_trimmed": sample["source_is_trimmed"],
            },
            "b_comparison": {
                "track_sequence": artifacts["track_sequence"],
                "debug_observations": artifacts["debug_observations"],
                "source_video": sample["video"],
            },
        }
        if artifacts.get("overlay"):
            payload["comparison_video"] = str(comparison_video)
        write_json(sample_dir / "sample.json", payload)
    reports = evaluate_dataset(input_root, comparison_root, compare_b=True, replay_cmc=True)
    for report in reports:
        result = current[report["sample_id"]]
        overlay = result["artifacts"].get("overlay")
        if not overlay:
            continue
        sample = next(item for item in manifest["samples"] if item["sample_id"] == report["sample_id"])
        output = comparison_root / "comparison_videos" / f"{report['sample_id']}.mp4"
        write_comparison_video(
            cvat_xml=sample["ground_truth"],
            trajectory_csv=result["artifacts"]["trajectory"],
            metadata_json=result["artifacts"]["metadata"],
            overlay_video=overlay,
            output_video=output,
            sample_id=sample["sample_id"],
            track_ids=tuple(sample["track_ids"]),
            prediction_frame_offset=sample["prediction_frame_offset"],
            source_is_trimmed=sample["source_is_trimmed"],
        )
    _write_c_predictions(comparison_root)


def run_postprocessing(manifest_path: str | Path, experiment_root: str | Path) -> dict[str, Any]:
    manifest = load_and_validate(manifest_path)
    root = Path(experiment_root).resolve()
    _run_postprocessing(manifest, root)
    variants = [
        path.name for path in sorted((root / "evaluation").iterdir())
        if path.is_dir() and path.name in VARIANTS
    ]
    decision = build_decision(root, variants)
    records = _collect_worker_records(root)
    status_payload = {
        "status": "completed",
        "records": records,
        "postprocess": str(root / "evaluation" / "postprocess" / "summary.json"),
        "decision": decision,
    }
    write_json(root / "status.json", status_payload)
    return {
        "status": "completed",
        "record_count": len(records),
        "postprocess": status_payload["postprocess"],
        "decision": decision,
    }


def _collect_worker_records(
    root: Path,
    *,
    exclude_variants: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = exclude_variants or set()
    records = []
    variants_root = root / "variants"
    if not variants_root.exists():
        return records
    for path in sorted(variants_root.glob("*/*/worker_result.json")):
        variant = path.parents[1].name
        if variant in excluded:
            continue
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _run_postprocessing(manifest: dict[str, Any], root: Path) -> None:
    baseline = root / "evaluation" / "D0" / "b_inputs"
    output = root / "evaluation" / "postprocess"
    classifier = RFClassifier()
    rule_filter = RuleFilter()
    rows = []
    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        a_path = baseline / sample_id / "a_track_sequence.json"
        gt_path = baseline / sample_id / "gt_matched_track_sequence.json"
        if not a_path.exists() or not gt_path.exists():
            continue
        a = TrackSequence.model_validate_json(a_path.read_text(encoding="utf-8"))
        gt = TrackSequence.model_validate_json(gt_path.read_text(encoding="utf-8"))
        worker = json.loads(
            (root / "variants" / "D0" / sample_id / "worker_result.json").read_text(encoding="utf-8")
        )
        diagnostics = _read_debug(worker["artifacts"]["debug_observations"])
        gt_runtime = build_feature_vector(gt)
        raw_runtime = build_feature_vector(a)
        rows.append({
            "variant": "D0_raw", "sample_id": sample_id,
            **_compare_tracks(a, gt),
            "a_features": _features(raw_runtime),
            "gt_features": _features(gt_runtime),
            "b_status": raw_runtime.feature_vector.feature_status,
            **_prediction(raw_runtime, classifier, rule_filter),
        })
        for variant, mode, adaptive in [
            ("E3a", "kf", False), ("E3b", "kf", True), ("E6", "rts", True),
        ]:
            result = refine_track(
                a, mode=mode, adaptive_measurement=adaptive,
                observation_diagnostics=diagnostics,
            )
            variant_dir = output / variant / sample_id
            write_json(variant_dir / "track_sequence.json", result.track.model_dump(mode="json"))
            write_json(variant_dir / "diagnostics.json", result.diagnostics)
            comparison = _compare_tracks(result.track, gt)
            runtime = build_feature_vector(result.track)
            prediction = _prediction(runtime, classifier, rule_filter)
            rows.append({
                "variant": variant, "sample_id": sample_id,
                **comparison,
                "a_features": _features(runtime),
                "gt_features": _features(gt_runtime),
                "b_status": runtime.feature_vector.feature_status,
                **prediction,
            })
    write_json(output / "summary.json", rows)


def _features(runtime) -> dict[str, Any] | None:
    features = runtime.feature_vector.features
    return features.model_dump() if features else None


def _compare_tracks(a: TrackSequence, gt: TrackSequence) -> dict[str, Any]:
    gt_by_frame = {point.frame_index: point for point in gt.history}
    errors = []
    for point in a.history:
        reference = gt_by_frame.get(point.frame_index)
        if reference is None:
            continue
        errors.append(float(np.hypot(
            (point.cx - reference.cx) * a.processed_width,
            (point.cy - reference.cy) * a.processed_height,
        )))
    return {
        "matched_points": len(errors),
        "position_error_median_px": float(np.median(errors)) if errors else None,
        "position_error_p95_px": float(np.quantile(errors, 0.95)) if errors else None,
    }


def _write_c_predictions(evaluation_root: Path) -> None:
    classifier = RFClassifier()
    rule_filter = RuleFilter()
    rows = []
    for path in sorted((evaluation_root / "b_inputs").glob("*/*_track_sequence.json")):
        track = TrackSequence.model_validate_json(path.read_text(encoding="utf-8"))
        runtime = build_feature_vector(track)
        rows.append({
            "sample_id": path.parent.name,
            "input": path.name.removesuffix("_track_sequence.json"),
            **_prediction(runtime, classifier, rule_filter),
        })
    write_json(evaluation_root / "c_predictions.json", rows)


def _prediction(runtime, classifier, rule_filter) -> dict[str, Any]:
    filtered = rule_filter.apply(runtime.feature_vector)
    if not filtered.passed or not runtime.accepted:
        return {
            "rule_passed": filtered.passed,
            "reject_reason": filtered.reject_reason,
            "prediction": "not_run",
            "prediction_confidence": None,
        }
    label, confidence = classifier.predict(runtime.feature_vector)
    return {
        "rule_passed": True,
        "reject_reason": None,
        "prediction": label,
        "prediction_confidence": confidence,
    }


def _read_debug(path: str | Path) -> dict[int, dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {int(row["frame_index"]): row for row in csv.DictReader(handle)}


def _preflight(manifest: dict[str, Any], variants: list[str]) -> dict[str, Any]:
    return {
        "status": "ready",
        "sample_count": len(manifest["samples"]),
        "variants": variants,
        "estimated_tracking_jobs": len(manifest["samples"]) * len(variants),
        "blocked": [],
    }


def _provenance(manifest_path: str | Path) -> dict[str, Any]:
    root = Path(__file__).parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary"], cwd=root, capture_output=True, check=True,
    ).stdout
    return {
        "code_commit": commit,
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "manifest_sha256": sha256(manifest_path),
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": __import__("cv2").__version__,
        "pid": os.getpid(),
    }


def build_decision(root: Path, variants: list[str]) -> dict[str, Any]:
    baseline_path = root / "evaluation" / "D0" / "summary.json"
    if not baseline_path.exists():
        return {"status": "blocked", "reason": "D0 evaluation missing"}
    baseline = {row["sample_id"]: row for row in json.loads(baseline_path.read_text()) ["samples"]}
    decisions = []
    for variant in variants:
        if variant in {"D0", "D0_repeat"}:
            continue
        path = root / "evaluation" / variant / "summary.json"
        if not path.exists():
            decisions.append({"variant": variant, "status": "blocked"})
            continue
        current = {row["sample_id"]: row for row in json.loads(path.read_text())["samples"]}
        regressions = []
        improvements = []
        for sample_id, base in baseline.items():
            if sample_id not in current:
                regressions.append(f"{sample_id}:missing")
                continue
            candidate = current[sample_id]
            base_p95 = base["reference_metrics"]["error_px"]["p95"]
            candidate_p95 = candidate["reference_metrics"]["error_px"]["p95"]
            base_coverage = base["coverage"]["attempted_window_observation_ratio"]
            candidate_coverage = candidate["coverage"]["attempted_window_observation_ratio"]
            tolerance = max(base_p95 * 0.05, 0.25)
            if candidate_p95 > base_p95 + tolerance:
                regressions.append(f"{sample_id}:position_p95")
            if candidate_coverage < base_coverage - 0.01:
                regressions.append(f"{sample_id}:coverage")
            improvements.append((base_p95 - candidate_p95) / max(base_p95, 1e-9))
        median_improvement = float(np.median(improvements)) if improvements else None
        decisions.append({
            "variant": variant,
            "status": "needs_review" if regressions else (
                "pass" if median_improvement is not None and median_improvement >= 0.10 else "reject"
            ),
            "median_position_p95_improvement": median_improvement,
            "regressions": regressions,
            "note": "manual false-reacquisition and motion-signal review still required",
        })
    result = {"status": "completed", "decisions": decisions}
    write_json(root / "evaluation" / "decision.json", result)
    return result
