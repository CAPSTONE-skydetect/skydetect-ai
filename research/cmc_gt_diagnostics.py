"""Diagnose whether tracking, CMC, or source switching drives GT error spikes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ai_server.schemas import StabilizationInfo, TrackPoint, TrackSequence
from research.tracking_feature_comparison import _comparison, _feature_result


DEFAULT_SAMPLES = ("bird_2_apart_fix", "drone_5_apart_fix", "drone_6_apart_fix")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_debug(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["frame_index"]): row for row in csv.DictReader(handle)}


def _bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _camera_steps(path: Path, width: int, height: int) -> dict[int, float]:
    payload = _read_json(path)
    matrices = {
        int(item["frame_index"]): np.asarray(item["current_to_reference"], dtype=float)
        for item in payload["frames"]
    }
    center = np.array([width / 2.0, height / 2.0, 1.0], dtype=float)
    result: dict[int, float] = {}
    previous_index: int | None = None
    for frame_index in sorted(matrices):
        if previous_index is None:
            result[frame_index] = 0.0
        else:
            local = np.linalg.inv(matrices[previous_index]) @ matrices[frame_index]
            moved = local @ center
            moved /= moved[2]
            result[frame_index] = float(np.linalg.norm(moved[:2] - center[:2]))
        previous_index = frame_index
    return result


def _raw_tracks(report: dict[str, Any]) -> tuple[TrackSequence, TrackSequence]:
    video = report["video"]
    width, height = int(video["source_width"]), int(video["source_height"])
    a_points, gt_points = [], []
    for row in report["frames"]:
        frame_index = int(row["frame_index"])
        timestamp_ms = int(row["timestamp_ms"])
        gt_width, gt_height = row["gt_bbox_size"]
        common = {
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "w": min(1.0, float(gt_width) / width),
            "h": min(1.0, float(gt_height) / height),
        }
        a_points.append(TrackPoint(
            **common,
            cx=float(row["prediction_center"][0]) / width,
            cy=float(row["prediction_center"][1]) / height,
            conf=float(row["confidence"]),
        ))
        gt_points.append(TrackPoint(
            **common,
            cx=float(row["gt_center"][0]) / width,
            cy=float(row["gt_center"][1]) / height,
            conf=1.0,
        ))
    base = {
        "track_id": int(report["target"]["track_id"]),
        "source_video_id": report["sample_id"],
        "processed_width": width,
        "processed_height": height,
        "stabilization": StabilizationInfo(applied=False, method="none"),
    }
    return TrackSequence(**base, history=a_points), TrackSequence(**base, history=gt_points)


def _feature_delta(rows: list[dict[str, Any]], feature: str) -> float | None:
    for row in rows:
        if row["feature"] == feature:
            return row["relative_difference_percent"]
    return None


def diagnose_sample(evaluation_root: Path, variant: str, sample_id: str) -> dict[str, Any]:
    report_path = evaluation_root / variant / "per_video" / f"{sample_id}.json"
    report = _read_json(report_path)
    trajectory = Path(report["inputs"]["trajectory_csv"])
    debug = _read_debug(trajectory.parent / "debug" / "observations.csv")
    b_dir = evaluation_root / variant / "b_inputs" / sample_id
    camera_path = b_dir / "camera_transforms.json"
    camera_steps = _camera_steps(
        camera_path,
        int(report["video"]["processed_width"]),
        int(report["video"]["processed_height"]),
    )

    rows = report["frames"]
    errors = [float(row["error_px"]) for row in rows]
    threshold = _percentile(errors, 0.95) or 0.0
    top_rows = [row for row in rows if float(row["error_px"]) >= threshold]
    all_sources = Counter(str(row.get("tracking_source") or "unknown") for row in rows)
    top_sources = Counter(str(row.get("tracking_source") or "unknown") for row in top_rows)

    joined = []
    for row in rows:
        frame_index = int(row["frame_index"])
        debug_row = debug.get(frame_index, {})
        joined.append({
            "frame_index": frame_index,
            "error_px": float(row["error_px"]),
            "tracking_source": str(row.get("tracking_source") or "unknown"),
            "background_inlier_ratio": float(debug_row.get("background_inlier_ratio") or 0.0),
            "background_points": int(float(debug_row.get("background_points") or 0)),
            "camera_step_px": float(camera_steps.get(frame_index, 0.0)),
            "visible": _bool(debug_row.get("visible")),
        })

    raw_a, raw_gt = _raw_tracks(report)
    raw_a_result, raw_gt_result = _feature_result(raw_a), _feature_result(raw_gt)
    raw_comparison = _comparison(raw_a_result, raw_gt_result)
    cmc_comparison = report["b_features"].get("comparisons", [])
    feature_names = ("speed_cv", "acceleration_p95", "turn_rate_median", "tortuosity")

    return {
        "sample_id": sample_id,
        "matched_frames": len(rows),
        "error_px": {
            "median": _percentile(errors, 0.5),
            "p95": threshold,
            "max": max(errors) if errors else None,
        },
        "error_correlations": {
            "camera_step_px": _correlation(
                [row["error_px"] for row in joined],
                [row["camera_step_px"] for row in joined],
            ),
            "background_inlier_ratio": _correlation(
                [row["error_px"] for row in joined],
                [row["background_inlier_ratio"] for row in joined],
            ),
        },
        "camera_step_px": {
            "median": _percentile([row["camera_step_px"] for row in joined], 0.5),
            "p95": _percentile([row["camera_step_px"] for row in joined], 0.95),
            "max": max((row["camera_step_px"] for row in joined), default=None),
        },
        "background": {
            "inlier_ratio_median": _percentile(
                [row["background_inlier_ratio"] for row in joined], 0.5
            ),
            "point_count_median": _percentile(
                [float(row["background_points"]) for row in joined], 0.5
            ),
        },
        "tracking_sources": {
            "all": dict(all_sources),
            "top_5_percent_error": dict(top_sources),
        },
        "worst_frames": sorted(joined, key=lambda row: row["error_px"], reverse=True)[:10],
        "b_feature_relative_error_percent": {
            feature: {
                "raw_frame_coordinates": _feature_delta(raw_comparison, feature),
                "cmc_coordinates": _feature_delta(cmc_comparison, feature),
            }
            for feature in feature_names
        },
        "camera_replay_max_residual_px": report["b_features"]
        .get("camera_verification", {})
        .get("max_residual_px"),
        "interpretation_guardrail": (
            "GT and A receive the same replayed CMC transform. Localization error cannot "
            "independently validate camera motion, but raw-vs-CMC B features can expose "
            "motion-signal sensitivity to that transform."
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    def number(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        "# CMC / GT diagnostic",
        "",
        "The same replayed CMC transform is applied to A and GT. Therefore this report "
        "does not treat low localization error as independent proof that CMC is correct.",
        "",
        "| sample | error P95 | camera step P95 | corr(error, camera) | corr(error, inlier) | top-error sources |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["samples"]:
        sources = ", ".join(
            f"{name}:{count}" for name, count in row["tracking_sources"]["top_5_percent_error"].items()
        )
        corr_camera = row["error_correlations"]["camera_step_px"]
        corr_inlier = row["error_correlations"]["background_inlier_ratio"]
        lines.append(
            f"| {row['sample_id']} | {row['error_px']['p95']:.3f}px | "
            f"{row['camera_step_px']['p95']:.3f}px | "
            f"{number(corr_camera)} | {number(corr_inlier)} | {sources} |"
        )
    lines.extend(["", "## Raw vs CMC B-feature relative error", ""])
    for row in payload["samples"]:
        lines.extend([
            f"### {row['sample_id']}",
            "",
            "| feature | raw | CMC |",
            "|---|---:|---:|",
        ])
        for feature, values in row["b_feature_relative_error_percent"].items():
            raw, cmc = values["raw_frame_coordinates"], values["cmc_coordinates"]
            raw_text = "n/a" if raw is None else f"{raw:.2f}%"
            cmc_text = "n/a" if cmc is None else f"{cmc:.2f}%"
            lines.append(f"| {feature} | {raw_text} | {cmc_text} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def run(evaluation_root: Path, variant: str, samples: tuple[str, ...], output: Path) -> dict[str, Any]:
    payload = {
        "variant": variant,
        "samples": [diagnose_sample(evaluation_root, variant, sample) for sample in samples],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--variant", default="D0")
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.evaluation_root, args.variant, tuple(args.samples), args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
