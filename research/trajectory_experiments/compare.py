"""Compare completed A variants, including B on exactly shared observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import numpy as np

from ai_server.schemas import TrackSequence
from ai_server.services.feature_core import build_feature_vector
from research.features import FEATURE_COLUMNS
from research.io import write_json


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _features(path: Path, frames: set[int]):
    track = TrackSequence.model_validate_json(path.read_text(encoding="utf-8"))
    history = [point for point in track.history if point.frame_index in frames]
    result = build_feature_vector(track.model_copy(update={"history": history, "quality": None}))
    if not result.accepted:
        return {"status": "rejected", "reasons": result.reasons}
    return {"status": "accepted", "values": result.feature_vector.features.model_dump()}


def compare(root: Path, variants: list[str]) -> dict:
    baseline = {row["sample_id"]: row for row in _read(root / "D0/summary.json")["samples"]}
    outputs = []
    for variant in variants:
        samples = {row["sample_id"]: row for row in _read(root / variant / "summary.json")["samples"]}
        if samples.keys() != baseline.keys():
            raise ValueError("variant sample set differs from baseline")
        objects = []
        for sample_id, base in baseline.items():
            candidate = samples[sample_id]
            base_frames = {row["frame_index"]: row for row in base["frames"]}
            candidate_frames = {row["frame_index"]: row for row in candidate["frames"]}
            common = set(base_frames) & set(candidate_frames)
            spatial = {}
            b_features = {}
            for name, report, frames in (("D0", base, base_frames), (variant, candidate, candidate_frames)):
                scale = 1280.0 / report["video"]["source_width"]
                selected = [frames[key] for key in sorted(common)]
                if any(frames[key]["timestamp_ms"] != base_frames[key]["timestamp_ms"] for key in common):
                    raise ValueError("variant timestamps differ")
                spatial[name] = {
                    "p95_px_1280": float(np.quantile([row["error_px"] * scale for row in selected], .95)),
                    "normalized_width_p95": float(np.quantile([row["error_bbox_width"] for row in selected], .95)),
                    "coverage": report["coverage"]["attempted_window_observation_ratio"],
                    "longest_missing_run_frames": report["timeline"]["longest_missing_run_frames"],
                    "jitter_p95_px_1280": report["reference_metrics"]["error_change_px"]["p95"],
                }
                folder = root / name / "b_inputs" / sample_id
                predicted = _features(folder / "a_track_sequence.json", common)
                gt = _features(folder / "gt_matched_track_sequence.json", common)
                if predicted["status"] != "accepted" or gt["status"] != "accepted":
                    b_features[name] = {"status": "not_comparable", "prediction": predicted, "gt": gt}
                    continue
                differences = {}
                for feature in FEATURE_COLUMNS:
                    a, g = predicted["values"][feature], gt["values"][feature]
                    differences[feature] = {"a": a, "gt": g, "absolute_error": abs(a - g),
                                            "relative_error_percent": 100 * abs(a - g) / abs(g) if abs(g) > 1e-9 else None}
                b_features[name] = {"status": "completed", "features": differences}
            objects.append({"sample_id": sample_id, "common_frames": len(common),
                            "spatial": spatial, "b_features": b_features})
        aggregated = {}
        for name in ("D0", variant):
            acc = [row["b_features"][name]["features"]["acceleration_p95"]["relative_error_percent"]
                   for row in objects if row["b_features"][name]["status"] == "completed"]
            aggregated[name] = {
                "position_p95_median_px_1280_common_frames": median(row["spatial"][name]["p95_px_1280"] for row in objects),
                "coverage_macro": mean(row["spatial"][name]["coverage"] for row in objects),
                "acceleration_error_median_percent_common_frames": median(acc) if len(acc) == len(objects) and all(v is not None for v in acc) else None,
                "b_comparable_objects": len(acc),
            }
        outputs.append({"variant": variant, "aggregate": aggregated, "objects": objects})
    result = {"scope": "development, paired common observed frames per variant versus D0",
              "jitter_scope": "each variant's own consecutive observed pairs; also inspect coverage",
              "comparisons": outputs}
    write_json(root / "paired_comparison.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    args = parser.parse_args()
    result = compare(args.root, args.variants)
    print(json.dumps([{key: row[key] for key in ("variant", "aggregate")} for row in result["comparisons"]], indent=2))
