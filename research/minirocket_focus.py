"""Audit a frozen production MiniRocket on paired development tracks.

Run with the deployment branch's Python environment. No fitting or threshold
selection is performed. Rotations are sensitivity probes, not a new predictor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def rotate_channels(x: np.ndarray, quarter_turns: int) -> np.ndarray:
    output = x.copy()
    for _ in range(quarter_turns % 4):
        for offset in (0, 2):
            output[:, offset] = -x[:, offset + 1]
            output[:, offset + 1] = x[:, offset]
        x = output.copy()
    return output


def aggregation_scores(scores: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(scores)), "median": float(np.median(scores)),
            "majority": float(np.mean(scores >= 0) - .5),
            "any_positive_exploratory": float(np.max(scores))}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(branch: Path, pilot: Path, output: Path, model_path: Path | None = None) -> dict:
    sys.path.insert(0, str(branch.resolve()))
    from ai_server.schemas import TrackSequence
    from ai_server.services.classifier import MiniRocketClassifier
    from ai_server.services.sequence_model import validate_windows
    from research.minirocket_inference import decision_margins
    from research.trajectory_sequence import CHANNELS, CONTRACT_VERSION, SequenceConfig, window_track
    import joblib

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("audit output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    model_path = model_path or branch / "models/minirocket_classifier.joblib"
    version = joblib.load(model_path).get("contract_version")
    experimental = version != CONTRACT_VERSION
    frame_scale = version == "trajectory-sequence-frame-scale-1.0.0"
    dual_scale = version == "trajectory-sequence-dual-scale-1.0.0"
    stack_scale = version == "trajectory-sequence-dual-scale-stack-1.0.0"
    if experimental:
        if stack_scale:
            if __package__:
                from .minirocket_stack import StackMiniRocketClassifier, decision_margins as dual_margins
            else:
                from minirocket_stack import StackMiniRocketClassifier, decision_margins as dual_margins
            classifier = StackMiniRocketClassifier(model_path)
        elif dual_scale:
            if __package__:
                from .minirocket_dualscale import DualScaleMiniRocketClassifier, decision_margins as dual_margins
            else:
                from minirocket_dualscale import DualScaleMiniRocketClassifier, decision_margins as dual_margins
            classifier = DualScaleMiniRocketClassifier(model_path)
        elif frame_scale:
            if __package__:
                from .minirocket_framescale import FrameScaleMiniRocketClassifier, decision_margins, scale_windows
            else:
                from minirocket_framescale import FrameScaleMiniRocketClassifier, decision_margins, scale_windows
            classifier = FrameScaleMiniRocketClassifier(model_path)
        elif __package__:
            from .minirocket_orbit import OrbitMiniRocketClassifier, decision_margins
            classifier = OrbitMiniRocketClassifier(model_path)
        else:
            from minirocket_orbit import OrbitMiniRocketClassifier, decision_margins
            classifier = OrbitMiniRocketClassifier(model_path)
    else:
        classifier = MiniRocketClassifier(model_path)
    bundle = classifier.model
    heads = [bundle["baseline"]["classifier"], bundle["framescale"]["classifier"]] if dual_scale or stack_scale else [bundle["classifier"]]
    assert bundle["classes"] == ["bird", "drone"]
    assert all(head.classes_.tolist() == bundle["classes"] for head in heads)
    assert bundle["channels"] == list(CHANNELS)
    assert bundle.get("input_contract_version", bundle["contract_version"]) == CONTRACT_VERSION
    assert bundle.get("input_contract_id", bundle["contract_id"]) == SequenceConfig().fingerprint
    assert bundle["validation_used_for_fit"] is False
    truth = {row["sample_id"]: row["target"]["label"].split("_")[0]
             for row in _read(pilot / "evaluation/D0/summary.json")["samples"]}
    summaries, window_rows, probes = [], [], []
    packed = []
    lineage = []
    parity = 0
    for sample_id, label in truth.items():
        baseline = pilot / "evaluation/D0/b_inputs" / sample_id
        candidate = pilot / "evaluation/N10/b_inputs" / sample_id
        inputs = {"GT": baseline / "gt_matched_track_sequence.json",
                  "D0": baseline / "a_track_sequence.json",
                  "N10": candidate / "a_track_sequence.json"}
        reference_clock = None
        for kind, path in inputs.items():
            data = _read(path)
            clock = [(p["frame_index"], p["timestamp_ms"]) for p in data["history"]]
            if reference_clock is None:
                reference_clock = clock
            if clock != reference_clock:
                raise ValueError("paired input clocks differ")
            windows, rejected = window_track(data, SequenceConfig())
            x = validate_windows(np.stack([row["X"] for row in windows]))
            if frame_scale:
                x = scale_windows(x, [row["normalization_scale"] for row in windows])
                validate_windows(x)
            if dual_scale or stack_scale:
                scales = [row["normalization_scale"] for row in windows]
                decision_margins = lambda model, values: dual_margins(model, values, scales)
            scores = decision_margins(bundle, x)
            prediction = classifier.predict(TrackSequence.model_validate(data))
            if not np.allclose(scores, prediction.window_scores, rtol=0, atol=1e-9):
                raise ValueError("production/research score mismatch")
            if prediction.label != ("drone" if scores.mean() >= 0 else "bird"):
                raise ValueError("production/research label mismatch")
            parity += 1
            aggregates = aggregation_scores(scores)
            summaries.append({"sample_id": sample_id, "truth": label, "input": kind,
                              "prediction": prediction.label, "correct": prediction.label == label,
                              "margin": float(scores.mean()), "windows": len(scores),
                              "positive_windows": int(np.sum(scores >= 0)),
                              "rejected_windows": len(rejected),
                              **{name + "_prediction": "drone" if value >= 0 else "bird"
                                 for name, value in aggregates.items()}})
            for index, (row, score) in enumerate(zip(windows, scores)):
                window_rows.append({"sample_id": sample_id, "truth": label, "input": kind,
                                    "window_index": row["window_index"],
                                    "relative_start_s": row["relative_start_s"],
                                    "absolute_start_s": row["window_start_s"],
                                    "absolute_end_s": row["window_end_s"], "margin": float(score),
                                    "prediction": "drone" if score >= 0 else "bird",
                                    "source_fps": row["source_fps"],
                                    "normalization_scale": row["normalization_scale"],
                                    "missing_fraction": row["missing_fraction"],
                                    "anti_alias_applied": row["anti_alias_applied"],
                                    "quality_flags": row["quality_flags"]})
                packed.append(row["X"])
                lineage.append(f"{sample_id}:{kind}:w{index}")
            for turns in range(4):
                rotated = rotate_channels(x, turns)
                validate_windows(rotated)
                decision = decision_margins(bundle, rotated)
                margin = float(decision.mean())
                probes.append({"sample_id": sample_id, "truth": label, "input": kind,
                               "rotation_degrees": turns * 90, "margin": margin,
                               "prediction": "drone" if margin >= 0 else "bird",
                               "windows": len(decision)})
    _csv(output / "tracks.csv", summaries)
    _csv(output / "windows.csv", window_rows)
    _csv(output / "rotation_sensitivity.csv", probes)
    np.savez_compressed(output / "inference_windows.npz", X=np.stack(packed), sample_id=np.array(lineage))
    aggregations = {name: {kind: sum(row[name + "_prediction"] == row["truth"]
                                   for row in summaries if row["input"] == kind)
                           for kind in ("GT", "D0", "N10")}
                    for name in ("mean", "median", "majority", "any_positive_exploratory")}
    flips = []
    for sample_id in truth:
        selected = [row for row in probes if row["sample_id"] == sample_id and row["input"] == "GT"]
        if len({row["prediction"] for row in selected}) > 1:
            flips.append(sample_id)
    result = {
        "scope": "paired six-object development audit, not independent performance",
        "fitting_performed": False, "production_changed": False,
        "production_parity_inputs": 0 if experimental else parity,
        "experimental_adapter_parity_inputs": parity if experimental else 0,
        "model": {key: bundle[key] for key in ("classes", "channels", "contract_version", "contract_id",
                                                 "alpha", "arm", "train_groups", "train_windows", "package_manifest_sha256")},
        "checks": {"class_sign": True, "channel_order": True, "contract_id": True,
                   "paired_frame_clocks": True, "float32_finite_and_displacement": True,
                   "original_training_npz_replay": "checked separately by retraining; not by this inference audit"},
        "aggregation_correct_counts_out_of_six": aggregations,
        "gt_rotation_label_flips": flips,
        "tracks": summaries, "rotation_probes": probes,
        "limitations": ["Same filming group contains both birds; no independent bird-source test here.",
                        "Rotation is a sensitivity probe, not verified test-time augmentation.",
                        "Any-positive is an exploratory upper response; not selected on these six cases.",
                        "Training-source overlap cannot be excluded without source inventory."],
        "source_hashes": {str(path): _hash(path) for path in (
            model_path, branch / "research/trajectory_sequence.py",
            branch / "research/minirocket_inference.py", branch / "ai_server/services/classifier.py",
            branch / "ai_server/services/sequence_model.py", Path(__file__))},
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    result = run(args.branch_root.resolve(), args.pilot.resolve(), args.output.resolve(),
                 args.model.resolve() if args.model else None)
    print(json.dumps({key: value for key, value in result.items() if key not in {"tracks", "rotation_probes", "source_hashes"}}, indent=2))
