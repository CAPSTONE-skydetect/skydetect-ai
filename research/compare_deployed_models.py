"""Compare deployed RF and MiniRocket branches on identical TrackSequence inputs.

The orchestrator intentionally calls each branch with its own Python runtime. This
keeps the comparison tied to the collaborator's checked-in preprocessing and model
artifact instead of reimplementing either classifier in the experiment branch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


INPUT_VARIANTS = ("a", "gt_matched", "gt_full")
MODEL_NAMES = ("rf", "minirocket")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_label(label: str) -> str:
    """Collapse CVAT instance names such as bird_1 to the classifier label."""
    value = label.strip().lower()
    if value.startswith("bird"):
        return "bird"
    if value.startswith("drone"):
        return "drone"
    raise ValueError(f"Unsupported ground-truth label: {label!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run_json(command: list[str], *, cwd: Path) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {
            "status": "runtime_error",
            "prediction": None,
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "status": "runtime_error",
            "prediction": None,
            "error": f"Invalid JSON output: {exc}: {result.stdout[-1000:]}",
        }


def run_model_batch(
    inputs: list[Path],
    *,
    model: str,
    branch_root: Path,
    python: Path,
    script_path: Path,
) -> list[dict[str, Any]]:
    result = _run_json(
        [
            str(python),
            str(script_path),
            "batch-worker",
            "--model",
            model,
            "--branch-root",
            str(branch_root),
            "--inputs",
            *[str(path) for path in inputs],
        ],
        cwd=branch_root,
    )
    if isinstance(result, list):
        return result
    error = result.get("error", "batch worker did not return a list")
    return [{"status": "runtime_error", "prediction": None, "error": error}] * len(inputs)


def rf_worker(branch_root: Path, input_path: Path) -> int:
    """Run the RF branch's production feature and prediction services."""
    sys.path.insert(0, str(branch_root.resolve()))
    from ai_server.schemas import TrackSequence
    from ai_server.services.feature_core import build_feature_vector
    from ai_server.services.prediction import classify_feature_vector

    track = TrackSequence.model_validate(_read_json(input_path))
    extraction = build_feature_vector(track)
    prediction = classify_feature_vector(extraction.feature_vector)
    payload = prediction.model_dump(mode="json")
    payload.update(
        {
            "status": "predicted" if prediction.label != "uncertain" else "uncertain",
            "prediction": prediction.label,
            "feature_status": extraction.feature_vector.feature_status,
            "feature_reasons": extraction.reasons,
            "features": (
                extraction.feature_vector.features.model_dump(mode="json")
                if extraction.feature_vector.features is not None
                else None
            ),
        }
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def batch_worker(model: str, branch_root: Path, input_paths: list[Path]) -> int:
    sys.path.insert(0, str(branch_root.resolve()))
    results: list[dict[str, Any]] = []
    if model == "rf":
        from ai_server.schemas import TrackSequence
        from ai_server.services.feature_core import build_feature_vector
        from ai_server.services.prediction import classify_feature_vector

        for input_path in input_paths:
            try:
                track = TrackSequence.model_validate(_read_json(input_path))
                extraction = build_feature_vector(track)
                prediction = classify_feature_vector(extraction.feature_vector)
                payload = prediction.model_dump(mode="json")
                payload.update(
                    {
                        "status": (
                            "predicted" if prediction.label != "uncertain" else "uncertain"
                        ),
                        "prediction": prediction.label,
                        "feature_status": extraction.feature_vector.feature_status,
                        "feature_reasons": extraction.reasons,
                        "features": (
                            extraction.feature_vector.features.model_dump(mode="json")
                            if extraction.feature_vector.features is not None
                            else None
                        ),
                    }
                )
                results.append(payload)
            except Exception as exc:  # keep the rest of the evaluation running
                results.append(
                    {"status": "runtime_error", "prediction": None, "error": repr(exc)}
                )
    else:
        import numpy as np
        from ai_server.schemas import TrackSequence
        from ai_server.services.classifier import MiniRocketClassifier
        from research.minirocket_inference import predict_track

        classifier = MiniRocketClassifier(branch_root / "models/minirocket_classifier.joblib")
        for input_path in input_paths:
            try:
                track = TrackSequence.model_validate(_read_json(input_path))
                prediction = classifier.predict(track)
                reference = predict_track(classifier.model, track.model_dump(mode="json"))
                expected = reference.get("window_margins", [])
                score_match = len(expected) == len(prediction.window_scores) and np.allclose(
                    expected, prediction.window_scores, rtol=0, atol=1e-9,
                )
                label_match = prediction.label == (reference.get("prediction") or "uncertain")
                if not score_match or not label_match:
                    raise ValueError("production/research MiniRocket predictions disagree")
                results.append({
                    "status": "predicted" if prediction.label != "uncertain" else "uncertain",
                    "prediction": prediction.label, "margin": prediction.decision_score,
                    "windows": len(prediction.window_scores), "window_margins": prediction.window_scores,
                    "window_starts_s": prediction.window_starts_s,
                    "rejections": prediction.window_rejections, "abstain_reason": prediction.abstain_reason,
                    "score_is_probability": False,
                    "implementation": "production_MiniRocketClassifier",
                    "research_parity_verified": True,
                })
            except Exception as exc:  # keep the rest of the evaluation running
                results.append(
                    {"status": "runtime_error", "prediction": None, "error": repr(exc)}
                )
    print(json.dumps(results, ensure_ascii=False))
    return 0


def load_truth(summary_path: Path) -> dict[str, str]:
    summary = _read_json(summary_path)
    return {
        sample["sample_id"]: normalize_label(sample["target"]["label"])
        for sample in summary["samples"]
    }


def discover_inputs(input_root: Path, truth: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, label in truth.items():
        sample_root = input_root / sample_id
        for variant in INPUT_VARIANTS:
            input_path = (sample_root / f"{variant}_track_sequence.json").resolve()
            if not input_path.exists():
                raise FileNotFoundError(input_path)
            track = _read_json(input_path)
            rows.append(
                {
                    "sample_id": sample_id,
                    "truth": label,
                    "variant": variant,
                    "input_path": input_path,
                    "point_count": len(track["history"]),
                }
            )
    return rows


def _flatten_result(base: dict[str, Any], model: str, result: dict[str, Any]) -> dict[str, Any]:
    prediction = result.get("prediction") or result.get("label")
    row = {
        "sample_id": base["sample_id"],
        "truth": base["truth"],
        "variant": base["variant"],
        "point_count": base["point_count"],
        "model": model,
        "status": result.get("status", "unknown"),
        "prediction": prediction,
        "correct": prediction == base["truth"],
        "confidence": result.get("confidence"),
        "margin": result.get("margin"),
        "windows": result.get("windows"),
        "reject_reason": (result.get("rule_filter") or {}).get("reject_reason"),
        "window_rejections": len(result.get("rejections") or []),
        "error": result.get("error"),
        "details": result,
    }
    return row


def _score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    counts = Counter(row["prediction"] or "no_prediction" for row in rows)
    per_class: dict[str, Any] = {}
    for label in ("bird", "drone"):
        class_rows = [row for row in rows if row["truth"] == label]
        class_correct = sum(bool(row["correct"]) for row in class_rows)
        per_class[label] = {
            "count": len(class_rows),
            "correct": class_correct,
            "recall": class_correct / len(class_rows) if class_rows else None,
        }
    recalls = [value["recall"] for value in per_class.values() if value["recall"] is not None]
    return {
        "count": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
        "prediction_counts": dict(sorted(counts.items())),
        "per_class": per_class,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for model in MODEL_NAMES:
        for variant in INPUT_VARIANTS:
            selected = [
                row for row in rows if row["model"] == model and row["variant"] == variant
            ]
            by_condition[f"{model}:{variant}"] = _score_rows(selected)

    transitions: dict[str, list[dict[str, Any]]] = {model: [] for model in MODEL_NAMES}
    for model in MODEL_NAMES:
        model_rows = [row for row in rows if row["model"] == model]
        for sample_id in sorted({row["sample_id"] for row in model_rows}):
            indexed = {
                row["variant"]: row
                for row in model_rows
                if row["sample_id"] == sample_id
            }
            a_row = indexed["a"]
            gt_row = indexed["gt_matched"]
            transitions[model].append(
                {
                    "sample_id": sample_id,
                    "truth": a_row["truth"],
                    "gt_matched_prediction": gt_row["prediction"],
                    "a_prediction": a_row["prediction"],
                    "prediction_changed": gt_row["prediction"] != a_row["prediction"],
                    "gt_correct_a_wrong": bool(gt_row["correct"] and not a_row["correct"]),
                    "gt_wrong_a_correct": bool(not gt_row["correct"] and a_row["correct"]),
                    "rf_confidence_delta": (
                        a_row["confidence"] - gt_row["confidence"]
                        if model == "rf"
                        and a_row["confidence"] is not None
                        and gt_row["confidence"] is not None
                        else None
                    ),
                    "minirocket_margin_delta": (
                        a_row["margin"] - gt_row["margin"]
                        if model == "minirocket"
                        and a_row["margin"] is not None
                        and gt_row["margin"] is not None
                        else None
                    ),
                }
            )
    return {"by_condition": by_condition, "gt_matched_to_a": transitions}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "truth",
        "variant",
        "point_count",
        "model",
        "status",
        "prediction",
        "correct",
        "confidence",
        "margin",
        "windows",
        "reject_reason",
        "window_rejections",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# RF / MiniRocket deployed-branch comparison",
        "",
        "> This is a six-object pilot case study, not an independent final benchmark.",
        "> RF confidence is a probability; MiniRocket margin is a signed decision score.",
        "",
        "## Aggregate",
        "",
        "| Model | Input | Correct | Accuracy | Balanced accuracy | Predictions |",
        "|---|---|---:|---:|---:|---|",
    ]
    for model in MODEL_NAMES:
        for variant in INPUT_VARIANTS:
            score = summary["by_condition"][f"{model}:{variant}"]
            lines.append(
                f"| {model} | {variant} | {score['correct']}/{score['count']} | "
                f"{score['accuracy']:.1%} | {score['balanced_accuracy']:.1%} | "
                f"{json.dumps(score['prediction_counts'], ensure_ascii=False)} |"
            )

    lines.extend(
        [
            "",
            "## Per-object comparison",
            "",
            "| Sample | Truth | Input | RF | RF confidence | MiniRocket | MR margin | MR windows |",
            "|---|---|---|---|---:|---|---:|---:|",
        ]
    )
    index = {(row["sample_id"], row["variant"], row["model"]): row for row in rows}
    for sample_id in sorted({row["sample_id"] for row in rows}):
        for variant in INPUT_VARIANTS:
            rf = index[(sample_id, variant, "rf")]
            mr = index[(sample_id, variant, "minirocket")]
            confidence = "-" if rf["confidence"] is None else f"{rf['confidence']:.4f}"
            margin = "-" if mr["margin"] is None else f"{mr['margin']:.4f}"
            windows = "-" if mr["windows"] is None else str(mr["windows"])
            lines.append(
                f"| {sample_id} | {rf['truth']} | {variant} | {rf['prediction']} | "
                f"{confidence} | {mr['prediction']} | {margin} | {windows} |"
            )

    lines.extend(["", "## GT-matched to A transitions", ""])
    for model in MODEL_NAMES:
        lines.extend([f"### {model}", ""])
        for item in summary["gt_matched_to_a"][model]:
            marker = "changed" if item["prediction_changed"] else "same"
            lines.append(
                f"- `{item['sample_id']}`: GT `{item['gt_matched_prediction']}` -> "
                f"A `{item['a_prediction']}` ({marker})"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def orchestrate(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    truth = load_truth(args.summary)
    inputs = discover_inputs(args.input_root, truth)
    script_path = Path(__file__).resolve()
    rows: list[dict[str, Any]] = []

    input_paths = [base["input_path"] for base in inputs]
    rf_results = run_model_batch(
        input_paths,
        model="rf",
        branch_root=args.rf_root,
        python=args.rf_python,
        script_path=script_path,
    )
    mr_results = run_model_batch(
        input_paths,
        model="minirocket",
        branch_root=args.minirocket_root,
        python=args.minirocket_python,
        script_path=script_path,
    )
    for base, rf_result, mr_result in zip(inputs, rf_results, mr_results, strict=True):
        rows.append(_flatten_result(base, "rf", rf_result))
        rows.append(_flatten_result(base, "minirocket", mr_result))
        print(f"evaluated {base['sample_id']} / {base['variant']}", flush=True)

    provenance = {
        "rf": {
            "revision": git_revision(args.rf_root),
            "model_sha256": sha256(args.rf_root / "models/rf_classifier.pkl"),
        },
        "minirocket": {
            "revision": git_revision(args.minirocket_root),
            "model_sha256": sha256(
                args.minirocket_root / "models/minirocket_classifier.joblib"
            ),
        },
        "input_root": str(args.input_root.resolve()),
        "ground_truth_summary": str(args.summary.resolve()),
    }
    summary = summarize(rows)
    summary["provenance"] = provenance
    _write_json(output / "raw_results.json", rows)
    _write_json(output / "summary.json", summary)
    write_csv(output / "results.csv", rows)
    write_report(output / "report.md", rows, summary)
    print(f"wrote {output / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--input-root", type=Path, required=True)
    compare.add_argument("--summary", type=Path, required=True)
    compare.add_argument("--rf-root", type=Path, required=True)
    compare.add_argument("--rf-python", type=Path, required=True)
    compare.add_argument("--minirocket-root", type=Path, required=True)
    compare.add_argument("--minirocket-python", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    worker = subparsers.add_parser("rf-worker")
    worker.add_argument("--branch-root", type=Path, required=True)
    worker.add_argument("--input", type=Path, required=True)
    batch = subparsers.add_parser("batch-worker")
    batch.add_argument("--model", choices=MODEL_NAMES, required=True)
    batch.add_argument("--branch-root", type=Path, required=True)
    batch.add_argument("--inputs", type=Path, nargs="+", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "rf-worker":
        return rf_worker(args.branch_root, args.input)
    if args.command == "batch-worker":
        return batch_worker(args.model, args.branch_root, args.inputs)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
