"""Compare two frozen classifiers on paired GT/A tracks without fitting."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(manifest_path, output):
    manifest_path = manifest_path.resolve()
    spec = json.loads(manifest_path.read_text())
    root = manifest_path.parent

    def resolve(value):
        return (root / value).resolve()

    branch = resolve(spec["branch_root"])
    sys.path.insert(0, str(branch))
    from ai_server.schemas import TrackSequence
    from ai_server.services.classifier import MiniRocketClassifier
    from minirocket_stack import StackMiniRocketClassifier

    classifiers = {}
    for name, constructor in (("baseline", MiniRocketClassifier),
                              ("candidate", StackMiniRocketClassifier)):
        entry = spec["models"][name]
        path = resolve(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"frozen model hash mismatch: {name}")
        classifiers[name] = constructor(path)

    prepared = []
    seen = set()
    for sample in spec["samples"]:
        if sample["sample_id"] in seen:
            raise ValueError("duplicate sample_id")
        seen.add(sample["sample_id"])
        if sample["truth"] not in ("bird", "drone") or not sample["source_group_id"]:
            raise ValueError("explicit truth and source group are required")
        tracks = {}
        for kind in ("GT", "D0"):
            entry = sample["inputs"][kind]
            path = resolve(entry["path"])
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"input hash mismatch: {sample['sample_id']} {kind}")
            tracks[kind] = TrackSequence.model_validate_json(path.read_text(encoding="utf-8-sig"))
        times = lambda track: [(p.frame_index, p.timestamp_ms) for p in track.history]
        if times(tracks["GT"]) != times(tracks["D0"]):
            raise ValueError("GT/A must use identical observed frames and timestamps")
        for field in ("processed_width", "processed_height", "stabilization"):
            if getattr(tracks["GT"], field) != getattr(tracks["D0"], field):
                raise ValueError(f"GT/A coordinate metadata mismatch: {field}")
        prepared.append((sample, tracks))

    output.mkdir(parents=True, exist_ok=False)
    rows, raw = [], []
    for sample, tracks in prepared:
        for kind, track in tracks.items():
            for name, classifier in classifiers.items():
                result = asdict(classifier.predict(track))
                row = dict(sample_id=sample["sample_id"], source_group_id=sample["source_group_id"],
                           truth=sample["truth"], input=kind, model=name,
                           prediction=result["label"], margin=result["decision_score"],
                           correct=result["label"] == sample["truth"],
                           windows=len(result.get("window_scores") or []))
                rows.append(row)
                raw.append(dict(**row, details=result))
                print(json.dumps(row), flush=True)
    (output / "manifest.frozen.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    (output / "results.json").write_text(json.dumps(dict(
        cohort=spec["cohort"], limitations=spec.get("limitations", []),
        manifest_sha256=sha256(manifest_path),
        runner_sha256=sha256(Path(__file__)),
        score_is_probability=False, model_scores_have_different_scales=True,
        predictions=raw), ensure_ascii=False, indent=2))
    if rows:
        with (output / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.output)
