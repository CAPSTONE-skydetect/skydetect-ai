"""Exploratory bird false alarms; source independence and tracking GT are unknown."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
# Selected from decoded images before any model scores were inspected.
ROIS = {
    "B01": (0, (226., 210., 12., 8.)),
    "B02": (0, (1480., 602., 52., 16.)),
    "B03": (0, (1577., 539., 44., 12.)),
    "B04": (0, (1180., 797., 65., 20.)),
    "B05": (100, (294., 224., 12., 13.)),
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def worker(root, sample_id, variant):
    import cv2
    cv2.setNumThreads(1)
    from ai_server.services.manual_roi_tracker import process_manual_roi_video
    from ai_server.tracking_schemas import TrackingTuning
    sample = next(s for s in json.loads((root / "inventory.json").read_text()) if s["sample_id"] == sample_id)
    start, bbox = ROIS[sample_id]
    out = root / "tracking" / variant / sample_id
    try:
        if hashlib.sha256(Path(sample["video"]).read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError("source video changed")
        result = process_manual_roi_video(sample["video"], source_video_id=f"bird_{sample_id}_{variant}",
                                          target_bbox=bbox, init_frame_index=start, max_seconds=10,
                                          stabilize=True, resize_width=1280, output_dir=out / "runs",
                                          write_overlay=True, tuning=TrackingTuning(target_scale_klt=variant == "AR1"))
        write(out / "result.json", dict(status="completed", sample_id=sample_id, variant=variant,
                                        start_frame=start, bbox=bbox, artifacts=result.artifacts,
                                        metadata=result.metadata, metrics=result.metrics))
    except Exception as error:
        write(out / "result.json", dict(status="failed", sample_id=sample_id, variant=variant, reason=str(error)))


def run(root, python):
    def one(pair):
        sample, variant = pair
        out = root / "tracking" / variant / sample
        out.mkdir(parents=True, exist_ok=True)
        path = out / "result.json"
        if not path.exists():
            try:
                with (out / "worker.log").open("w") as stream:
                    subprocess.run([python, "-m", "research.bird_false_positive_check", "worker", "--root", str(root),
                                    "--sample", sample, "--variant", variant], cwd=ROOT, check=True,
                                   stdout=stream, stderr=subprocess.STDOUT, timeout=600,
                                   env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
            except Exception as error:
                write(path, dict(status="failed", sample_id=sample, variant=variant, reason=str(error)))
        return json.loads(path.read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(one, (sample, variant)) for sample in ROIS for variant in ("D0", "AR1")]):
            result = future.result()
            print(result["sample_id"], result["variant"], result["status"], flush=True)


def classify(root, branch, model):
    sys.path.insert(0, str(branch))
    from ai_server.schemas import TrackSequence
    from ai_server.services.classifier import MiniRocketClassifier
    from minirocket_stack import StackMiniRocketClassifier
    from ard100_benchmark import BASELINE_SHA, CANDIDATE_SHA, digest
    baseline = branch / "models/minirocket_classifier.joblib"
    if digest(baseline) != BASELINE_SHA or digest(model) != CANDIDATE_SHA:
        raise ValueError("frozen C model changed")
    classifiers = dict(baseline=MiniRocketClassifier(baseline), candidate=StackMiniRocketClassifier(model))
    rows, details = [], []
    for sample in ROIS:
        for variant in ("D0", "AR1"):
            path = root / "tracking" / variant / sample / "result.json"
            tracking = json.loads(path.read_text()) if path.exists() else {}
            for name, classifier in classifiers.items():
                row = dict(sample_id=sample, tracking_variant=variant, model=name, truth="bird",
                           prediction="unavailable", margin=None, windows=0, visible_ratio=None)
                try:
                    track_path = Path(tracking["artifacts"]["track_sequence"])
                    track = TrackSequence.model_validate_json(track_path.read_text())
                    result = asdict(classifier.predict(track))
                    row.update(prediction=result["label"], margin=result["decision_score"],
                               windows=len(result.get("window_scores") or []), visible_ratio=tracking["metrics"]["visible_ratio"])
                    details.append(dict(**row, track_sha256=digest(track_path), result=result))
                except Exception as error:
                    details.append(dict(**row, reason=str(error)))
                rows.append(row)
    with (root / "classification.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write(root / "classification_details.json", details)
    groups = []
    for variant in ("D0", "AR1"):
        for model_name in classifiers:
            subset = [r for r in rows if r["tracking_variant"] == variant and r["model"] == model_name]
            groups.append(dict(tracking_variant=variant, model=model_name, files=len(subset),
                               bird_correct=sum(r["prediction"] == "bird" for r in subset),
                               drone_false_alarm=sum(r["prediction"] == "drone" for r in subset),
                               uncertain=sum(r["prediction"] == "uncertain" for r in subset),
                               unavailable=sum(r["prediction"] == "unavailable" for r in subset)))
    result = dict(label_evidence="user confirms these are bird videos; initial ROIs visually selected",
                  baseline_sha256=BASELINE_SHA, candidate_sha256=CANDIDATE_SHA,
                  candidate_adapter_sha256=digest(ROOT / "research/minirocket_stack.py"),
                  training_source_independence="unknown per user; no match against available known video hashes",
                  limitations=["not an independent test", "no temporal GT: target identity after initialization unverified",
                               "file-level observations, filming-session independence unknown",
                               "B05 has multiple birds; upper bird at frame100 selected before inference"], summaries=groups)
    write(root / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["worker", "run", "classify"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--sample", choices=list(ROIS))
    p.add_argument("--variant", choices=["D0", "AR1"])
    p.add_argument("--branch-root", type=Path)
    p.add_argument("--model", type=Path)
    a = p.parse_args()
    if a.command == "worker":
        worker(a.root.resolve(), a.sample, a.variant)
    elif a.command == "run":
        run(a.root.resolve(), sys.executable)
    else:
        classify(a.root.resolve(), a.branch_root.resolve(), a.model.resolve())
