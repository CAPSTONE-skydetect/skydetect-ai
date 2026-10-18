"""Frozen first-10-second ARD100 GT / D0 / MiniRocket evaluation.

VOC XML suffixes are one-based as in the supplied extraction script. No
post-initialization GT is passed to the tracker. Missing labels are not filled.
"""

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
import time
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA = "9024731a649b5e0a84ead3fcc45c9dc0a692ce305cea6e5ec8b7d720867a1e5a"
CANDIDATE_SHA = "5ba8fc4ac843f30c8a2a590add3f5854f6cf23093d1b8ed1f3b0731b95ec8aab"
TUNING = dict(klt_accept_conf=.4, recovery_conf=.58, update_conf=.76,
              search_radius_multiplier=2.5, exit_confirmation_frames=3,
              online_update_enabled=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def parse_voc(data):
    root = ET.fromstring(data)
    width, height = int(root.findtext("size/width")), int(root.findtext("size/height"))
    objects = root.findall("object")
    if not objects:
        return width, height, None
    if len(objects) != 1 or objects[0].findtext("name", "").lower() != "drone":
        raise ValueError("expected exactly one explicitly labeled Drone per frame")
    box = objects[0].find("bndbox")
    coords = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
    x1, y1, x2, y2 = coords
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError("bbox outside declared image")
    # VOC exporter conventions can differ by one pixel. Preserve its actual
    # coordinates rather than claiming a subpixel-accurate human reference.
    return width, height, coords


def prepare(source, out):
    import cv2
    import numpy as np

    out.mkdir(parents=True, exist_ok=False)
    records = []
    with zipfile.ZipFile(source / "annotations.zip") as archive:
        grouped = {}
        for name in archive.namelist():
            if name.endswith(".xml"):
                grouped.setdefault(Path(name).parent.name, []).append(name)
        for split in ("test", "train"):
            for video in sorted((source / f"{split}_videos").glob("*.mp4")):
                sample = dict(sample_id=video.stem, split=split, video=str(video.resolve()),
                              source_group_id=f"ARD100:{video.stem}", truth="drone")
                try:
                    cap = cv2.VideoCapture(str(video))
                    w, h, n = (int(cap.get(k)) for k in (
                        cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if min(w, h, n, fps) <= 0:
                        raise ValueError("invalid video geometry")
                    boxes = {}
                    start = None
                    for name in sorted(grouped.get(video.stem, []), key=lambda s: int(Path(s).stem.rsplit("_", 1)[1])):
                        index = int(Path(name).stem.rsplit("_", 1)[1]) - 1
                        if start is not None and index > start + int(round(10 * fps)):
                            break
                        if index < 0 or index >= n:
                            raise ValueError("annotation frame outside original video")
                        # Initialization is the first valid annotation, never a
                        # segment selected after seeing tracking or C results.
                        aw, ah, bbox = parse_voc(archive.read(name))
                        if (aw, ah) != (w, h) or index in boxes:
                            raise ValueError("annotation geometry mismatch or duplicate frame")
                        if bbox is not None:
                            boxes[index] = bbox
                            if start is None:
                                start = index
                    if not boxes:
                        raise ValueError("no initial annotated target in video")
                    start = min(boxes)
                    end = min(n - 1, start + int(round(10 * fps)))
                    if end > max(boxes):
                        # Read remaining requested labels when initialization
                        # is later than the initial scan, without inventing GT.
                        for name in grouped.get(video.stem, []):
                            index = int(Path(name).stem.rsplit("_", 1)[1]) - 1
                            if index in boxes or not start <= index <= end:
                                continue
                            aw, ah, bbox = parse_voc(archive.read(name))
                            if (aw, ah) != (w, h):
                                raise ValueError("annotation dimensions mismatch")
                            if bbox is not None:
                                boxes[index] = bbox
                    selected = {f: b for f, b in sorted(boxes.items()) if start <= f <= end}
                    xml = ET.Element("annotations")
                    task = ET.SubElement(ET.SubElement(xml, "meta"), "task")
                    ET.SubElement(task, "size").text = str(n)
                    size = ET.SubElement(task, "original_size")
                    ET.SubElement(size, "width").text = str(w)
                    ET.SubElement(size, "height").text = str(h)
                    track = ET.SubElement(xml, "track", id="0", label="drone")
                    for f, b in selected.items():
                        ET.SubElement(track, "box", frame=str(f), outside="0", occluded="0",
                                      keyframe="1", **dict(zip(("xtl", "ytl", "xbr", "ybr"), map(str, b))))
                    target = out / "inputs" / video.stem / "gt.xml"
                    target.parent.mkdir(parents=True)
                    ET.ElementTree(xml).write(target, encoding="utf-8", xml_declaration=True)
                    x1, y1, x2, y2 = boxes[start]
                    sample.update(status="ready", width=w, height=h, fps=fps, frame_count=n,
                                  init_frame_index=start, end_frame=end, max_seconds=(end-start)/fps,
                                  target_bbox=[x1, y1, x2-x1, y2-y1], gt_xml=str(target.resolve()),
                                  gt_sha256=digest(target), video_sha256=digest(video),
                                  annotated_frames=len(selected), attempted_frames=end-start+1,
                                  median_bbox_width_1280=float(np.median([b[2]-b[0] for b in selected.values()]))*min(1280/w, 1),
                                  median_bbox_height_1280=float(np.median([b[3]-b[1] for b in selected.values()]))*min(1280/w, 1))
                    if video.stem in {"phantom02", "phantom04", "phantom141", "phantom09"}:
                        panels = []
                        cap = cv2.VideoCapture(str(video))
                        for f in list(selected)[::max(1, len(selected)//3)][:3]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                            ok, frame = cap.read()
                            if not ok:
                                raise ValueError("alignment preview decode failed")
                            x1, y1, x2, y2 = map(int, selected[f])
                            cx, cy = (x1+x2)//2, (y1+y2)//2
                            crop = frame[max(0,cy-55):min(h,cy+55), max(0,cx-80):min(w,cx+80)].copy()
                            cv2.rectangle(crop, (x1-max(0,cx-80), y1-max(0,cy-55)),
                                          (x2-max(0,cx-80), y2-max(0,cy-55)), (0,255,0), 1)
                            crop = cv2.resize(crop, (640,440), interpolation=cv2.INTER_NEAREST)
                            cv2.putText(crop, f"{video.stem} frame={f} XML={f+1:04}", (12,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
                            panels.append(crop)
                        cap.release()
                        cv2.imwrite(str(out / f"alignment_{video.stem}.jpg"), np.concatenate(panels, axis=1))
                except Exception as error:
                    sample.update(status="preflight_failed", reason=str(error))
                records.append(sample)
                print(sample["sample_id"], sample["status"], flush=True)
    payload = dict(protocol="ARD100-frozen-first10s-v1", source=str(source.resolve()),
                   annotations_sha256=digest(source / "annotations.zip"),
                   source_independence="user attests no overlap with previous training sources",
                   gt_numbering="XML suffix minus 1; supplied extractor starts at 0001",
                   roi_policy="first valid full-object GT bbox, initialization only",
                   d0_tuning=TUNING, stabilize=True, resize_width=1280,
                   duration_seconds=10, samples=records)
    write(out / "protocol.json", payload)
    return payload


def worker(protocol, sample_id, out):
    import cv2
    cv2.setNumThreads(1)
    from ai_server.services.manual_roi_tracker import process_manual_roi_video
    from ai_server.tracking_schemas import TrackingTuning
    from research.tracking_evaluation import evaluate_manifest, write_comparison_video
    from research.tracking_feature_comparison import compare_b_features

    config = json.loads(protocol.read_text())
    sample = next(s for s in config["samples"] if s["sample_id"] == sample_id)
    folder = out / "samples" / sample_id
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(sample_id=sample_id, split=sample["split"], status="failed")
    start = time.monotonic()
    try:
        if digest(sample["video"]) != sample["video_sha256"] or digest(sample["gt_xml"]) != sample["gt_sha256"]:
            raise ValueError("frozen input hash changed")
        overlay = sample_id in {"phantom02", "phantom04", "phantom141", "phantom09"}
        tracking = process_manual_roi_video(
            sample["video"], source_video_id=f"ARD100_{sample_id}_{config.get('variant', 'D0')}",
            target_bbox=tuple(sample["target_bbox"]), init_frame_index=sample["init_frame_index"],
            max_seconds=sample["max_seconds"], stabilize=config["stabilize"],
            tuning=TrackingTuning(**config["d0_tuning"]), output_dir=folder / "runs",
            resize_width=config["resize_width"], write_overlay=overlay)
        artifacts = tracking.artifacts
        write(folder / "tracking_result.json", dict(artifacts=artifacts, metrics=tracking.metrics,
                                                     metadata=tracking.metadata))
        manifest = dict(sample_id=sample_id, ground_truth=dict(path=sample["gt_xml"], track_ids=[0]),
                        prediction=dict(trajectory=artifacts["trajectory"], metadata=artifacts["metadata"]),
                        alignment=dict(prediction_frame_offset=0, source_is_trimmed=False),
                        b_comparison=dict(track_sequence=artifacts["track_sequence"],
                                          debug_observations=artifacts["debug_observations"], source_video=sample["video"]))
        write(folder / "sample.json", manifest)
        report = evaluate_manifest(folder / "sample.json")
        write(folder / "tracking_evaluation.json", report)
        result.update(tracking_evaluation=report, tracking_artifacts=artifacts)
        comparison = compare_b_features(report, manifest=manifest, manifest_root=folder,
                                       output_dir=folder / "b_inputs", replay_cmc=True)
        result["b_comparison"] = comparison
        if comparison["status"] == "not_comparable":
            result["partial_exports"] = export_partial_gt(sample, tracking, folder)
        if overlay:
            write_comparison_video(cvat_xml=sample["gt_xml"], trajectory_csv=artifacts["trajectory"],
                                   metadata_json=artifacts["metadata"], overlay_video=artifacts["overlay"],
                                   output_video=folder / "comparison.mp4", sample_id=sample_id, track_ids=(0,))
        result["status"] = "completed" if comparison["status"] != "not_comparable" else "paired_export_unavailable"
        if comparison["status"] == "not_comparable":
            result["reason"] = "; ".join(comparison.get("reasons", []))
    except Exception as error:
        import traceback
        result.update(reason=str(error), traceback=traceback.format_exc())
    result["elapsed_seconds"] = time.monotonic() - start
    write(folder / "result.json", result)
    return result


def export_partial_gt(sample, tracking, folder):
    """Export each support independently; never clip an out-of-range GT center."""
    from ai_server.schemas import TrackSequence, TrackPoint
    from research.tracking_evaluation import load_cvat_track, load_video_geometry, _transform_point
    from ai_server.services.tracking_video_io import transform_point
    geometry = load_video_geometry(tracking.artifacts["metadata"])
    gt = load_cvat_track(sample["gt_xml"], fps=geometry.fps, track_ids=(0,))
    transforms = folder / "b_inputs/camera_transforms.json"
    if not transforms.exists():
        return dict(reason="verified CMC matrices unavailable")
    payload = json.loads(transforms.read_text())
    if payload["verification"]["max_residual_px"] > payload["verification"]["tolerance_px"]:
        return dict(reason="CMC replay verification failed")
    matrices = {row["frame_index"]: row["current_to_reference"] for row in payload["frames"]}
    a_by_frame = {p.frame_index: p for p in tracking.track.history}
    gt_frames = {p.frame_index for p in gt.observations}
    base = tracking.track.model_dump(exclude={"history", "quality"})
    import numpy as np
    statuses = {}
    matched_a = [p for p in tracking.track.history if p.frame_index in gt_frames]
    if matched_a:
        write(folder / "b_inputs/a_track_sequence.json", TrackSequence(**base, history=matched_a).model_dump(mode="json"))
    for name, matched in (("gt_full", False), ("gt_matched", True)):
        points, invalid = [], []
        for p in gt.observations:
            if p.frame_index not in matrices or (matched and p.frame_index not in a_by_frame):
                continue
            x, y = _transform_point(geometry.original_to_processed, p.cx, p.cy)
            x, y = transform_point(np.asarray(matrices[p.frame_index], dtype=np.float32), x, y)
            cx, cy = x / geometry.processed_width, y / geometry.processed_height
            if not 0 <= cx <= 1 or not 0 <= cy <= 1:
                invalid.append(p.frame_index)
                continue
            points.append(TrackPoint(frame_index=p.frame_index,
                                     timestamp_ms=a_by_frame[p.frame_index].timestamp_ms if p.frame_index in a_by_frame else round(p.frame_index / geometry.fps * 1000),
                                     cx=cx, cy=cy, w=p.width/geometry.source_width,
                                     h=p.height/geometry.source_height, conf=1.0))
        if invalid or not points:
            statuses[name] = dict(status="unavailable", outside_frames=invalid,
                                  reason="GT outside normalized bounds" if invalid else "no matched points")
        else:
            write(folder / f"b_inputs/{name}_track_sequence.json", TrackSequence(**base, history=points).model_dump(mode="json"))
            statuses[name] = dict(status="exported", points=len(points))
    write(folder / "b_inputs/partial_export_status.json", statuses)
    return statuses


def batch(protocol, out, python, jobs, limit):
    config = json.loads(protocol.read_text())
    samples = [s for s in config["samples"] if s["status"] == "ready"]
    if limit:
        samples = samples[:limit]
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for package in ("ai_server", "research")
                   for p in sorted((ROOT / package).rglob("*.py"))}
    lock = out / "execution_sources.json"
    if lock.exists() and json.loads(lock.read_text())["source_hashes"] != code_hashes:
        raise ValueError("source changed since batch began; use a new output directory")
    write(lock, dict(source_hashes=code_hashes, protocol_sha256=digest(protocol), python=python))

    def one(sample):
        folder = out / "samples" / sample["sample_id"]
        previous = folder / "result.json"
        if previous.exists():
            return json.loads(previous.read_text())
        folder.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
        try:
            with (folder / "worker.log").open("w") as log:
                subprocess.run([python, "-m", "research.ard100_benchmark", "worker", "--protocol", str(protocol),
                                "--output", str(out), "--sample-id", sample["sample_id"]],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            return json.loads(previous.read_text())
        except Exception as error:
            result = dict(sample_id=sample["sample_id"], split=sample["split"], status="failed", reason=str(error))
            write(previous, result)
            return result

    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for future in as_completed([pool.submit(one, s) for s in samples]):
            row = future.result()
            results.append(row)
            print(f"{len(results)}/{len(samples)} {row['sample_id']} {row['status']} {row.get('reason','')}", flush=True)
            write(out / "progress.json", dict(completed=len(results), requested=len(samples), results=results))


def classify(protocol, out, branch, candidate):
    sys.path.insert(0, str(branch))
    from ai_server.schemas import TrackSequence
    from ai_server.services.classifier import MiniRocketClassifier
    from minirocket_stack import StackMiniRocketClassifier
    baseline = branch / "models/minirocket_classifier.joblib"
    if digest(baseline) != BASELINE_SHA or digest(candidate) != CANDIDATE_SHA:
        raise ValueError("frozen model changed")
    models = dict(baseline=MiniRocketClassifier(baseline), candidate=StackMiniRocketClassifier(candidate))
    config = json.loads(protocol.read_text())
    rows, details = [], []
    for s in config["samples"]:
        folder = out / "samples" / s["sample_id"]
        result_path = folder / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        for kind, filename in (("GT_full", "gt_full_track_sequence.json"),
                               ("GT_matched", "gt_matched_track_sequence.json"),
                               ("D0_matched", "a_track_sequence.json"),
                               ("D0", None)):
            track_path = folder / "b_inputs" / filename if filename else folder / "unavailable.json"
            if kind == "D0":
                saved = result.get("tracking_artifacts", {}).get("track_sequence")
                track_path = Path(saved) if saved else folder / "unavailable.json"
            for name, model in models.items():
                row = dict(sample_id=s["sample_id"], split=s["split"], input=kind, model=name,
                           truth="drone", prediction="unavailable", margin=None, windows=0,
                           bbox_width_1280=s.get("median_bbox_width_1280"),
                           status=result.get("status", s.get("status")), reason=result.get("reason", s.get("reason", "")))
                try:
                    if not track_path.exists():
                        raise ValueError(row["reason"] or "paired GT/A export unavailable")
                    track = TrackSequence.model_validate_json(track_path.read_text())
                    pred = asdict(model.predict(track))
                    row.update(prediction=pred["label"], margin=pred["decision_score"],
                               windows=len(pred.get("window_scores") or []), status="inferred", reason=pred.get("abstain_reason") or "")
                    details.append(dict(sample_id=s["sample_id"], input=kind, model=name,
                                        track_sha256=digest(track_path), result=pred))
                except Exception as error:
                    row["reason"] = str(error)
                rows.append(row)
        print(s["sample_id"], "classified", flush=True)
    with (out / "classification.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write(out / "classification_details.json", details)
    summary = []
    for split in ("test", "train"):
        for kind in ("GT_full", "GT_matched", "D0_matched", "D0"):
            for model in models:
                group = [r for r in rows if (r["split"], r["input"], r["model"]) == (split, kind, model)]
                n = len(group)
                positive = sum(r["prediction"] == "drone" for r in group)
                negative = sum(r["prediction"] == "bird" for r in group)
                uncertain = sum(r["prediction"] == "uncertain" for r in group)
                summary.append(dict(split=split, input=kind, model=model, requested=n,
                                    correct_drone=positive, wrong_bird=negative, uncertain=uncertain,
                                    unavailable=n-positive-negative-uncertain,
                                    end_to_end_drone_recall=positive/n if n else None))
    write(out / "summary.json", dict(protocol_sha256=digest(protocol), baseline_sha256=BASELINE_SHA,
                                     candidate_sha256=CANDIDATE_SHA, summaries=summary,
                                     limitations=["drone-only: no bird false-positive rate or balanced accuracy",
                                                  "first 10 seconds from initial annotated bbox, not full videos",
                                                  "source independence attested by user, filming-session independence unverified",
                                                  "VOC coordinates preserved; subpixel annotation accuracy not assumed"]))
    lines = ["# ARD100 frozen comparison", "", "First 10 seconds; initial GT bbox only; no retraining.", "",
             "| Split | Input | Model | Drone correct | Bird wrong | Uncertain | Unavailable | Total |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['split']} | {s['input']} | {s['model']} | {s['correct_drone']} | {s['wrong_bird']} | {s['uncertain']} | {s['unavailable']} | {s['requested']} |")
    lines += ["", "All requested videos remain in denominators. Matched GT follows A's observed frames; full GT does not.",
              "Drone-only results cannot measure bird false alarms. These are fixed clips, not full-sequence scores."]
    (out / "report.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "worker", "run", "classify"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--branch-root", type=Path)
    parser.add_argument("--candidate", type=Path)
    a = parser.parse_args()
    if a.command == "prepare":
        prepare(a.source.resolve(), a.output.resolve())
    elif a.command == "worker":
        worker(a.protocol.resolve(), a.sample_id, a.output.resolve())
    elif a.command == "run":
        batch(a.protocol.resolve(), a.output.resolve(), a.python, a.jobs, a.limit)
    else:
        print(json.dumps(classify(a.protocol.resolve(), a.output.resolve(), a.branch_root.resolve(), a.candidate.resolve()), indent=2))
