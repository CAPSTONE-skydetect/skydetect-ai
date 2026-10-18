"""Train and evaluate a local YOLO re-acquisition candidate on frozen ARD100.

This is a research-only experiment. The fixed development videos are excluded
from training and confidence calibration; D0 is retained as the production-like
baseline and is never modified by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from html import escape
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from research.ard100_sot_probe import digest
from research.tracking_evaluation import evaluate_files
from ai_server.services.tracking_video_io import iter_tracking_video_frames


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/trajectory_experiments/ard100-frozen-20261018-v3"
OUTPUT_DEFAULT = ROOT / "artifacts/trajectory_experiments/ard100-yolo-local-reacquisition-20261018-v2"
FRAME_WIDTH = 1280
CROP_SIDE = 192
IMAGE_SIZE = 384
BOXES_PER_VIDEO = 24
EPOCHS = 12
SEED = 20261018


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def _read_boxes(path: Path, end_frame: int) -> list[dict]:
    root = ET.parse(path).getroot()
    rows = []
    for track in root.findall("./track"):
        if track.attrib.get("label", "").lower() != "drone":
            continue
        for box in track.findall("./box"):
            if box.attrib.get("outside", "0").lower() in {"1", "true"}:
                continue
            frame_index = int(box.attrib["frame"])
            if frame_index > end_frame:
                continue
            rows.append({
                "frame_index": frame_index,
                "x1": float(box.attrib["xtl"]),
                "y1": float(box.attrib["ytl"]),
                "x2": float(box.attrib["xbr"]),
                "y2": float(box.attrib["ybr"]),
            })
    return sorted(rows, key=lambda row: row["frame_index"])


def _selected_dev_samples(protocol: dict) -> list[dict]:
    ready = [row for row in protocol["samples"] if row["split"] == "train" and row["status"] == "ready"]
    return ready[::max(1, len(ready) // 12)][:12]


def _crop(frame: np.ndarray, cx: float, cy: float, side: int = CROP_SIDE) -> tuple[np.ndarray, int, int]:
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    right, bottom = left + side, top + side
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - frame.shape[1]), max(0, bottom - frame.shape[0])
    if pad_left or pad_top or pad_right or pad_bottom:
        frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101,
        )
        left += pad_left
        top += pad_top
    return frame[top:top + side, left:left + side].copy(), left - pad_left, top - pad_top


def _label_line(box: dict, left: int, top: int) -> str | None:
    x1 = max(0.0, box["x1"] - left)
    y1 = max(0.0, box["y1"] - top)
    x2 = min(float(CROP_SIDE), box["x2"] - left)
    y2 = min(float(CROP_SIDE), box["y2"] - top)
    width, height = x2 - x1, y2 - y1
    if width < 2 or height < 2:
        return None
    return f"0 {(x1 + x2) / (2 * CROP_SIDE):.6f} {(y1 + y2) / (2 * CROP_SIDE):.6f} {width / CROP_SIDE:.6f} {height / CROP_SIDE:.6f}"


def _write_patch(
    image_dir: Path,
    label_dir: Path,
    name: str,
    patch: np.ndarray,
    label: str | None,
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_dir / f"{name}.jpg"), patch, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"could not write training patch: {name}")
    (label_dir / f"{name}.txt").write_text(f"{label}\n" if label else "", encoding="utf-8")


def _generate_dataset(protocol: dict, dev_ids: set[str], output: Path) -> dict:
    pool = [
        row for row in protocol["samples"]
        if row["split"] == "train" and row["status"] == "ready" and row["sample_id"] not in dev_ids
    ]
    ordered = sorted(pool, key=lambda row: row["sample_id"])
    validation_ids = {row["sample_id"] for row in ordered[::5]}
    train_samples = [row for row in ordered if row["sample_id"] not in validation_ids]
    validation_samples = [row for row in ordered if row["sample_id"] in validation_ids]
    rng = random.Random(SEED)
    counts = {"train": {"positive": 0, "negative": 0}, "val": {"positive": 0, "negative": 0}}
    used = {"train": [], "val": []}
    data_root = output / "dataset"

    for split, samples in (("train", train_samples), ("val", validation_samples)):
        image_dir, label_dir = data_root / "images" / split, data_root / "labels" / split
        for sample in samples:
            boxes = _read_boxes(Path(sample["gt_xml"]), int(sample["end_frame"]))
            if not boxes:
                continue
            indices = np.linspace(0, len(boxes) - 1, min(BOXES_PER_VIDEO, len(boxes)), dtype=int)
            selected = [boxes[i] for i in sorted(set(indices.tolist()))]
            by_frame = {box["frame_index"]: box for box in selected}
            scale = FRAME_WIDTH / float(sample["width"])
            capture = cv2.VideoCapture(sample["video"])
            if not capture.isOpened():
                raise RuntimeError(f"could not open {sample['sample_id']}")
            frame_index = 0
            positives = negatives = 0
            try:
                while frame_index <= max(by_frame):
                    ok, source_frame = capture.read()
                    if not ok:
                        break
                    box = by_frame.get(frame_index)
                    if box is not None:
                        height = int(round(sample["height"] * scale))
                        frame = cv2.resize(source_frame, (FRAME_WIDTH, height), interpolation=cv2.INTER_AREA)
                        scaled = {key: value * scale for key, value in box.items() if key in {"x1", "y1", "x2", "y2"}}
                        cx, cy = (scaled["x1"] + scaled["x2"]) / 2, (scaled["y1"] + scaled["y2"]) / 2
                        dx = rng.uniform(-0.24 * CROP_SIDE, 0.24 * CROP_SIDE)
                        dy = rng.uniform(-0.24 * CROP_SIDE, 0.24 * CROP_SIDE)
                        patch, left, top = _crop(frame, cx + dx, cy + dy)
                        label = _label_line(scaled, left, top)
                        if label:
                            stem = f"{sample['sample_id']}_{frame_index:06d}_pos"
                            _write_patch(image_dir, label_dir, stem, patch, label)
                            positives += 1

                        angle = rng.uniform(0, 2 * math.pi)
                        distance = rng.uniform(0.72, 1.25) * CROP_SIDE
                        neg_cx, neg_cy = cx + math.cos(angle) * distance, cy + math.sin(angle) * distance
                        negative, _, _ = _crop(frame, neg_cx, neg_cy)
                        neg_stem = f"{sample['sample_id']}_{frame_index:06d}_neg"
                        _write_patch(image_dir, label_dir, neg_stem, negative, None)
                        negatives += 1
                    frame_index += 1
            finally:
                capture.release()
            counts[split]["positive"] += positives
            counts[split]["negative"] += negatives
            used[split].append(sample["sample_id"])
            print(f"prepared {split} {sample['sample_id']}: +{positives} / -{negatives}", flush=True)

    yaml_path = data_root / "data.yaml"
    yaml_path.write_text(
        f"path: {data_root.resolve()}\ntrain: images/train\nval: images/val\nnames: [drone]\n",
        encoding="utf-8",
    )
    if min(counts["train"]["positive"], counts["val"]["positive"]) == 0:
        raise RuntimeError(f"empty YOLO split: {counts}")
    return {
        "data_yaml": str(yaml_path),
        "crop_side_source_pixels": CROP_SIDE,
        "model_image_size": IMAGE_SIZE,
        "samples_per_video_cap": BOXES_PER_VIDEO,
        "counts": counts,
        "train_video_ids": used["train"],
        "validation_video_ids": used["val"],
        "development_video_ids_excluded": sorted(dev_ids),
    }


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _calibrate_confidence(model, dataset: dict, output: Path) -> dict:
    image_dir = Path(dataset["data_yaml"]).parent / "images/val"
    label_dir = Path(dataset["data_yaml"]).parent / "labels/val"
    paths = sorted(image_dir.glob("*.jpg"))
    image_records = []
    batch_size = 12
    for start in range(0, len(paths), batch_size):
        chunk_paths = paths[start:start + batch_size]
        images = [cv2.imread(str(path)) for path in chunk_paths]
        if any(image is None for image in images):
            raise RuntimeError("could not read validation patch")
        predictions = model.predict(
            source=images, imgsz=IMAGE_SIZE, conf=0.001, device="cpu", verbose=False,
        )
        for path, prediction in zip(chunk_paths, predictions):
            label_path = label_dir / f"{path.stem}.txt"
            line = label_path.read_text(encoding="utf-8").strip()
            target = None
            if line:
                _, cx, cy, width, height = map(float, line.split())
                target = np.array([
                    (cx - width / 2) * CROP_SIDE,
                    (cy - height / 2) * CROP_SIDE,
                    (cx + width / 2) * CROP_SIDE,
                    (cy + height / 2) * CROP_SIDE,
                ])
            boxes = []
            if prediction.boxes is not None:
                for xyxy, confidence in zip(prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.conf.cpu().numpy()):
                    boxes.append((float(confidence), np.asarray(xyxy, dtype=float)))
            image_records.append({"target": target, "boxes": boxes})

    confidences = sorted({0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, *[
        confidence for record in image_records for confidence, _ in record["boxes"]
    ]})
    best = None
    for threshold in confidences:
        tp = fp = fn = 0
        for record in image_records:
            accepted = [box for confidence, box in record["boxes"] if confidence >= threshold]
            target = record["target"]
            if target is None:
                fp += len(accepted)
            elif not accepted:
                fn += 1
            else:
                overlaps = [_iou(target, box) for box in accepted]
                hits = sum(overlap >= 0.5 for overlap in overlaps)
                tp += int(hits > 0)
                fn += int(hits == 0)
                fp += len(accepted) - int(hits > 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f05 = 1.25 * precision * recall / (0.25 * precision + recall) if precision + recall else 0.0
        candidate = (f05, precision, threshold, recall, tp, fp, fn)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        raise RuntimeError("could not calibrate local detector confidence")
    return {
        "confidence_threshold": best[2],
        "validation_precision_at_iou_0_5": best[1],
        "validation_recall_at_iou_0_5": best[3],
        "validation_f0_5_at_iou_0_5": best[0],
        "validation_tp": best[4],
        "validation_fp": best[5],
        "validation_fn": best[6],
        "validation_patch_count": len(image_records),
        "threshold_selection": "maximize F0.5 on separate source-video validation crops; no dev video used",
    }


def _read_d0(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _candidate_rows_for_sample(sample: dict, baseline: Path, output_sample: Path, model, threshold: float) -> dict:
    result_path = baseline / "samples" / sample["sample_id"] / "result.json"
    base_result = json.loads(result_path.read_text())
    artifacts = base_result["tracking_artifacts"]
    d0_path = Path(artifacts["trajectory"])
    source_metadata = json.loads(Path(artifacts["metadata"]).read_text())
    d0_rows = _read_d0(d0_path)
    by_frame = {int(row["frame_index"]): row for row in d0_rows}
    cap_end = min(int(sample["end_frame"]), max(by_frame))
    batch: list[tuple[int, dict, np.ndarray, int, int]] = []
    candidate_rows: dict[int, dict] = {}
    reacquired = detector_accepted = attempted = 0
    output_sample.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        nonlocal reacquired, detector_accepted
        if not batch:
            return
        images = [item[2] for item in batch]
        predictions = model.predict(source=images, imgsz=IMAGE_SIZE, conf=0.001, device="cpu", verbose=False)
        for (frame_index, d0_row, _, left, top), prediction in zip(batch, predictions):
            row = dict(d0_row)
            visible_before = str(d0_row.get("visible", "false")).lower() == "true"
            center_x = float(d0_row["raw_x"])
            center_y = float(d0_row["raw_y"])
            detections = []
            if prediction.boxes is not None:
                for xyxy, confidence in zip(prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = map(float, xyxy)
                    bx, by = left + x1, top + y1
                    cx, cy = left + (x1 + x2) / 2, top + (y1 + y2) / 2
                    distance = math.hypot(cx - center_x, cy - center_y)
                    if distance <= CROP_SIDE * 0.52 and x2 - x1 >= 2 and y2 - y1 >= 2:
                        detections.append((float(confidence), -distance, bx, by, x2 - x1, y2 - y1, cx, cy))
            chosen = max(detections, default=None)
            row["detector_confidence"] = chosen[0] if chosen else ""
            row["detector_dx"] = chosen[6] - center_x if chosen else ""
            row["detector_dy"] = chosen[7] - center_y if chosen else ""
            row["detector_accepted"] = bool(chosen and chosen[0] >= threshold)
            if row["detector_accepted"]:
                confidence, _, x, y, width, height, cx, cy = chosen
                row.update(
                    raw_x=cx,
                    raw_y=cy,
                    compensated_x=cx,
                    compensated_y=cy,
                    bbox_x=x,
                    bbox_y=y,
                    bbox_width=width,
                    bbox_height=height,
                    confidence=confidence,
                    visible=True,
                    lifecycle_state="ACTIVE",
                    tracking_source="yolo_local_reacquisition" if not visible_before else "yolo_local_detection",
                )
                detector_accepted += 1
                reacquired += int(not visible_before)
            candidate_rows[frame_index] = row
        batch.clear()

    for frame_index, frame in iter_tracking_video_frames(
        sample["video"], resize_width=FRAME_WIDTH, max_frames=cap_end + 1,
    ):
        d0_row = by_frame.get(frame_index)
        if d0_row is None:
            continue
        attempted += 1
        row = dict(d0_row)
        row.update(detector_confidence="", detector_dx="", detector_dy="", detector_accepted=False)
        candidate_rows[frame_index] = row
        if frame_index == int(sample["init_frame_index"]):
            continue
        try:
            cx, cy = float(d0_row["raw_x"]), float(d0_row["raw_y"])
        except (KeyError, ValueError):
            continue
        patch, left, top = _crop(frame, cx, cy)
        batch.append((frame_index, d0_row, patch, left, top))
        if len(batch) >= 12:
            flush()
    flush()

    ordered_rows = [candidate_rows[index] for index in sorted(candidate_rows)]
    csv_path = output_sample / "trajectory.csv"
    fields = list(dict.fromkeys([*d0_rows[0].keys(), "detector_confidence", "detector_dx", "detector_dy", "detector_accepted"]))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered_rows)

    metadata = dict(source_metadata)
    metadata["tracker_method"] = "ard100_train_yolo11n_local_reacquisition"
    metadata["tracking_parameters"] = {
        "base_tracker": "frozen D0 prediction",
        "local_detector": "YOLO11n trained from scratch on ARD100 train videos",
        "crop_side_processed_pixels": CROP_SIDE,
        "model_image_size": IMAGE_SIZE,
        "confidence_threshold": threshold,
        "threshold_calibration": "separate train-source validation videos",
        "gt_used_after_initialization": False,
        "detector_runs_on_d0_lost_frames": True,
    }
    metadata["visible_frame_count"] = sum(str(row.get("visible", "false")).lower() == "true" for row in ordered_rows)
    metadata["lost_frame_count"] = attempted - metadata["visible_frame_count"]
    metadata_out = output_sample / "metadata.json"
    _write_json(metadata_out, metadata)
    report = evaluate_files(
        cvat_xml=sample["gt_xml"], trajectory_csv=csv_path, metadata_json=metadata_out,
        sample_id=sample["sample_id"], track_ids=(0,),
    )
    _write_json(output_sample / "tracking_evaluation.json", report)
    packed = {
        "sample_id": sample["sample_id"],
        "status": "completed",
        "attempted_frames": attempted,
        "detector_accepted": detector_accepted,
        "reacquired_lost_frames": reacquired,
        "threshold": threshold,
        "tracking_evaluation": report,
        "trajectory": str(csv_path),
        "metadata": str(metadata_out),
    }
    _write_json(output_sample / "result.json", packed)
    return packed


def _metric_row(report: dict) -> dict:
    return {
        "localization": report["localization"]["normalized_threshold_success"]["within_0_5_bbox_diagonal"]["full_gt_ratio"],
        "observation": report["coverage"]["full_gt_observation_ratio"],
        "median_error_px": report["reference_metrics"]["error_px"]["median"],
        "median_error_bbox_width": report["reference_metrics"]["error_bbox_width"]["median"],
        "jitter_px": report["reference_metrics"]["error_change_px"]["median"],
        "temporal_pair_coverage": report["temporal"]["pair_coverage"],
    }


def _baseline_metric(sample_id: str, baseline: Path) -> dict:
    result = json.loads((baseline / "samples" / sample_id / "result.json").read_text())
    return _metric_row(result["tracking_evaluation"])


def _aggregate(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    return {
        "mean_localization": statistics.mean(row["localization"] for row in values),
        "mean_observation": statistics.mean(row["observation"] for row in values),
        "median_video_median_error_px": statistics.median(row["median_error_px"] for row in values if row["median_error_px"] is not None),
        "median_video_median_error_bbox_width": statistics.median(row["median_error_bbox_width"] for row in values if row["median_error_bbox_width"] is not None),
        "median_video_jitter_px": statistics.median(row["jitter_px"] for row in values if row["jitter_px"] is not None),
        "mean_temporal_pair_coverage": statistics.mean(row["temporal_pair_coverage"] for row in values),
    }


def _render_overlay(sample: dict, baseline: Path, output_sample: Path, candidate_path: Path) -> str | None:
    if sample["sample_id"] not in {"phantom105", "phantom111", "phantom133", "phantom81"}:
        return None
    base_result = json.loads((baseline / "samples" / sample["sample_id"] / "result.json").read_text())
    d0_path = Path(base_result["tracking_artifacts"]["trajectory"])
    d0_rows = _read_d0(d0_path)
    with candidate_path.open(newline="", encoding="utf-8-sig") as stream:
        candidate_rows = list(csv.DictReader(stream))
    d0_by_frame = {int(row["frame_index"]): row for row in d0_rows}
    candidate_by_frame = {int(row["frame_index"]): row for row in candidate_rows}
    gt_by_frame = {row["frame_index"]: row for row in _read_boxes(Path(sample["gt_xml"]), int(sample["end_frame"]))}
    output_path = output_sample / "comparison.mp4"
    capture = cv2.VideoCapture(sample["video"])
    fps = float(sample["fps"])
    frame_height = int(round(sample["height"] * FRAME_WIDTH / sample["width"]))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FRAME_WIDTH, frame_height))
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        return None
    end = min(int(sample["end_frame"]), int(sample["init_frame_index"] + fps * 10))
    try:
        for frame_index in range(end + 1):
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, (FRAME_WIDTH, frame_height), interpolation=cv2.INTER_AREA)
            gt = gt_by_frame.get(frame_index)
            if gt:
                scale = FRAME_WIDTH / sample["width"]
                x1, y1, x2, y2 = (int(round(gt[k] * scale)) for k in ("x1", "y1", "x2", "y2"))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 240, 60), 1, cv2.LINE_AA)
                cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 3, (60, 240, 60), -1, cv2.LINE_AA)
            for source, table, color in (("D0", d0_by_frame, (240, 80, 40)), ("YOLO", candidate_by_frame, (20, 150, 255))):
                row = table.get(frame_index)
                if row and str(row.get("visible", "false")).lower() == "true":
                    point = (int(round(float(row["raw_x"]))), int(round(float(row["raw_y"]))))
                    cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)
                    cv2.putText(frame, source, (point[0] + 5, point[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 26), (20, 20, 20), -1)
            cv2.putText(frame, f"{sample['sample_id']} frame={frame_index}  GT=green D0=blue YOLO=orange",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    return str(output_path) if output_path.exists() and output_path.stat().st_size else None


def _write_report(output: Path, summary: dict) -> None:
    aggregate = summary["aggregate"]
    metric_defs = (
        ("mean_localization", "위치 성공률"),
        ("mean_observation", "관측률"),
        ("median_video_median_error_px", "영상별 중앙 오차 중앙값 (px)"),
        ("median_video_median_error_bbox_width", "객체 폭 대비 오차 중앙값"),
        ("median_video_jitter_px", "잔차 변화 중앙값 (px)"),
        ("mean_temporal_pair_coverage", "연속 프레임 비교 비율"),
    )
    overview = "".join(
        f"<tr><th>{escape(label)}</th><td>{aggregate['D0'][key]:.4f}</td><td>{aggregate['YOLO'][key]:.4f}</td></tr>"
        for key, label in metric_defs
    )
    rows = []
    for item in summary["per_video"]:
        d0, yolo = item["D0"], item["YOLO"]
        links = f"<a href='samples/{escape(item['sample_id'])}/trajectory.csv'>궤적 CSV</a>"
        if item.get("comparison_video"):
            links += f" · <a href='samples/{escape(item['sample_id'])}/comparison.mp4'>비교 영상</a>"
        rows.append(
            f"<tr><td>{escape(item['sample_id'])}</td><td>{d0['localization']:.1%}</td><td>{yolo['localization']:.1%}</td>"
            f"<td>{d0['observation']:.1%}</td><td>{yolo['observation']:.1%}</td>"
            f"<td>{d0['median_error_px']:.1f}</td><td>{yolo['median_error_px']:.1f}</td>"
            f"<td>{item['reacquired_lost_frames']}</td><td>{item['detector_accepted']}</td><td>{links}</td></tr>"
        )
    gate = summary["gate"]
    gate_text = "통과" if gate["passed"] else "미통과"
    html = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARD100 YOLO 로컬 재획득</title><style>
body{{font:16px/1.55 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#17212b;background:#f4f6f7}}
h1,h2{{line-height:1.2}}section{{background:#fff;border:1px solid #d6dfe3;border-radius:12px;padding:20px 24px;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:14px;background:#fff}}th,td{{padding:8px;border-bottom:1px solid #dce3e7;text-align:right}}th:first-child,td:first-child{{text-align:left}}thead th{{background:#eaf0f2}}.status{{font-weight:700;color:{'#19743a' if gate['passed'] else '#9a3a25'}}}.small{{font-size:13px;color:#52616a}}
</style><body><h1>ARD100 YOLO 로컬 재획득 실험</h1>
<section><p><b>개발 게이트: <span class="status">{gate_text}</span></b>. 운영 추적기와 기본 설정은 바꾸지 않았습니다.</p>
<p>YOLO11n을 사전학습 가중치 없이 ARD100 학습 영상으로 학습했습니다. D0의 현재 또는 예측 중심에서 192×192 처리 픽셀 패치를 잘라 384 입력으로 검사합니다. 별도 학습 영상 그룹으로 신뢰도 문턱을 정했고, 문턱을 넘는 검출만 좌표로 채택합니다. D0가 lost인 프레임도 재획득을 시도합니다. 개발용 12개 영상은 학습·문턱 조정에서 제외했습니다.</p>
<p class="small">지역 패치 안에 대상이 없거나 모델이 확신하지 못하면 D0 결과를 유지합니다. 이 구조는 전역 탐지기가 아니며 D0 예측 위치에서 96px 안팎을 탐색하는 실험입니다.</p></section>
<section><h2>전체 비교</h2><table><thead><tr><th>지표</th><th>D0</th><th>YOLO 로컬 후보</th></tr></thead><tbody>{overview}</tbody></table>
<p>채택 게이트: 위치 성공률 상승, 관측률 2%p 이내 유지, 영상별 중앙 위치 오차와 흔들림 중앙값 비악화.</p></section>
<section><h2>영상별 비교</h2><div style="overflow:auto"><table><thead><tr><th>영상</th><th>D0 위치</th><th>YOLO 위치</th><th>D0 관측</th><th>YOLO 관측</th><th>D0 오차</th><th>YOLO 오차</th><th>복구 프레임</th><th>수락 검출</th><th>파일</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>학습 기록</h2><p>학습 원본 {len(summary['dataset']['train_video_ids'])}개, 검증 원본 {len(summary['dataset']['validation_video_ids'])}개, 개발 원본 {len(summary['dataset']['development_video_ids_excluded'])}개 제외. 학습 양성 {summary['dataset']['counts']['train']['positive']}, 음성 {summary['dataset']['counts']['train']['negative']}개 패치.</p>
<p>검증에서 고른 confidence={summary['calibration']['confidence_threshold']:.4f}, precision={summary['calibration']['validation_precision_at_iou_0_5']:.3f}, recall={summary['calibration']['validation_recall_at_iou_0_5']:.3f}, F0.5={summary['calibration']['validation_f0_5_at_iou_0_5']:.3f}.</p>
<p class="small">가중치 SHA-256: {summary['training']['weights_sha256']}<br>실험 스크립트 SHA-256: {summary['source_hashes']['experiment_script']}<br>고정 프로토콜 SHA-256: {summary['source_hashes']['frozen_protocol']}</p></section></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def run(baseline: Path, output: Path, epochs: int = EPOCHS) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment: {output}")
    output.mkdir(parents=True)
    yolo_config_dir = output / "ultralytics_config"
    (yolo_config_dir / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)
    cv2.setNumThreads(1)
    random.seed(SEED)
    np.random.seed(SEED)

    import ultralytics
    from ultralytics import YOLO

    protocol_path = baseline / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    dev_samples = _selected_dev_samples(protocol)
    dev_ids = {row["sample_id"] for row in dev_samples}
    started = time.monotonic()
    dataset = _generate_dataset(protocol, dev_ids, output)
    model_yaml = Path(ultralytics.__file__).parent / "cfg/models/11/yolo11.yaml"
    if not model_yaml.exists():
        raise FileNotFoundError(f"local YOLO architecture config not found: {model_yaml}")
    model = YOLO(str(model_yaml))
    run_dir = output / "yolo_runs"
    model.train(
        data=dataset["data_yaml"],
        epochs=epochs,
        patience=4,
        imgsz=IMAGE_SIZE,
        batch=8,
        workers=0,
        device="cpu",
        pretrained=False,
        project=str(run_dir),
        name="ard100_local_yolo11n",
        exist_ok=False,
        cache=False,
        plots=False,
        amp=False,
        seed=SEED,
        deterministic=True,
        mosaic=0.0,
        mixup=0.0,
        fliplr=0.5,
        flipud=0.0,
        translate=0.1,
        scale=0.15,
        hsv_h=0.015,
        hsv_s=0.35,
        hsv_v=0.35,
        verbose=True,
    )
    weight_path = run_dir / "ard100_local_yolo11n" / "weights" / "best.pt"
    if not weight_path.exists():
        raise FileNotFoundError(f"YOLO training did not produce best.pt: {weight_path}")
    detector = YOLO(str(weight_path))
    calibration = _calibrate_confidence(detector, dataset, output)

    per_video = []
    for sample in dev_samples:
        print(f"evaluating {sample['sample_id']}", flush=True)
        sample_output = output / "samples" / sample["sample_id"]
        result = _candidate_rows_for_sample(sample, baseline, sample_output, detector, calibration["confidence_threshold"])
        result["comparison_video"] = _render_overlay(sample, baseline, sample_output, Path(result["trajectory"]))
        _write_json(sample_output / "result.json", result)
        per_video.append({
            "sample_id": sample["sample_id"],
            "D0": _baseline_metric(sample["sample_id"], baseline),
            "YOLO": _metric_row(result["tracking_evaluation"]),
            "detector_accepted": result["detector_accepted"],
            "reacquired_lost_frames": result["reacquired_lost_frames"],
            "comparison_video": result["comparison_video"],
        })

    aggregate = {name: _aggregate(per_video, name) for name in ("D0", "YOLO")}
    gate = {
        "passed": (
            aggregate["YOLO"]["mean_localization"] > aggregate["D0"]["mean_localization"]
            and aggregate["YOLO"]["mean_observation"] >= aggregate["D0"]["mean_observation"] - 0.02
            and aggregate["YOLO"]["median_video_median_error_px"] <= aggregate["D0"]["median_video_median_error_px"]
            and aggregate["YOLO"]["median_video_jitter_px"] <= aggregate["D0"]["median_video_jitter_px"]
        ),
        "passed_dev_videos": len(dev_samples),
        "interpretation": "development gate only; passing does not establish production readiness",
    }
    hashes = {
        "experiment_script": digest(Path(__file__)),
        "evaluator": digest(ROOT / "research/tracking_evaluation.py"),
        "frozen_protocol": digest(protocol_path),
        "weights": digest(weight_path),
    }
    summary = {
        "protocol": "ARD100-yolo-local-reacquisition-v1",
        "dataset": dataset,
        "training": {
            "architecture": "YOLO11n",
            "initialization": "random initialization; no external/pretrained weights",
            "epochs_requested": epochs,
            "weights": str(weight_path),
            "weights_sha256": hashes["weights"],
        },
        "calibration": calibration,
        "source_hashes": hashes,
        "aggregate": aggregate,
        "gate": gate,
        "per_video": per_video,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    _write_json(output / "protocol.json", {
        "protocol": summary["protocol"],
        "baseline": str(baseline.resolve()),
        "development_sample_ids": sorted(dev_ids),
        "dataset": dataset,
        "training": summary["training"],
        "calibration": calibration,
        "candidate": "YOLO11n local detector centered on D0 current/predicted location; lost-frame reacquisition allowed",
        "uses_ground_truth_after_initialization": False,
        "source_hashes": hashes,
    })
    _write_json(output / "summary.json", summary)
    _write_report(output, summary)
    print(json.dumps({"aggregate": aggregate, "calibration": calibration, "gate": gate, "output": str(output)}, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    run(args.baseline, args.output, args.epochs)


if __name__ == "__main__":
    main()
