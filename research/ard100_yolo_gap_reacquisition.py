"""Test causal YOLO re-acquisition over D0 frame gaps using the frozen model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from ai_server.services.tracking_video_io import iter_tracking_video_frames
from research.ard100_sot_probe import digest
from research.tracking_evaluation import evaluate_files
from research.ard100_yolo_local_reacquisition import (
    BASELINE,
    CROP_SIDE,
    FRAME_WIDTH,
    IMAGE_SIZE,
    _aggregate,
    _baseline_metric,
    _crop,
    _metric_row,
    _read_d0,
    _render_overlay,
    _selected_dev_samples,
    _write_json,
    _write_report,
)


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_RUN = ROOT / "artifacts/trajectory_experiments/ard100-yolo-local-reacquisition-20261018-v2"
OUTPUT_DEFAULT = ROOT / "artifacts/trajectory_experiments/ard100-yolo-gap-reacquisition-20261018-v3"
MAX_GAP_FRAMES = 10
CONFIRM_DISTANCE_PX = 24.0


def _select_detection(prediction, origin: tuple[int, int], anchor: tuple[float, float], threshold: float):
    if prediction.boxes is None:
        return None
    left, top = origin
    detections = []
    for xyxy, confidence in zip(prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(float, xyxy)
        width, height = x2 - x1, y2 - y1
        if width < 2 or height < 2:
            continue
        cx, cy = left + (x1 + x2) / 2, top + (y1 + y2) / 2
        distance = math.hypot(cx - anchor[0], cy - anchor[1])
        if distance <= CROP_SIDE * 0.52:
            detections.append({
                "confidence": float(confidence),
                "x": left + x1,
                "y": top + y1,
                "width": width,
                "height": height,
                "cx": cx,
                "cy": cy,
                "distance": distance,
                "accepted": float(confidence) >= threshold,
            })
    return max(detections, key=lambda item: (item["confidence"], -item["distance"])) if detections else None


def _run_video(sample: dict, baseline: Path, output: Path, model, threshold: float) -> dict:
    base_result = json.loads((baseline / "samples" / sample["sample_id"] / "result.json").read_text())
    artifacts = base_result["tracking_artifacts"]
    source_csv = Path(artifacts["trajectory"])
    source_metadata = json.loads(Path(artifacts["metadata"]).read_text())
    source_rows = _read_d0(source_csv)
    visible_d0 = {
        int(row["frame_index"]): row
        for row in source_rows
        if str(row.get("visible", "false")).lower() == "true"
    }
    end_frame = min(int(sample["end_frame"]), int(source_metadata["attempted_frame_count"]) - 1)
    fps = float(sample["fps"])
    output_dir = output / "samples" / sample["sample_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    d0_history: list[tuple[int, float, float, dict]] = []
    raw_detections: dict[int, dict] = {}
    anchors: dict[int, dict] = {}
    pending: list[tuple[int, np.ndarray, int, int, float, float]] = []
    frame_count = 0

    def flush() -> None:
        if not pending:
            return
        predictions = model.predict(
            source=[item[1] for item in pending], imgsz=IMAGE_SIZE,
            conf=0.001, device="cpu", verbose=False,
        )
        for item, prediction in zip(pending, predictions):
            frame_index, _, left, top, anchor_x, anchor_y = item
            raw_detections[frame_index] = _select_detection(
                prediction, (left, top), (anchor_x, anchor_y), threshold,
            )
        pending.clear()

    for frame_index, frame in iter_tracking_video_frames(
        sample["video"], resize_width=FRAME_WIDTH, max_frames=end_frame + 1,
    ):
        frame_count += 1
        observed = visible_d0.get(frame_index)
        if observed is not None:
            anchor_x, anchor_y = float(observed["raw_x"]), float(observed["raw_y"])
            d0_history.append((frame_index, anchor_x, anchor_y, observed))
            d0_history = d0_history[-2:]
            is_gap = False
            base_row = dict(observed)
        else:
            if not d0_history:
                continue
            gap = frame_index - d0_history[-1][0]
            if gap < 1 or gap > MAX_GAP_FRAMES:
                continue
            last_frame, last_x, last_y, last_row = d0_history[-1]
            if len(d0_history) >= 2:
                previous_frame, previous_x, previous_y, _ = d0_history[-2]
                delta = max(1, last_frame - previous_frame)
                velocity_x = (last_x - previous_x) / delta
                velocity_y = (last_y - previous_y) / delta
            else:
                velocity_x = velocity_y = 0.0
            anchor_x = last_x + velocity_x * gap
            anchor_y = last_y + velocity_y * gap
            is_gap = True
            box_width = float(last_row.get("bbox_width", 0) or 0)
            box_height = float(last_row.get("bbox_height", 0) or 0)
            base_row = dict(last_row)
            base_row.update(
                frame_index=frame_index,
                timestamp_ms=int(round(frame_index * 1000 / fps)),
                raw_x=anchor_x,
                raw_y=anchor_y,
                compensated_x=anchor_x,
                compensated_y=anchor_y,
                bbox_x=anchor_x - box_width / 2,
                bbox_y=anchor_y - box_height / 2,
                lifecycle_state="LOST",
                tracking_source="constant_velocity_gap_prediction",
                visible=False,
            )
        patch, left, top = _crop(frame, anchor_x, anchor_y)
        anchors[frame_index] = {
            "row": base_row,
            "is_gap": is_gap,
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
        }
        pending.append((frame_index, patch, left, top, anchor_x, anchor_y))
        if len(pending) >= 12:
            flush()

    flush()

    confirmed_gap_frames = set()
    for frame_index, detection in raw_detections.items():
        if not detection or not detection["accepted"] or not anchors[frame_index]["is_gap"]:
            continue
        for neighbor_index in (frame_index - 1, frame_index + 1):
            neighbor = raw_detections.get(neighbor_index)
            if not neighbor or not neighbor["accepted"]:
                continue
            distance = math.hypot(detection["cx"] - neighbor["cx"], detection["cy"] - neighbor["cy"])
            if distance <= CONFIRM_DISTANCE_PX:
                confirmed_gap_frames.add(frame_index)
                if anchors.get(neighbor_index, {}).get("is_gap"):
                    confirmed_gap_frames.add(neighbor_index)

    candidate_rows = []
    detector_accepted = reacquired = 0
    for frame_index in sorted(anchors):
        anchor = anchors[frame_index]
        row = dict(anchor["row"])
        detection = raw_detections.get(frame_index)
        usable = bool(detection and detection["accepted"] and (not anchor["is_gap"] or frame_index in confirmed_gap_frames))
        row["detector_confidence"] = detection["confidence"] if detection else ""
        row["detector_dx"] = detection["cx"] - anchor["anchor_x"] if detection else ""
        row["detector_dy"] = detection["cy"] - anchor["anchor_y"] if detection else ""
        row["detector_accepted"] = usable
        if anchor["is_gap"] and not usable:
            continue
        if usable:
            row.update(
                raw_x=detection["cx"],
                raw_y=detection["cy"],
                compensated_x=detection["cx"],
                compensated_y=detection["cy"],
                bbox_x=detection["x"],
                bbox_y=detection["y"],
                bbox_width=detection["width"],
                bbox_height=detection["height"],
                confidence=detection["confidence"],
                visible=True,
                lifecycle_state="ACTIVE",
                tracking_source="yolo_gap_reacquisition" if anchor["is_gap"] else "yolo_local_detection",
            )
            detector_accepted += 1
            reacquired += int(anchor["is_gap"])
        candidate_rows.append(row)

    csv_path = output_dir / "trajectory.csv"
    fields = list(dict.fromkeys([
        *source_rows[0].keys(), "detector_confidence", "detector_dx", "detector_dy", "detector_accepted",
    ]))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)

    metadata = dict(source_metadata)
    metadata["tracker_method"] = "ard100_train_yolo11n_causal_gap_reacquisition"
    metadata["tracking_parameters"] = {
        "base_tracker": "frozen D0 observations",
        "local_detector": "frozen YOLO11n from the previous split-safe run",
        "search_crop_processed_pixels": CROP_SIDE,
        "detector_input_size": IMAGE_SIZE,
        "confidence_threshold": threshold,
        "max_gap_frames": MAX_GAP_FRAMES,
        "gap_motion_model": "constant velocity from last two D0 observations; causal only",
        "gap_confirmation": f"two accepted adjacent detections within {CONFIRM_DISTANCE_PX:.1f}px",
        "gt_used_after_initialization": False,
    }
    metadata["visible_frame_count"] = len(candidate_rows)
    metadata["lost_frame_count"] = max(0, int(metadata["attempted_frame_count"]) - len(candidate_rows))
    metadata_path = output_dir / "metadata.json"
    _write_json(metadata_path, metadata)
    evaluation = evaluate_files(
        cvat_xml=sample["gt_xml"], trajectory_csv=csv_path, metadata_json=metadata_path,
        sample_id=sample["sample_id"], track_ids=(0,),
    )
    packed = {
        "sample_id": sample["sample_id"],
        "status": "completed",
        "video_frames_read": frame_count,
        "d0_observed_frames": len(visible_d0),
        "detector_accepted_frames": detector_accepted,
        "gap_frames_reacquired": reacquired,
        "gap_frames_candidate_count": sum(anchor["is_gap"] for anchor in anchors.values()),
        "threshold": threshold,
        "tracking_evaluation": evaluation,
        "trajectory": str(csv_path),
        "metadata": str(metadata_path),
    }
    _write_json(output_dir / "tracking_evaluation.json", evaluation)
    _write_json(output_dir / "result.json", packed)
    return packed


def run(baseline: Path, previous_run: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment: {output}")
    output.mkdir(parents=True)
    config_dir = output / "ultralytics_config"
    (config_dir / "Ultralytics").mkdir(parents=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    cv2.setNumThreads(1)

    from ultralytics import YOLO

    old_summary = json.loads((previous_run / "summary.json").read_text())
    weights = Path(old_summary["training"]["weights"])
    if not weights.is_file():
        raise FileNotFoundError(f"trained YOLO weights missing: {weights}")
    threshold = float(old_summary["calibration"]["confidence_threshold"])
    detector = YOLO(str(weights))
    protocol = json.loads((baseline / "protocol.json").read_text())
    samples = _selected_dev_samples(protocol)
    started = time.monotonic()
    per_video = []
    for sample in samples:
        print(f"gap evaluation {sample['sample_id']}", flush=True)
        result = _run_video(sample, baseline, output, detector, threshold)
        result["comparison_video"] = _render_overlay(
            sample, baseline, output / "samples" / sample["sample_id"], Path(result["trajectory"]),
        )
        _write_json(output / "samples" / sample["sample_id"] / "result.json", result)
        per_video.append({
            "sample_id": sample["sample_id"],
            "D0": _baseline_metric(sample["sample_id"], baseline),
            "YOLO": _metric_row(result["tracking_evaluation"]),
            "detector_accepted": result["detector_accepted_frames"],
            "reacquired_lost_frames": result["gap_frames_reacquired"],
            "comparison_video": result["comparison_video"],
        })
    aggregate = {key: _aggregate(per_video, key) for key in ("D0", "YOLO")}
    gate = {
        "passed": (
            aggregate["YOLO"]["mean_localization"] > aggregate["D0"]["mean_localization"]
            and aggregate["YOLO"]["mean_observation"] >= aggregate["D0"]["mean_observation"] - 0.02
            and aggregate["YOLO"]["median_video_median_error_px"] <= aggregate["D0"]["median_video_median_error_px"]
            and aggregate["YOLO"]["median_video_jitter_px"] <= aggregate["D0"]["median_video_jitter_px"]
        ),
        "passed_dev_videos": len(samples),
        "interpretation": "development gate only; passing does not establish production readiness",
    }
    hashes = {
        "experiment_script": digest(Path(__file__)),
        "frozen_protocol": digest(baseline / "protocol.json"),
        "gap_experiment_script": digest(Path(__file__)),
        "detector_experiment_script": digest(ROOT / "research/ard100_yolo_local_reacquisition.py"),
        "evaluator": digest(ROOT / "research/tracking_evaluation.py"),
        "frozen_protocol": digest(baseline / "protocol.json"),
        "weights": old_summary["training"]["weights_sha256"],
    }
    summary = {
        "protocol": "ARD100-yolo-causal-gap-reacquisition-v1",
        "dataset": old_summary["dataset"],
        "training": old_summary["training"],
        "calibration": old_summary["calibration"],
        "source_hashes": hashes,
        "aggregate": aggregate,
        "gate": gate,
        "per_video": per_video,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    _write_json(output / "protocol.json", {
        "protocol": summary["protocol"],
        "baseline": str(baseline.resolve()),
        "reused_detector_run": str(previous_run.resolve()),
        "development_sample_ids": [sample["sample_id"] for sample in samples],
        "max_gap_frames": MAX_GAP_FRAMES,
        "motion_model": "constant velocity from last two D0 observations, without future frames",
        "gap_confirmation": f"two accepted adjacent detections within {CONFIRM_DISTANCE_PX:.1f}px",
        "confidence_threshold_reused_from_source_video_validation": threshold,
        "uses_ground_truth_after_initialization": False,
        "source_hashes": hashes,
    })
    _write_json(output / "summary.json", summary)
    _write_report(output, summary)
    print(json.dumps({"aggregate": aggregate, "gate": gate, "reacquired": sum(row["reacquired_lost_frames"] for row in per_video), "output": str(output)}, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--previous-run", type=Path, default=PREVIOUS_RUN)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(args.baseline, args.previous_run, args.output)


if __name__ == "__main__":
    main()
