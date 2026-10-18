"""Audit gap-only YOLO hybrid tracks visually and through frozen MiniRocket models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "artifacts/trajectory_experiments/ard100-frozen-20261018-v3"
DEFAULT_HYBRID = ROOT / "artifacts/trajectory_experiments/ard100-yolo-hybrid-gate-20261018-v2"
DEFAULT_OUTPUT = ROOT / "artifacts/trajectory_experiments/ard100-yolo-hybrid-downstream-20261018-v1"
DEFAULT_RUNTIME = ROOT.parent / "skydetect-ai-minirocket-runtime"
DEFAULT_TRAINING_ZIP = Path.home() / "Downloads/real_reference_comparison_v1.zip"
EXPECTED_STACK_SHA256 = "5ba8fc4ac843f30c8a2a590add3f5854f6cf23093d1b8ed1f3b0731b95ec8aab"
RECOVERY_SOURCE = "yolo_hybrid_gap_reacquisition"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def transform_point(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    point = np.asarray(matrix, dtype=np.float64) @ np.asarray([x, y, 1.0])
    if not np.isfinite(point).all() or abs(point[2]) < 1e-12:
        raise ValueError("invalid homogeneous point transform")
    return float(point[0] / point[2]), float(point[1] / point[2])


def transform_box(matrix: np.ndarray, x: float, y: float, width: float,
                  height: float) -> tuple[float, float, float, float]:
    corners = [transform_point(matrix, px, py) for px, py in (
        (x, y), (x + width, y), (x, y + height), (x + width, y + height),
    )]
    xs, ys = zip(*corners)
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    return left, top, right - left, bottom - top


def accepted_recoveries(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("tracking_source") == RECOVERY_SOURCE
            and _is_true(row.get("detector_accepted")) and _is_true(row.get("visible"))]


def extend_track_payload(
    d0_payload: dict,
    recovery_rows: list[dict[str, str]],
    matrices: dict[int, np.ndarray],
) -> dict:
    """Append only accepted gap detections, mapped into D0's verified CMC space."""
    payload = json.loads(json.dumps(d0_payload))
    width, height = int(payload["processed_width"]), int(payload["processed_height"])
    history = list(payload["history"])
    existing = {int(point["frame_index"]) for point in history}
    for row in accepted_recoveries(recovery_rows):
        frame = int(row["frame_index"])
        if frame in existing:
            raise ValueError(f"hybrid recovery overlaps an observed D0 frame: {frame}")
        if frame not in matrices:
            raise ValueError(f"no verified CMC matrix for recovery frame {frame}")
        matrix = matrices[frame]
        x, y = transform_point(matrix, float(row["raw_x"]), float(row["raw_y"]))
        left, top, box_width, box_height = transform_box(
            matrix, float(row["bbox_x"]), float(row["bbox_y"]),
            float(row["bbox_width"]), float(row["bbox_height"]),
        )
        point = {
            "frame_index": frame,
            "timestamp_ms": int(row["timestamp_ms"]),
            "cx": x / width,
            "cy": y / height,
            "w": box_width / width,
            "h": box_height / height,
            "conf": float(row["confidence"]),
        }
        if not all(math.isfinite(float(point[key])) for key in ("cx", "cy", "w", "h", "conf")):
            raise ValueError(f"non-finite recovery point at frame {frame}")
        if not (0 <= point["cx"] <= 1 and 0 <= point["cy"] <= 1
                and 0 <= point["w"] <= 1 and 0 <= point["h"] <= 1
                and 0 <= point["conf"] <= 1):
            raise ValueError(f"recovery point outside TrackSequence bounds at frame {frame}")
        history.append(point)
        existing.add(frame)
    history.sort(key=lambda point: int(point["frame_index"]))
    payload["history"] = history
    payload["quality"] = None
    return payload


def read_matrices(path: Path) -> tuple[dict[int, np.ndarray], dict]:
    payload = read_json(path)
    verification = payload.get("verification", {})
    tolerance = float(verification.get("tolerance_px", 0))
    residual = float(verification.get("max_residual_px", math.inf))
    if not tolerance or residual > tolerance:
        raise ValueError(f"CMC replay verification failed: residual={residual}, tolerance={tolerance}")
    matrices = {int(row["frame_index"]): np.asarray(row["current_to_reference"], dtype=np.float64)
                for row in payload["frames"]}
    return matrices, payload


def verify_d0_coordinate_replay(d0_rows: list[dict[str, str]], d0_payload: dict,
                                matrices: dict[int, np.ndarray]) -> dict:
    points = {int(point["frame_index"]): point for point in d0_payload["history"]}
    width, height = int(d0_payload["processed_width"]), int(d0_payload["processed_height"])
    residuals = []
    for row in d0_rows:
        frame = int(row["frame_index"])
        if frame not in points or frame not in matrices or not _is_true(row.get("visible")):
            continue
        x, y = transform_point(matrices[frame], float(row["raw_x"]), float(row["raw_y"]))
        point = points[frame]
        residuals.append(math.hypot(x - float(point["cx"]) * width,
                                    y - float(point["cy"]) * height))
    if not residuals:
        raise ValueError("no D0 points available to verify CMC coordinate replay")
    max_residual = max(residuals)
    if max_residual > 0.05:
        raise ValueError(f"D0/CMC coordinate mismatch: max residual {max_residual:.4f}px")
    return {"matched_points": len(residuals), "median_residual_px": statistics.median(residuals),
            "max_residual_px": max_residual, "tolerance_px": 0.05}


def read_gt_track(path: Path, track_id: int = 0) -> tuple[str, dict[int, dict[str, float]]]:
    root = ET.parse(path).getroot()
    tracks = [track for track in root.findall("./track") if int(track.attrib["id"]) == track_id]
    if len(tracks) != 1:
        raise ValueError(f"expected one CVAT track id {track_id} in {path}")
    track = tracks[0]
    boxes = {}
    for box in track.findall("./box"):
        if _is_true(box.attrib.get("outside", "0")):
            continue
        left, top = float(box.attrib["xtl"]), float(box.attrib["ytl"])
        right, bottom = float(box.attrib["xbr"]), float(box.attrib["ybr"])
        boxes[int(box.attrib["frame"])] = {
            "cx": (left + right) / 2, "cy": (top + bottom) / 2,
            "x": left, "y": top, "w": right - left, "h": bottom - top,
            "occluded": float(box.attrib.get("occluded", "0")),
        }
    return str(track.attrib.get("label", "unknown")), boxes


def gap_anchor(d0_rows: list[dict[str, str]], frame: int) -> tuple[float, float] | None:
    previous = [row for row in d0_rows if _is_true(row.get("visible"))
                and int(row["frame_index"]) < frame]
    previous.sort(key=lambda row: int(row["frame_index"]))
    if not previous:
        return None
    last = previous[-1]
    x, y = float(last["raw_x"]), float(last["raw_y"])
    if len(previous) == 1:
        return x, y
    prior = previous[-2]
    delta = max(1, int(last["frame_index"]) - int(prior["frame_index"]))
    gap = frame - int(last["frame_index"])
    return (x + (x - float(prior["raw_x"])) * gap / delta,
            y + (y - float(prior["raw_y"])) * gap / delta)


def build_recovery_audit(sample: dict, rows: list[dict[str, str]], d0_rows: list[dict[str, str]],
                         gt_boxes: dict[int, dict], processed_width: int,
                         processed_height: int) -> list[dict]:
    sx, sy = processed_width / int(sample["width"]), processed_height / int(sample["height"])
    results = []
    for row in accepted_recoveries(rows):
        frame = int(row["frame_index"])
        gt = gt_boxes.get(frame)
        anchor = gap_anchor(d0_rows, frame)
        base = {
            "sample_id": sample["sample_id"], "frame_index": frame,
            "timestamp_ms": int(row["timestamp_ms"]),
            "detector_confidence": float(row.get("detector_confidence") or row["confidence"]),
            "gt_available": gt is not None,
        }
        if gt is not None:
            gx, gy = gt["cx"] * sx, gt["cy"] * sy
            box_width = gt["w"] * sx
            hybrid_error = math.hypot(float(row["raw_x"]) - gx, float(row["raw_y"]) - gy)
            base.update(gt_bbox_width_px=box_width, hybrid_error_px=hybrid_error,
                        hybrid_error_bbox_width=(hybrid_error / box_width if box_width else None),
                        gt_center_x=gx, gt_center_y=gy)
            if anchor is not None:
                anchor_error = math.hypot(anchor[0] - gx, anchor[1] - gy)
                base.update(anchor_error_px=anchor_error,
                            anchor_error_bbox_width=(anchor_error / box_width if box_width else None),
                            hybrid_improves_anchor=hybrid_error < anchor_error)
        results.append(base)
    return results


def build_splice_audit(sample: dict, recovery_rows: list[dict[str, str]],
                       d0_payload: dict, hybrid_payload: dict,
                       gt_boxes: dict[int, dict], matrices: dict[int, np.ndarray]) -> list[dict]:
    """Measure one-frame motion discontinuities around inserted recovery points."""
    width, height = int(d0_payload["processed_width"]), int(d0_payload["processed_height"])
    scale_x, scale_y = width / int(sample["width"]), height / int(sample["height"])
    d0_points = {int(p["frame_index"]): p for p in d0_payload["history"]}
    hybrid_points = {int(p["frame_index"]): p for p in hybrid_payload["history"]}
    gt_reference = {}
    for frame, box in gt_boxes.items():
        if frame not in matrices:
            continue
        gt_reference[frame] = transform_point(
            matrices[frame], box["cx"] * scale_x, box["cy"] * scale_y,
        )

    rows = []
    for recovery in accepted_recoveries(recovery_rows):
        frame = int(recovery["frame_index"])
        inserted = hybrid_points.get(frame)
        if inserted is None:
            continue
        ix, iy = float(inserted["cx"]) * width, float(inserted["cy"]) * height
        for side, neighbor_frame in (("before", frame - 1), ("after", frame + 1)):
            neighbor = d0_points.get(neighbor_frame)
            if neighbor is None or frame not in gt_reference or neighbor_frame not in gt_reference:
                continue
            nx, ny = float(neighbor["cx"]) * width, float(neighbor["cy"]) * height
            gx0, gy0 = gt_reference[neighbor_frame]
            gx1, gy1 = gt_reference[frame]
            if side == "before":
                observed_dx, observed_dy = ix - nx, iy - ny
                truth_dx, truth_dy = gx1 - gx0, gy1 - gy0
            else:
                observed_dx, observed_dy = nx - ix, ny - iy
                truth_dx, truth_dy = gx0 - gx1, gy0 - gy1
            rows.append({
                "sample_id": sample["sample_id"], "recovery_frame": frame,
                "side": side, "observed_step_px": math.hypot(observed_dx, observed_dy),
                "gt_step_px": math.hypot(truth_dx, truth_dy),
                "step_vector_error_px": math.hypot(observed_dx - truth_dx, observed_dy - truth_dy),
            })
    return rows


def _center_crop(image: np.ndarray, x: float, y: float, side: int = 88) -> np.ndarray:
    height, width = image.shape[:2]
    left, top = int(round(x - side / 2)), int(round(y - side / 2))
    right, bottom = left + side, top + side
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        left += pad_left
        top += pad_top
    return image[top:top + side, left:left + side].copy()


def _draw_local_marks(crop: np.ndarray, origin_x: float, origin_y: float,
                      gt: dict | None, d0: tuple[float, float] | None,
                      hybrid: tuple[float, float], scale_x: float,
                      scale_y: float) -> np.ndarray:
    """Draw marks in crop-local processed-pixel coordinates."""
    ox, oy = origin_x - 44, origin_y - 44
    scale = 2
    crop = cv2.resize(crop, (176, 176), interpolation=cv2.INTER_NEAREST)
    if gt:
        x1, y1 = gt["x"] * scale_x, gt["y"] * scale_y
        x2, y2 = (gt["x"] + gt["w"]) * scale_x, (gt["y"] + gt["h"]) * scale_y
        cv2.rectangle(crop, (int((x1 - ox) * scale), int((y1 - oy) * scale)),
                      (int((x2 - ox) * scale), int((y2 - oy) * scale)), (0, 255, 0), 1)
    for point, color, radius in ((d0, (255, 80, 30), 3), (hybrid, (0, 130, 255), 3)):
        if point is None:
            continue
        px, py = int((point[0] - ox) * scale), int((point[1] - oy) * scale)
        if 0 <= px < 176 and 0 <= py < 176:
            cv2.circle(crop, (px, py), radius, color, -1, cv2.LINE_AA)
    if gt:
        gcx, gcy = gt["cx"] * scale_x, gt["cy"] * scale_y
        px, py = int((gcx - ox) * scale), int((gcy - oy) * scale)
        if 0 <= px < 176 and 0 <= py < 176:
            cv2.drawMarker(crop, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 9, 1)
    return crop


def render_recovery_sheet(sample: dict, video_rows: list[dict[str, str]], d0_rows: list[dict[str, str]],
                          gt_boxes: dict[int, dict], output: Path,
                          processed_width: int, processed_height: int) -> None:
    candidates = accepted_recoveries(video_rows)
    if not candidates:
        return
    sx, sy = processed_width / int(sample["width"]), processed_height / int(sample["height"])
    cap = cv2.VideoCapture(sample["video"])
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source video: {sample['video']}")
    header_h, row_h, gap = 44, 198, 10
    sheet = np.full((header_h + len(candidates) * (row_h + gap), 560, 3), 248, dtype=np.uint8)
    cv2.putText(sheet, f"{sample['sample_id']} | accepted gap recoveries: {len(candidates)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    for index, row in enumerate(candidates):
        frame_index = int(row["frame_index"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (processed_width, processed_height), interpolation=cv2.INTER_AREA)
        gt = gt_boxes.get(frame_index)
        gx = gt["cx"] * sx if gt else float(row["raw_x"])
        gy = gt["cy"] * sy if gt else float(row["raw_y"])
        anchor = gap_anchor(d0_rows, frame_index)
        hybrid = (float(row["raw_x"]), float(row["raw_y"]))
        centers = [("GT-centered", gx, gy), ("D0-anchor-centered", *(anchor or hybrid)),
                   ("Hybrid-centered", *hybrid)]
        top = header_h + index * (row_h + gap)
        error_text = "GT n/a"
        if gt:
            e = math.hypot(hybrid[0] - gx, hybrid[1] - gy)
            error_text = f"hybrid-GT {e:.1f}px"
            if anchor:
                error_text += f", D0-anchor {math.hypot(anchor[0]-gx, anchor[1]-gy):.1f}px"
        cv2.putText(sheet, f"f={frame_index} t={int(row['timestamp_ms'])/1000:.2f}s conf={float(row['confidence']):.2f} | {error_text}",
                    (8, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (25, 25, 25), 1, cv2.LINE_AA)
        for panel, (label, cx, cy) in enumerate(centers):
            x0 = 8 + panel * 184
            crop = _center_crop(frame, cx, cy)
            crop = _draw_local_marks(crop, cx, cy, gt, anchor, hybrid, sx, sy)
            sheet[top + 20:top + 196, x0:x0 + 176] = crop
            cv2.putText(sheet, label, (x0 + 3, top + 194), cv2.FONT_HERSHEY_SIMPLEX,
                        0.39, (45, 45, 45), 1, cv2.LINE_AA)
    cap.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"failed to write recovery sheet: {output}")


def render_overlay_video(sample: dict, d0_rows: list[dict[str, str]],
                         hybrid_rows: list[dict[str, str]], gt_boxes: dict[int, dict],
                         output: Path, processed_width: int, processed_height: int) -> None:
    cap = cv2.VideoCapture(sample["video"])
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source video: {sample['video']}")
    fps = float(sample["fps"])
    d0_by_frame = {int(row["frame_index"]): row for row in d0_rows if _is_true(row.get("visible"))}
    hybrid_by_frame = {int(row["frame_index"]): row for row in hybrid_rows if _is_true(row.get("visible"))}
    recovered = {int(row["frame_index"]) for row in accepted_recoveries(hybrid_rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (processed_width, processed_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video writer: {output}")
    frame_index = 0
    max_frame = int(sample["end_frame"])
    while frame_index <= max_frame:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (processed_width, processed_height), interpolation=cv2.INTER_AREA)
        gt = gt_boxes.get(frame_index)
        if gt:
            sx, sy = processed_width / int(sample["width"]), processed_height / int(sample["height"])
            p1 = (round(gt["x"] * sx), round(gt["y"] * sy))
            p2 = (round((gt["x"] + gt["w"]) * sx), round((gt["y"] + gt["h"]) * sy))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
        d0 = d0_by_frame.get(frame_index)
        if d0:
            cv2.circle(frame, (round(float(d0["raw_x"])), round(float(d0["raw_y"]))),
                       4, (255, 80, 30), -1, cv2.LINE_AA)
        hy = hybrid_by_frame.get(frame_index)
        if hy:
            color = (0, 130, 255) if frame_index in recovered else (255, 255, 255)
            cv2.circle(frame, (round(float(hy["raw_x"])), round(float(hy["raw_y"]))),
                       3 if frame_index not in recovered else 5, color, -1, cv2.LINE_AA)
            if frame_index in recovered:
                cv2.circle(frame, (round(float(hy["raw_x"])), round(float(hy["raw_y"]))),
                           7, (0, 130, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (8, 8), (570, 45), (25, 25, 25), -1)
        cv2.putText(frame, "GT bbox=green | D0=blue | Hybrid recovery=orange | common point=white",
                    (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{sample['sample_id']}  frame={frame_index}  t={frame_index/fps:.2f}s",
                    (processed_width - 390, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_index += 1
    writer.release()
    cap.release()


def _training_video_hashes(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    with zipfile.ZipFile(path) as archive:
        inventory_name = next((name for name in archive.namelist()
                               if name.endswith("/source_inventory.csv")), None)
        if inventory_name is None:
            raise ValueError("training zip is missing source_inventory.csv")
        text = archive.read(inventory_name).decode("utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    return {row["video_sha256"].strip().lower() for row in rows
            if row.get("video_sha256", "").strip()}


def _classifiers(runtime_root: Path, stack_model: Path, baseline_model: Path):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(runtime_root))
    sys.path.insert(0, str(ROOT / "research"))
    from ai_server.services.classifier import MiniRocketClassifier
    from minirocket_stack import StackMiniRocketClassifier
    return {
        "deployed_baseline": MiniRocketClassifier(baseline_model),
        "frozen_stack_candidate": StackMiniRocketClassifier(stack_model),
    }


def _model_metadata(path: Path) -> dict:
    import joblib
    bundle = joblib.load(path)
    return {key: bundle.get(key) for key in (
        "contract_version", "contract_id", "classes", "channels", "alpha",
        "arm", "train_groups", "train_windows", "package_manifest_sha256",
    )}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(output: Path, summary: dict, predictions: list[dict],
                   recovery_rows: list[dict], visuals: list[dict]) -> None:
    table_rows = []
    by_key = {(row["sample_id"], row["input"], row["model"]): row for row in predictions}
    for sample_id in summary["development_sample_ids"]:
        entries = []
        for model in ("deployed_baseline", "frozen_stack_candidate"):
            d0 = by_key[(sample_id, "D0", model)]
            hy = by_key[(sample_id, "Hybrid", model)]
            entries.append(
                f"{html.escape(model)}: {html.escape(d0['prediction'])} -> "
                f"{html.escape(hy['prediction'])}; margin {fmt(d0['decision_margin'])} -> "
                f"{fmt(hy['decision_margin'])}; windows {d0['accepted_windows']} -> {hy['accepted_windows']}"
            )
        table_rows.append(f"<tr><td>{html.escape(sample_id)}</td><td>{html.escape(summary['truth_by_sample'][sample_id])}</td>"
                          f"<td>{'<br>'.join(entries)}</td></tr>")
    recoveries = summary["recovery_metrics"]
    recovery_table = "".join(
        f"<tr><td>{html.escape(sample)}</td><td>{data['count']}</td><td>{data['gt_scored']}</td>"
        f"<td>{fmt(data.get('anchor_error_median_px'))}</td><td>{fmt(data.get('hybrid_error_median_px'))}</td>"
        f"<td>{data.get('hybrid_better_count', 'n/a')}/{data.get('gt_scored', 0)}</td></tr>"
        for sample, data in recoveries["by_sample"].items() if data["count"]
    )
    visual_blocks = []
    for item in visuals:
        video = f"<video controls preload='metadata' src='{html.escape(item['video'])}'></video>" if item.get("video") else ""
        sheet = f"<a href='{html.escape(item['sheet'])}'><img src='{html.escape(item['sheet'])}' alt='Recovery contact sheet for {html.escape(item['sample_id'])}'></a>" if item.get("sheet") else ""
        visual_blocks.append(f"<section><h3>{html.escape(item['sample_id'])}</h3>{video}{sheet}</section>")
    cls_summary = summary["classification_summary"]
    recovery_summary = summary["recovery_metrics"]
    severe = [row for row in recovery_rows if row.get("gt_available")
              and row.get("hybrid_error_bbox_width") is not None
              and float(row["hybrid_error_bbox_width"]) > 1.0]
    splice_rows = read_csv(output / "splice_audit.csv") if (output / "splice_audit.csv").exists() else []
    phantom66_splices = [row for row in splice_rows if row["sample_id"] == "phantom66"]
    splice_text = ""
    if phantom66_splices:
        details = []
        for row in phantom66_splices:
            details.append(
                f"f{row['recovery_frame']} {row['side']}: A step {float(row['observed_step_px']):.1f}px "
                f"vs GT {float(row['gt_step_px']):.1f}px (motion-vector error "
                f"{float(row['step_vector_error_px']):.1f}px)"
            )
        splice_text = " 인접 프레임 진단은 " + "; ".join(details) + "로, 위치는 맞아도 삽입 경계에 가짜 급격 이동이 생겼음을 보여줍니다."
    details_path = output / "classification_details.json"
    details = json.loads(details_path.read_text(encoding="utf-8")) if details_path.exists() else {}
    detail_by_key = {
        (row["sample_id"], row["input"], row["model"]): row.get("details", {})
        for row in details.get("predictions", [])
    }
    phantom66_lines = []
    for model, display in (("deployed_baseline", "배포 baseline"),
                           ("frozen_stack_candidate", "실험 stack 후보")):
        d0 = detail_by_key.get(("phantom66", "D0", model), {})
        hy = detail_by_key.get(("phantom66", "Hybrid", model), {})
        if d0 and hy:
            d0_map = {round(float(start), 2): float(score)
                      for start, score in zip(d0.get("window_starts_s", []), d0.get("window_scores", []))}
            hy_map = {round(float(start), 2): float(score)
                      for start, score in zip(hy.get("window_starts_s", []), hy.get("window_scores", []))}
            changes = []
            for start in sorted(set(d0_map) | set(hy_map)):
                if start not in d0_map:
                    changes.append(f"새 {start:g}s 창 margin {hy_map[start]:.3f}")
                elif start in hy_map and abs(d0_map[start] - hy_map[start]) > 1e-6:
                    changes.append(f"{start:g}s 창 {d0_map[start]:.3f}→{hy_map[start]:.3f}")
            phantom66_lines.append(
                f"{display}: 판정 {d0.get('label')}→{hy.get('label')}, "
                f"유효 창 {len(d0.get('window_scores', []))}→{len(hy.get('window_scores', []))}; "
                f"{html.escape('; '.join(changes))}."
            )
    phantom66_text = " ".join(phantom66_lines) or "phantom66 창별 margin 상세를 사용할 수 없습니다."
    body = f"""<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARD100 Hybrid downstream audit</title>
<style>
body{{max-width:1180px;margin:32px auto;padding:0 22px;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;color:#18202a;background:#f5f7f7}}
h1,h2{{line-height:1.2}}h1{{font-size:30px}}h2{{margin-top:38px;border-bottom:1px solid #ccd4d6;padding-bottom:8px}}
.note{{background:#fff7dc;border-left:4px solid #d19b00;padding:14px 18px}}table{{border-collapse:collapse;width:100%;background:white;margin:14px 0}}
th,td{{border:1px solid #d9e0e2;padding:8px;text-align:left;vertical-align:top}}th{{background:#e9eff0}}
section{{background:#fff;padding:12px;margin:20px 0;border:1px solid #d9e0e2}}video{{width:min(100%,960px);display:block;margin:8px 0 18px}}section img{{width:min(100%,600px);height:auto}}
code{{background:#eef1f2;padding:2px 5px}}small{{color:#59666a}}
</style>
<h1>ARD100: D0 vs YOLO gap-hybrid, A에서 C까지</h1>
<p>고정 개발 코호트 12개. GT는 라벨 및 사후 점검에만 사용했고, Hybrid 시퀀스 생성이나 MiniRocket 입력 수정에는 사용하지 않았습니다.</p>
<div class="note"><b>해석 제한:</b> 이 12개 ARD100 샘플은 모두 드론 라벨입니다. 따라서 아래는 드론 재현 여부와 D0→Hybrid 입력 민감도 감사이지, bird/drone 분류 정확도 또는 독립 일반화 성능 평가가 아닙니다. margin은 확률이 아닙니다.</div>
<h2>복구 구간 사후 대조</h2>
<p>총 {summary['recovery_metrics']['total']}개 gap recovery 중 CVAT bbox가 있는 {summary['recovery_metrics']['gt_scored_total']}개를 비교했습니다. 각 픽셀은 1280×720 처리 프레임 기준입니다.</p>
<table><thead><tr><th>영상</th><th>복구 수</th><th>GT 대조 수</th><th>D0 CV 예측 오차 중앙값</th><th>Hybrid 오차 중앙값</th><th>Hybrid가 낮은 프레임</th></tr></thead><tbody>{recovery_table}</tbody></table>
<p>정규화 오차는 중심 오차를 해당 프레임의 GT bbox 폭으로 나눈 값입니다. 영상 점수 전체와 달리, 이는 복구된 gap 프레임에 한정한 사후 분석입니다.</p>
<h3>결과를 쉽게 읽으면</h3>
<ul><li>전체 median 오차는 D0 단순 gap 예측 {fmt(recovery_summary['anchor_error_median_px'])}px에서 Hybrid {fmt(recovery_summary['hybrid_error_median_px'])}px로 낮아졌고, 복구점 {recovery_summary['hybrid_better_count']}/{recovery_summary['gt_scored_total']}개에서 Hybrid가 더 가까웠습니다. 다만 이 전체 수치만으로 모든 복구가 안전하다고 볼 수 없습니다.</li>
<li><b>phantom81:</b> 10개 중 5개는 초반의 올바른 복구였지만, 프레임 171·172·173·176·177은 배경 나무 쪽으로 잘못 붙었습니다. GT 중심에서 약 179~185px, GT bbox 폭의 약 12~14배 떨어져 있어 육안으로도 명확한 오탐입니다. 이 다섯 점 때문에 phantom81의 전체 median은 좋아지지 않았습니다.</li>
<li><b>phantom66:</b> 11개 복구점 각각은 GT에서 0.4~1.1px 수준으로 가까웠지만, D0의 주변 궤적과 신원/시간 연속성이 맞지 않는 점을 삽입한 뒤 두 MiniRocket 모두 drone→bird로 판정이 바뀌었습니다. {phantom66_text}{splice_text} 즉, 한 프레임의 위치 정답만 맞추는 것과 B/C가 쓸 연속 궤적을 복원하는 것은 별개입니다.</li>
<li><b>phantom30:</b> 10개 복구점은 모두 gap의 단순 속도 예측보다 GT에 가까웠고, MiniRocket은 drone 판정을 유지하면서 유효 2초 창이 1개 늘었습니다. 긍정 사례지만 단일 개발 샘플입니다.</li></ul>
<h2>MiniRocket 비교</h2>
<p>두 frozen 모델 각각에서 같은 TrackSequence 계약을 사용했습니다. 분류 결정 점수는 모델별 margin이며 확률이 아니므로, 서로 다른 모델 사이의 margin 크기는 직접 비교하지 않습니다.</p>
<table><thead><tr><th>샘플</th><th>CVAT 라벨</th><th>D0 → Hybrid (모델별 판정·margin·유효 창)</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<p>드론 라벨 12개에서 baseline D0→Hybrid 예측변화: {html.escape(json.dumps(cls_summary, ensure_ascii=False))}.</p>
<h2>복구 장면 육안 검수</h2>
<p>전체 비교 영상은 GT bbox=초록, D0 중심=파랑, Hybrid gap 복구=주황, 양쪽에 공통인 Hybrid 중심=흰색입니다. 정지 접촉표에서는 각 행이 복구 프레임이며 GT/D0/Hybrid 중심 주변을 각각 확대했습니다. 초록 사각형=GT bbox, 파랑 점=D0 기준, 주황 점=Hybrid detection, 노란 십자=GT center.</p>
{''.join(visual_blocks)}
<h2>재현 정보</h2>
<ul><li>요약 JSON: <a href="summary.json">summary.json</a></li><li>전체 분류 점수/창 상세: <a href="classification_details.json">classification_details.json</a></li><li>복구 프레임별 GT 대조: <a href="recovery_audit.csv">recovery_audit.csv</a></li><li>복구점 인접 운동 discontinuity: <a href="splice_audit.csv">splice_audit.csv</a></li><li>모델별 요약 표: <a href="classification.csv">classification.csv</a></li></ul>
<p><small>생성 스크립트는 연구용 산출물이며, 운영 모델 및 기존 D0/Hybrid 결과 파일은 수정하지 않았습니다.</small></p>
</html>"""
    (output / "report.html").write_text(body, encoding="utf-8")
    (output / "index.html").write_text(body, encoding="utf-8")


def fmt(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def run(args: argparse.Namespace) -> dict:
    baseline = args.baseline.resolve()
    hybrid_root = args.hybrid.resolve()
    output = args.output.resolve()
    runtime_root = args.runtime_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    protocol = read_json(baseline / "protocol.json")
    gate_summary = read_json(hybrid_root / "summary.json")
    selected_ids = gate_summary["development_sample_ids"]
    samples = {sample["sample_id"]: sample for sample in protocol["samples"]}
    if any(sample_id not in samples for sample_id in selected_ids):
        raise ValueError("hybrid development set and frozen protocol do not match")
    output.mkdir(parents=True, exist_ok=False)

    stack_model = args.stack_model.resolve()
    baseline_model = args.baseline_model.resolve()
    if sha256_file(stack_model) != EXPECTED_STACK_SHA256:
        raise ValueError("frozen stack candidate hash differs from audited model")
    classifiers = _classifiers(runtime_root, stack_model, baseline_model)
    from ai_server.schemas import TrackSequence
    from ai_server.utils.quality import build_track_quality

    training_video_hashes = _training_video_hashes(args.training_zip.resolve() if args.training_zip else None)
    models = {
        "deployed_baseline": {"path_name": baseline_model.name, "sha256": sha256_file(baseline_model),
                              "metadata": _model_metadata(baseline_model)},
        "frozen_stack_candidate": {"path_name": stack_model.name, "sha256": sha256_file(stack_model),
                                    "metadata": _model_metadata(stack_model)},
    }
    (output / "tracks").mkdir()
    (output / "visual").mkdir()
    predictions, prediction_details, recovery_rows, splice_rows, visuals = [], [], [], [], []
    transforms_audit, truth_by_sample, sample_sources = {}, {}, {}

    for sample_id in selected_ids:
        sample = samples[sample_id]
        sample_dir = baseline / "samples" / sample_id
        d0_sequence_path = sample_dir / "b_inputs/a_track_sequence.json"
        transforms_path = sample_dir / "b_inputs/camera_transforms.json"
        run_dirs = list((sample_dir / "runs").glob("*/trajectory.csv"))
        if len(run_dirs) != 1:
            raise ValueError(f"expected one frozen D0 trajectory for {sample_id}, got {len(run_dirs)}")
        d0_csv_path = run_dirs[0]
        d0_rows = read_csv(d0_csv_path)
        d0_payload = read_json(d0_sequence_path)
        matrices, transform_payload = read_matrices(transforms_path)
        transforms_audit[sample_id] = verify_d0_coordinate_replay(d0_rows, d0_payload, matrices)
        candidate_csv_path = hybrid_root / "samples" / sample_id / "trajectory.csv"
        candidate_rows = read_csv(candidate_csv_path) if candidate_csv_path.exists() else d0_rows
        recoveries = accepted_recoveries(candidate_rows)

        d0_track_data = json.loads(json.dumps(d0_payload))
        hybrid_track_data = extend_track_payload(d0_payload, candidate_rows, matrices)
        attempted_count = int(sample["end_frame"]) - int(sample["init_frame_index"]) + 1
        for payload in (d0_track_data, hybrid_track_data):
            validated = TrackSequence.model_validate(payload)
            quality = build_track_quality(validated.history, attempted_frame_count=attempted_count)
            payload["quality"] = quality.model_dump(mode="json")
        tracks = {
            "D0": TrackSequence.model_validate(d0_track_data),
            "Hybrid": TrackSequence.model_validate(hybrid_track_data),
        }
        if [p.frame_index for p in tracks["D0"].history] != [int(p["frame_index"]) for p in d0_payload["history"]]:
            raise AssertionError("D0 history changed during downstream preparation")
        for kind, track in tracks.items():
            track_path = output / "tracks" / sample_id / f"{kind}.json"
            track_path.parent.mkdir(parents=True, exist_ok=True)
            track_path.write_text(track.model_dump_json(indent=2), encoding="utf-8")

        label, gt_boxes = read_gt_track(Path(sample["gt_xml"]), track_id=0)
        truth_by_sample[sample_id] = label
        if label not in {"drone", "bird"}:
            raise ValueError(f"unexpected CVAT target label {label!r} for {sample_id}")
        sample_hash = sha256_file(Path(sample["video"]))
        sample_sources[sample_id] = {
            "source_video_sha256": sample_hash,
            "exact_hash_match_in_real_reference_package": sample_hash in training_video_hashes if training_video_hashes else None,
            "xml_label": label,
            "xml_track_id": 0,
            "d0_points": len(tracks["D0"].history),
            "hybrid_points": len(tracks["Hybrid"].history),
            "gap_recoveries_added": len(recoveries),
            "cmc_replay_method": transform_payload.get("method"),
            "cmc_replay_verification": transform_payload.get("verification"),
        }
        recovery_rows.extend(build_recovery_audit(
            sample, candidate_rows, d0_rows, gt_boxes, int(d0_payload["processed_width"]),
            int(d0_payload["processed_height"]),
        ))
        splice_rows.extend(build_splice_audit(
            sample, candidate_rows, d0_payload, hybrid_track_data, gt_boxes, matrices,
        ))
        if recoveries:
            sheet_path = output / "visual" / f"recoveries_{sample_id}.jpg"
            video_path = output / "visual" / f"overlay_{sample_id}.mp4"
            render_recovery_sheet(sample, candidate_rows, d0_rows, gt_boxes, sheet_path,
                                  int(d0_payload["processed_width"]), int(d0_payload["processed_height"]))
            render_overlay_video(sample, d0_rows, candidate_rows, gt_boxes, video_path,
                                 int(d0_payload["processed_width"]), int(d0_payload["processed_height"]))
            visuals.append({"sample_id": sample_id, "sheet": f"visual/{sheet_path.name}",
                            "video": f"visual/{video_path.name}", "recoveries": len(recoveries)})

        for kind, track in tracks.items():
            for model_name, classifier in classifiers.items():
                result = asdict(classifier.predict(track))
                row = {
                    "sample_id": sample_id, "truth": label, "input": kind, "model": model_name,
                    "prediction": result["label"], "decision_margin": result["decision_score"],
                    "margin_is_probability": False, "correct_on_drone_only_slice": result["label"] == label,
                    "accepted_windows": len(result.get("window_scores") or []),
                    "rejections": json.dumps(result.get("window_rejections") or {}, sort_keys=True),
                    "abstain_reason": result.get("abstain_reason"),
                }
                predictions.append(row)
                prediction_details.append({**row, "details": result})

    recovery_gt = [row for row in recovery_rows if row["gt_available"]]
    by_sample = {}
    for sample_id in selected_ids:
        subset = [row for row in recovery_rows if row["sample_id"] == sample_id]
        scored = [row for row in subset if row["gt_available"] and "hybrid_error_px" in row]
        anchor_values = [row["anchor_error_px"] for row in scored if "anchor_error_px" in row]
        hybrid_values = [row["hybrid_error_px"] for row in scored]
        by_sample[sample_id] = {
            "count": len(subset), "gt_scored": len(scored),
            "anchor_error_median_px": statistics.median(anchor_values) if anchor_values else None,
            "hybrid_error_median_px": statistics.median(hybrid_values) if hybrid_values else None,
            "anchor_error_median_bbox_width": statistics.median(
                [row["anchor_error_bbox_width"] for row in scored if "anchor_error_bbox_width" in row]
            ) if any("anchor_error_bbox_width" in row for row in scored) else None,
            "hybrid_error_median_bbox_width": statistics.median(
                [row["hybrid_error_bbox_width"] for row in scored]
            ) if scored else None,
            "hybrid_better_count": sum(bool(row.get("hybrid_improves_anchor")) for row in scored),
        }
    anchor_all = [row["anchor_error_px"] for row in recovery_gt if "anchor_error_px" in row]
    hybrid_all = [row["hybrid_error_px"] for row in recovery_gt if "hybrid_error_px" in row]
    splice_by_sample = {}
    for sample_id in selected_ids:
        rows = [row for row in splice_rows if row["sample_id"] == sample_id]
        splice_by_sample[sample_id] = {
            "adjacent_sides_scored": len(rows),
            "median_step_vector_error_px": statistics.median(
                float(row["step_vector_error_px"]) for row in rows
            ) if rows else None,
            "max_step_vector_error_px": max(
                (float(row["step_vector_error_px"]) for row in rows), default=None,
            ),
        }

    by_model = {}
    for model_name in models:
        d0 = {row["sample_id"]: row for row in predictions if row["model"] == model_name and row["input"] == "D0"}
        hy = {row["sample_id"]: row for row in predictions if row["model"] == model_name and row["input"] == "Hybrid"}
        by_model[model_name] = {
            "d0_drone_predictions": sum(row["prediction"] == "drone" for row in d0.values()),
            "hybrid_drone_predictions": sum(row["prediction"] == "drone" for row in hy.values()),
            "label_flips": [sample for sample in selected_ids if d0[sample]["prediction"] != hy[sample]["prediction"]],
            "mean_absolute_margin_change": statistics.mean(
                abs(float(hy[sample]["decision_margin"]) - float(d0[sample]["decision_margin"]))
                for sample in selected_ids if d0[sample]["decision_margin"] is not None and hy[sample]["decision_margin"] is not None
            ),
            "total_d0_windows": sum(int(row["accepted_windows"]) for row in d0.values()),
            "total_hybrid_windows": sum(int(row["accepted_windows"]) for row in hy.values()),
        }

    summary = {
        "title": "ARD100 D0 vs YOLO gap-hybrid downstream audit",
        "generated_from": {"baseline": str(baseline), "hybrid": str(hybrid_root)},
        "development_sample_ids": selected_ids,
        "truth_by_sample": truth_by_sample,
        "all_samples_share_same_truth": len(set(truth_by_sample.values())) == 1,
        "models": models,
        "sample_sources": sample_sources,
        "cmc_coordinate_replay_checks": transforms_audit,
        "recovery_metrics": {
            "total": len(recovery_rows), "gt_scored_total": len(recovery_gt),
            "anchor_error_median_px": statistics.median(anchor_all) if anchor_all else None,
            "hybrid_error_median_px": statistics.median(hybrid_all) if hybrid_all else None,
            "anchor_error_median_bbox_width": statistics.median(
                [row["anchor_error_bbox_width"] for row in recovery_gt if "anchor_error_bbox_width" in row]
            ) if recovery_gt else None,
            "hybrid_error_median_bbox_width": statistics.median(
                [row["hybrid_error_bbox_width"] for row in recovery_gt if "hybrid_error_bbox_width" in row]
            ) if recovery_gt else None,
            "hybrid_better_count": sum(bool(row.get("hybrid_improves_anchor")) for row in recovery_gt),
            "by_sample": by_sample,
        },
        "splice_metrics": {
            "adjacent_sides_scored": len(splice_rows),
            "median_step_vector_error_px": statistics.median(
                float(row["step_vector_error_px"]) for row in splice_rows
            ) if splice_rows else None,
            "max_step_vector_error_px": max(
                (float(row["step_vector_error_px"]) for row in splice_rows), default=None,
            ),
            "by_sample": splice_by_sample,
            "details": splice_rows,
        },
        "classification_summary": by_model,
        "training_overlap_audit": {
            "package_zip_checked": bool(training_video_hashes),
            "exact_video_hash_matches": [sid for sid, row in sample_sources.items()
                                          if row["exact_hash_match_in_real_reference_package"] is True],
            "limitation": "No exact video-file hash match does not rule out a re-encoded or otherwise related source recording.",
        },
        "limitations": [
            "All 12 selected ARD100 XML targets are drone-labelled; this is not a balanced bird/drone accuracy estimate.",
            "ARD100 train clips were used as a fixed development slice during tracker selection; the results are diagnostic, not an untouched test.",
            "GT is only used for post-hoc recovery audit and rendering, never to construct Hybrid coordinates or classifier inputs.",
            "The deployed baseline and experimental stack candidate are frozen; no model training or threshold tuning was performed.",
            "Margins are decision scores, not calibrated probabilities; scales differ between models.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "classification_details.json").write_text(
        json.dumps({"score_is_probability": False, "predictions": prediction_details},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(output / "classification.csv", predictions)
    _write_csv(output / "recovery_audit.csv", recovery_rows)
    _write_csv(output / "splice_audit.csv", splice_rows)
    _render_report(output, summary, predictions, recovery_rows, visuals)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--hybrid", type=Path, default=DEFAULT_HYBRID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--baseline-model", type=Path,
                        default=DEFAULT_RUNTIME / "models/minirocket_classifier.joblib")
    parser.add_argument("--stack-model", type=Path,
                        default=ROOT / "artifacts/trajectory_experiments/minirocket-stack-20261018/stack.joblib")
    parser.add_argument("--training-zip", type=Path, default=DEFAULT_TRAINING_ZIP)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"output": str(args.output.resolve()),
                      "recoveries": result["recovery_metrics"]["total"],
                      "gt_scored": result["recovery_metrics"]["gt_scored_total"],
                      "classification": result["classification_summary"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
