"""Train a cross-video patch verifier and test it around frozen D0 tracks.

The detector is research-only. ARD100 train videos selected for the fixed
development set are excluded from fitting. At inference it sees video pixels,
the D0 center, and the initialization box size; later GT is used only by the
offline evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics
import time
import xml.etree.ElementTree as ET

import cv2
import joblib
import numpy as np
from sklearn.svm import LinearSVC

from research.ard100_sot_probe import digest
from research.tracking_evaluation import evaluate_files
from ai_server.services.tracking_video_io import iter_tracking_video_frames


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/trajectory_experiments/ard100-frozen-20261018-v3"
OUTPUT_DEFAULT = ROOT / "artifacts/trajectory_experiments/ard100-local-patch-detector-20261018"
WINDOW = 64
CONTEXT_SCALE = 3.0
TRAIN_BOXES_PER_VIDEO = 24
SEARCH_OFFSETS = tuple(index * 0.75 for index in range(-5, 6))
SEARCH_SCALES = (0.75, 1.0, 1.5)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def _read_boxes(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    rows = []
    for track in root.findall("./track"):
        if track.attrib.get("label", "").lower() != "drone":
            continue
        for box in track.findall("./box"):
            if box.attrib.get("outside", "0") in {"1", "true"}:
                continue
            rows.append({
                "frame_index": int(box.attrib["frame"]),
                "x1": float(box.attrib["xtl"]),
                "y1": float(box.attrib["ytl"]),
                "x2": float(box.attrib["xbr"]),
                "y2": float(box.attrib["ybr"]),
            })
    return sorted(rows, key=lambda row: row["frame_index"])


def _make_hog() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        (WINDOW, WINDOW), (16, 16), (8, 8), (8, 8), 9,
    )


def _extract_patch(frame: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
    side = max(4, int(round(side)))
    half = side / 2.0
    left = int(math.floor(cx - half))
    top = int(math.floor(cy - half))
    right, bottom = left + side, top + side
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - frame.shape[1]), max(0, bottom - frame.shape[0])
    if pad_left or pad_top or pad_right or pad_bottom:
        frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101,
        )
        left += pad_left
        top += pad_top
    crop = frame[top:top + side, left:left + side]
    interpolation = cv2.INTER_CUBIC if side < WINDOW else cv2.INTER_AREA
    return cv2.resize(crop, (WINDOW, WINDOW), interpolation=interpolation)


def _feature(hog: cv2.HOGDescriptor, frame: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
    patch = _extract_patch(frame, cx, cy, side)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return hog.compute(gray).reshape(-1).astype(np.float32)


def _selected_dev_samples(protocol: dict) -> list[dict]:
    ready = [row for row in protocol["samples"] if row["split"] == "train" and row["status"] == "ready"]
    return ready[::max(1, len(ready) // 12)][:12]


def _fit_detector(protocol: dict, dev_ids: set[str], output: Path) -> dict:
    train_videos = [
        row for row in protocol["samples"]
        if row["split"] == "train" and row["status"] == "ready" and row["sample_id"] not in dev_ids
    ]
    hog = _make_hog()
    features: list[np.ndarray] = []
    labels: list[int] = []
    per_video = []
    skipped_tiny = 0
    extraction_started = time.monotonic()

    for sample in train_videos:
        boxes = _read_boxes(Path(sample["gt_xml"]))
        if not boxes:
            per_video.append({"sample_id": sample["sample_id"], "status": "no_boxes"})
            continue
        indices = np.linspace(0, len(boxes) - 1, min(TRAIN_BOXES_PER_VIDEO, len(boxes)), dtype=int)
        selected = [boxes[index] for index in sorted(set(indices.tolist()))]
        by_frame = {box["frame_index"]: box for box in selected}
        last_frame = max(by_frame)
        scale = 1280.0 / sample["width"]
        capture = cv2.VideoCapture(sample["video"])
        if not capture.isOpened():
            raise RuntimeError(f"could not open training video {sample['sample_id']}")
        positive_count = 0
        try:
            for frame_index in range(last_frame + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                box = by_frame.get(frame_index)
                if box is None:
                    continue
                processed_width = int(round(sample["width"] * scale))
                processed_height = int(round(sample["height"] * scale))
                frame = cv2.resize(frame, (processed_width, processed_height), interpolation=cv2.INTER_AREA)
                x1, y1, x2, y2 = (box[key] * scale for key in ("x1", "y1", "x2", "y2"))
                width, height = x2 - x1, y2 - y1
                if min(width, height) < 2.0:
                    skipped_tiny += 1
                    continue
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                side = CONTEXT_SCALE * max(width, height)
                features.append(_feature(hog, frame, cx, cy, side))
                labels.append(1)
                positive_count += 1

                # Hard negatives put the annotated object off-center in the same context window.
                for dx, dy in ((.42 * side, 0), (-.42 * side, 0), (0, .42 * side), (0, -.42 * side)):
                    features.append(_feature(hog, frame, cx + dx, cy + dy, side))
                    labels.append(0)
        finally:
            capture.release()
        per_video.append({"sample_id": sample["sample_id"], "status": "used", "positive_patches": positive_count})
        print(f"training patches {sample['sample_id']}: {positive_count}", flush=True)

    if not features or len(set(labels)) != 2:
        raise RuntimeError("insufficient positive/negative patches to fit local verifier")
    matrix = np.vstack(features)
    label_array = np.asarray(labels, dtype=np.int32)
    classifier = LinearSVC(C=0.05, class_weight="balanced", max_iter=5000, random_state=17)
    classifier.fit(matrix, label_array)
    bundle = {
        "classifier": classifier,
        "window": WINDOW,
        "context_scale": CONTEXT_SCALE,
        "search_scales": SEARCH_SCALES,
        "feature": "OpenCV HOG grayscale 64x64, 9 bins",
        "fit_videos": [row["sample_id"] for row in train_videos],
        "excluded_development_videos": sorted(dev_ids),
    }
    model_path = output / "local_patch_detector.joblib"
    joblib.dump(bundle, model_path)
    return {
        "videos_requested": len(train_videos),
        "videos_with_patches": sum(row.get("status") == "used" for row in per_video),
        "positive_patches": int((label_array == 1).sum()),
        "hard_negative_patches": int((label_array == 0).sum()),
        "tiny_boxes_skipped": skipped_tiny,
        "feature_dimensions": int(matrix.shape[1]),
        "elapsed_seconds": round(time.monotonic() - extraction_started, 2),
        "per_video": per_video,
        "model_path": str(model_path),
        "model_sha256": digest(model_path),
    }


def _read_d0_rows(path: Path) -> tuple[list[dict], dict[int, dict]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    visible = {int(row["frame_index"]): row for row in rows if row.get("visible", "").lower() == "true"}
    return rows, visible


def _candidate_at_frame(
    frame: np.ndarray,
    d0_row: dict,
    initial_width: float,
    initial_height: float,
    hog: cv2.HOGDescriptor,
    classifier: LinearSVC,
) -> dict | None:
    center_x, center_y = float(d0_row["raw_x"]), float(d0_row["raw_y"])
    base_side = max(initial_width, initial_height)
    step = max(2.0, 0.75 * base_side)
    candidates = []
    center_scores = []
    for scale in SEARCH_SCALES:
        side = CONTEXT_SCALE * base_side * scale
        for oy in SEARCH_OFFSETS:
            for ox in SEARCH_OFFSETS:
                x, y = center_x + ox * base_side, center_y + oy * base_side
                score = float(classifier.decision_function([_feature(hog, frame, x, y, side)])[0])
                distance = math.hypot(x - center_x, y - center_y)
                candidates.append((score, distance, x, y, scale))
                if ox == 0 and oy == 0:
                    center_scores.append(score)
    d0_score = max(center_scores)
    best = max(candidates, key=lambda item: (item[0], -item[1]))
    if d0_score >= 0:
        action = "verified_d0"
        chosen = (d0_score, 0.0, center_x, center_y, 1.0)
    elif best[0] >= 0:
        action = "local_reacquisition" if best[1] > 0 else "scale_verified"
        chosen = best
    else:
        return None
    return {
        "raw_x": chosen[2], "raw_y": chosen[3],
        "detector_score_at_d0": d0_score,
        "detector_score": chosen[0],
        "detector_action": action,
        "detector_dx": chosen[2] - center_x,
        "detector_dy": chosen[3] - center_y,
        "candidate_scale": chosen[4],
    }


def _evaluate_sample(sample: dict, baseline: Path, output: Path, bundle: dict) -> dict:
    result_path = baseline / "samples" / sample["sample_id"] / "result.json"
    base_result = json.loads(result_path.read_text())
    artifacts = base_result["tracking_artifacts"]
    d0_path = Path(artifacts["trajectory"])
    metadata_path = Path(artifacts["metadata"])
    source_metadata = json.loads(metadata_path.read_text())
    d0_rows, d0_visible = _read_d0_rows(d0_path)
    initial_width = sample["target_bbox"][2] * source_metadata["processed_width"] / sample["width"]
    initial_height = sample["target_bbox"][3] * source_metadata["processed_width"] / sample["width"]
    hog = _make_hog()
    classifier = bundle["classifier"]
    candidate_rows = []
    cap_end = int(sample["end_frame"])
    output_sample = output / "samples" / sample["sample_id"]
    output_sample.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()

    for frame_index, frame in iter_tracking_video_frames(sample["video"], resize_width=1280, max_frames=cap_end + 1):
        d0_row = d0_visible.get(frame_index)
        if d0_row is None:
            continue
        prediction = _candidate_at_frame(frame, d0_row, initial_width, initial_height, hog, classifier)
        if prediction is None:
            continue
        moved = prediction["detector_action"] == "local_reacquisition"
        row = dict(d0_row)
        row.update(prediction)
        row["frame_index"] = frame_index
        row["timestamp_ms"] = int(d0_row["timestamp_ms"])
        if moved:
            scale = prediction["candidate_scale"]
            width, height = initial_width * scale, initial_height * scale
            row.update(
                bbox_x=prediction["raw_x"] - width / 2,
                bbox_y=prediction["raw_y"] - height / 2,
                bbox_width=width,
                bbox_height=height,
                compensated_x=prediction["raw_x"],
                compensated_y=prediction["raw_y"],
                tracking_source="local_patch_detector",
            )
        else:
            row["tracking_source"] = "local_patch_verified_d0"
        row["visible"] = True
        row["lifecycle_state"] = "ACTIVE"
        candidate_rows.append(row)

    csv_path = output_sample / "trajectory.csv"
    fields = list(dict.fromkeys([*d0_rows[0].keys(), "detector_score_at_d0", "detector_score",
                                  "detector_action", "detector_dx", "detector_dy", "candidate_scale"]))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)

    metadata = dict(source_metadata)
    metadata["tracker_method"] = "ard100_train_local_hog_patch_verifier"
    metadata["tracking_parameters"] = {
        "base_tracker": "frozen D0",
        "verifier": "LinearSVC over HOG patches trained on other ARD100 train videos",
        "fit_excluded_sample_ids": bundle["excluded_development_videos"],
        "search_scales": list(SEARCH_SCALES),
        "search_offsets_in_initial_box_widths": list(SEARCH_OFFSETS),
        "acceptance": "keep D0 if its best-scale score >= 0; otherwise move to local best only if score >= 0; abstain otherwise",
        "gt_used_after_initialization": False,
    }
    metadata["visible_frame_count"] = len(candidate_rows)
    metadata["lost_frame_count"] = int(metadata["attempted_frame_count"]) - len(candidate_rows)
    metadata_path_out = output_sample / "metadata.json"
    _write_json(metadata_path_out, metadata)
    report = evaluate_files(
        cvat_xml=sample["gt_xml"], trajectory_csv=csv_path, metadata_json=metadata_path_out,
        sample_id=sample["sample_id"], track_ids=(0,),
    )
    _write_json(output_sample / "tracking_evaluation.json", report)
    packed = {
        "sample_id": sample["sample_id"],
        "status": "completed",
        "d0_visible_frames": len(d0_visible),
        "verifier_visible_frames": len(candidate_rows),
        "local_reacquisitions": sum(row["detector_action"] == "local_reacquisition" for row in candidate_rows),
        "verified_d0_frames": sum(row["detector_action"] == "verified_d0" for row in candidate_rows),
        "abstentions": len(d0_visible) - len(candidate_rows),
        "elapsed_seconds": round(time.monotonic() - start_time, 2),
        "tracking_evaluation": report,
        "trajectory": str(csv_path),
        "metadata": str(metadata_path_out),
    }
    _write_json(output_sample / "result.json", packed)
    return packed


def _metric_row(result: dict) -> dict:
    report = result["tracking_evaluation"]
    location = report["localization"]["normalized_threshold_success"]["within_0_5_bbox_diagonal"]["full_gt_ratio"]
    coverage = report["coverage"]["full_gt_observation_ratio"]
    reference = report["reference_metrics"]
    return {
        "localization": location,
        "observation": coverage,
        "median_error_px": reference["error_px"]["median"],
        "median_error_bbox_width": reference["error_bbox_width"]["median"],
        "jitter_px": reference["error_change_px"]["median"],
        "temporal_pair_coverage": report["temporal"]["pair_coverage"],
    }


def _baseline_metric(sample_id: str, baseline: Path) -> dict:
    result = json.loads((baseline / "samples" / sample_id / "result.json").read_text())
    return _metric_row({"tracking_evaluation": result["tracking_evaluation"]})


def _aggregate(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    return {
        "mean_localization": statistics.mean(row["localization"] for row in values),
        "mean_observation": statistics.mean(row["observation"] for row in values),
        "median_video_median_error_px": statistics.median(
            row["median_error_px"] for row in values if row["median_error_px"] is not None
        ),
        "median_video_jitter_px": statistics.median(
            row["jitter_px"] for row in values if row["jitter_px"] is not None
        ),
        "mean_temporal_pair_coverage": statistics.mean(
            row["temporal_pair_coverage"] for row in values
        ),
    }


def _render_comparison_video(sample: dict, baseline: Path, output: Path, result: dict) -> str | None:
    if sample["sample_id"] not in {"phantom105", "phantom111", "phantom133", "phantom81"}:
        return None
    d0_result = json.loads((baseline / "samples" / sample["sample_id"] / "result.json").read_text())
    d0_rows, d0_visible = _read_d0_rows(Path(d0_result["tracking_artifacts"]["trajectory"]))
    candidate_rows, candidate_visible = _read_d0_rows(Path(result["trajectory"]))
    del d0_rows, candidate_rows
    gt_by_frame = {row["frame_index"]: row for row in _read_boxes(Path(sample["gt_xml"]))}
    output_path = output / "samples" / sample["sample_id"] / "comparison.mp4"
    capture = cv2.VideoCapture(sample["video"])
    width = int(round(sample["width"] * 1280 / sample["width"]))
    height = int(round(sample["height"] * 1280 / sample["width"]))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), sample["fps"], (width, height))
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        return None
    end = min(int(sample["end_frame"]), int(sample["init_frame_index"] + sample["fps"] * 10))
    try:
        for frame_index in range(end + 1):
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            gt = gt_by_frame.get(frame_index)
            if gt:
                scale = 1280 / sample["width"]
                x1, y1, x2, y2 = (int(round(gt[k] * scale)) for k in ("x1", "y1", "x2", "y2"))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 240, 60), 1, cv2.LINE_AA)
                cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 3, (60,240,60), -1, cv2.LINE_AA)
            for table, color, label in ((d0_visible, (240,80,40), "D0"), (candidate_visible, (20,150,255), "LOCAL")):
                row = table.get(frame_index)
                if row:
                    point = (int(round(float(row["raw_x"]))), int(round(float(row["raw_y"]))))
                    cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)
                    cv2.putText(frame, label, (point[0]+5, point[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (width, 26), (20, 20, 20), -1)
            cv2.putText(frame, f"{sample['sample_id']} frame={frame_index}  GT=green D0=blue LOCAL=orange",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (240,240,240), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    return str(output_path) if output_path.exists() and output_path.stat().st_size else None


def _write_report(output: Path, summary: dict) -> None:
    aggregate = summary["aggregate"]
    keys = (
        ("mean_localization", "위치 성공률"),
        ("mean_observation", "관측률"),
        ("median_video_median_error_px", "영상별 중앙 오차의 중앙값 (px)"),
        ("median_video_jitter_px", "잔차 변화 중앙값 (px)"),
        ("mean_temporal_pair_coverage", "연속 프레임 비교 비율"),
    )
    metric_rows = "".join(
        f"<tr><th>{escape(label)}</th><td>{aggregate['D0'][key]:.4f}</td><td>{aggregate['verifier'][key]:.4f}</td></tr>"
        for key, label in keys
    )
    videos = []
    for row in summary["per_video"]:
        d0, candidate = row["D0"], row["verifier"]
        links = f"<a href='samples/{escape(row['sample_id'])}/trajectory.csv'>궤적</a>"
        if row.get("comparison_video"):
            links += f" · <a href='samples/{escape(row['sample_id'])}/comparison.mp4'>비교 영상</a>"
        videos.append(
            f"<tr><td>{escape(row['sample_id'])}</td>"
            f"<td>{d0['localization']:.1%}</td><td>{candidate['localization']:.1%}</td>"
            f"<td>{d0['observation']:.1%}</td><td>{candidate['observation']:.1%}</td>"
            f"<td>{d0['median_error_px']:.1f}</td><td>{candidate['median_error_px']:.1f}</td>"
            f"<td>{row['local_reacquisitions']}</td><td>{row['abstentions']}</td><td>{links}</td></tr>"
        )
    gate = summary["gate"]
    gate_text = "통과" if gate["passed"] else "미통과"
    html = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARD100 로컬 패치 검증 실험</title><style>
body{{font:16px/1.55 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#17212b;background:#f4f6f7}}
h1,h2{{line-height:1.2}}section{{background:#fff;border:1px solid #d6dfe3;border-radius:12px;padding:20px 24px;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:14px;background:#fff}}th,td{{padding:8px;border-bottom:1px solid #dce3e7;text-align:right}}th:first-child,td:first-child{{text-align:left}}thead th{{background:#eaf0f2}}.status{{font-weight:700;color:{'#19743a' if gate['passed'] else '#9a3a25'}}}.small{{font-size:13px;color:#52616a}}
</style><body><h1>ARD100 로컬 패치 검증 실험</h1>
<section><p><b>결과: <span class="status">{gate_text}</span></b> (개발 train 영상 12개). 이 결과는 실험용 후보의 교차 영상 검증이며 운영 A 변경이 아닙니다.</p>
<p>다른 51개 ARD100 train 영상의 CVAT bbox에서 HOG 패치 분류기를 학습했습니다. 각 프레임에서 D0 위치를 먼저 확인하고, 점수가 음수일 때만 약 ±4개체 폭 주변에서 양수 후보를 찾습니다. D0 중심이 양수면 유지하고, 대체 후보도 음수면 해당 관측을 누락 처리합니다. GT는 평가에만 사용합니다.</p>
<p class="small">로컬 선형 SVM 점수는 확률이 아닙니다. 0 기준은 학습된 결정 경계이며 개발 영상에 맞춰 임계값을 조정하지 않았습니다. 개발 영상 12개는 학습에서 통째로 제외했습니다.</p></section>
<section><h2>전체 비교</h2><table><thead><tr><th>지표</th><th>D0</th><th>로컬 검증 후보</th></tr></thead><tbody>{metric_rows}</tbody></table>
<p>채택 게이트: 위치 성공률 상승, 관측률 하락 2%p 이내, 영상별 중앙 오차와 잔차 변화 중앙값이 D0보다 커지지 않아야 합니다. 연속 프레임 비교 비율도 함께 표시합니다.</p></section>
<section><h2>영상별 비교</h2><div style="overflow:auto"><table><thead><tr><th>영상</th><th>D0 위치</th><th>후보 위치</th><th>D0 관측</th><th>후보 관측</th><th>D0 오차</th><th>후보 오차</th><th>재획득</th><th>누락</th><th>자료</th></tr></thead><tbody>{''.join(videos)}</tbody></table></div></section>
<section><h2>학습·실행 기록</h2><p>학습 영상 {summary['training']['videos_with_patches']}개, 양성 패치 {summary['training']['positive_patches']}, 하드 음성 패치 {summary['training']['hard_negative_patches']}. 후보 궤적은 D0가 관측을 낸 프레임에서만 검증하며, D0 자체의 누락을 복구했다고 주장하지 않습니다.</p>
<p class="small">모델 SHA-256: {summary['training']['model_sha256']}<br>코드 SHA-256: {summary['source_hashes']['experiment_script']}<br>프로토콜: {escape(summary['protocol_name'])}</p></section></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def run(baseline: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment: {output}")
    base_protocol_path = baseline / "protocol.json"
    protocol = json.loads(base_protocol_path.read_text())
    dev_samples = _selected_dev_samples(protocol)
    dev_ids = {row["sample_id"] for row in dev_samples}
    output.mkdir(parents=True)
    started = time.monotonic()
    training = _fit_detector(protocol, dev_ids, output)
    bundle = joblib.load(training["model_path"])
    per_video = []
    for sample in dev_samples:
        print(f"evaluating {sample['sample_id']}", flush=True)
        result = _evaluate_sample(sample, baseline, output, bundle)
        baseline_metrics = _baseline_metric(sample["sample_id"], baseline)
        candidate_metrics = _metric_row(result)
        result["comparison_video"] = _render_comparison_video(sample, baseline, output, result)
        _write_json(output / "samples" / sample["sample_id"] / "result.json", result)
        per_video.append({
            "sample_id": sample["sample_id"],
            "status": result["status"],
            "D0": baseline_metrics,
            "verifier": candidate_metrics,
            "local_reacquisitions": result["local_reacquisitions"],
            "verified_d0_frames": result["verified_d0_frames"],
            "abstentions": result["abstentions"],
            "comparison_video": result["comparison_video"],
        })
    aggregates = {name: _aggregate(per_video, name) for name in ("D0", "verifier")}
    passed = (
        aggregates["verifier"]["mean_localization"] > aggregates["D0"]["mean_localization"]
        and aggregates["verifier"]["mean_observation"] >= aggregates["D0"]["mean_observation"] - 0.02
        and aggregates["verifier"]["median_video_median_error_px"] <= aggregates["D0"]["median_video_median_error_px"]
        and aggregates["verifier"]["median_video_jitter_px"] <= aggregates["D0"]["median_video_jitter_px"]
    )
    source_hashes = {
        "experiment_script": digest(Path(__file__)),
        "evaluator": digest(ROOT / "research/tracking_evaluation.py"),
        "frozen_protocol": digest(base_protocol_path),
    }
    output_protocol = {
        "protocol": "ARD100-local-patch-verifier-v1",
        "baseline": str(baseline.resolve()),
        "development_sample_ids": sorted(dev_ids),
        "training_excludes_development_videos": True,
        "training_box_sample_cap_per_video": TRAIN_BOXES_PER_VIDEO,
        "context_scale": CONTEXT_SCALE,
        "search_offsets": list(SEARCH_OFFSETS),
        "search_scales": list(SEARCH_SCALES),
        "acceptance_rule": "keep D0 if best-scale score >= 0; else accept local best if >= 0; otherwise abstain",
        "threshold_source": "fixed zero decision boundary of fitted LinearSVC; no dev tuning",
        "uses_ground_truth_after_initialization": False,
        "source_hashes": source_hashes,
    }
    summary = {
        "protocol_name": output_protocol["protocol"],
        "training": training,
        "source_hashes": source_hashes,
        "aggregate": aggregates,
        "gate": {
            "passed": passed,
            "localization_strictly_improved": aggregates["verifier"]["mean_localization"] > aggregates["D0"]["mean_localization"],
            "observation_drop_within_2pp": aggregates["verifier"]["mean_observation"] >= aggregates["D0"]["mean_observation"] - 0.02,
            "median_error_not_worse": aggregates["verifier"]["median_video_median_error_px"] <= aggregates["D0"]["median_video_median_error_px"],
            "jitter_not_worse": aggregates["verifier"]["median_video_jitter_px"] <= aggregates["D0"]["median_video_jitter_px"],
            "passed_dev_videos": len(dev_samples),
            "interpretation": "development gate only; passing does not establish production readiness",
        },
        "per_video": per_video,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    _write_json(output / "protocol.json", output_protocol)
    _write_json(output / "summary.json", summary)
    _write_report(output, summary)
    print(json.dumps({"aggregate": aggregates, "gate": summary["gate"], "output": str(output)}, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(args.baseline, args.output)


if __name__ == "__main__":
    cv2.setNumThreads(1)
    main()
