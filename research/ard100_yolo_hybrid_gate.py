"""Calibrate a conservative D0 + local-YOLO fallback on source-held-out clips."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import time
from html import escape

import cv2
import numpy as np

from ai_server.services.tracking_video_io import iter_tracking_video_frames
from research.ard100_sot_probe import digest
from research.tracking_evaluation import evaluate_files
from research.ard100_yolo_gap_reacquisition import (
    CONFIRM_DISTANCE_PX,
    MAX_GAP_FRAMES,
    _select_detection,
)
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
    _selected_dev_samples,
    _write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DETECTOR_RUN = ROOT / "artifacts/trajectory_experiments/ard100-yolo-local-reacquisition-20261018-v2"
GAP_RUN = ROOT / "artifacts/trajectory_experiments/ard100-yolo-gap-reacquisition-20261018-v3"
OUTPUT_DEFAULT = ROOT / "artifacts/trajectory_experiments/ard100-yolo-hybrid-gate-20261018-v1"
D0_CONFIDENCE_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
MIN_VALIDATION_LOCALIZATION_GAIN = 0.01


def _load_baseline(sample: dict, baseline: Path) -> tuple[list[dict], dict, Path]:
    result = json.loads((baseline / "samples" / sample["sample_id"] / "result.json").read_text())
    artifacts = result["tracking_artifacts"]
    rows = _read_d0(Path(artifacts["trajectory"]))
    metadata = json.loads(Path(artifacts["metadata"]).read_text())
    return rows, metadata, Path(artifacts["trajectory"])


def _candidate_anchors(sample: dict, rows: list[dict], metadata: dict) -> dict[int, dict]:
    visible = {
        int(row["frame_index"]): row
        for row in rows
        if str(row.get("visible", "false")).lower() == "true"
    }
    end_frame = min(int(sample["end_frame"]), int(metadata["attempted_frame_count"]) - 1)
    anchors: dict[int, dict] = {}
    history: list[tuple[int, float, float, dict]] = []
    for frame_index in range(int(sample.get("init_frame_index", 0)), end_frame + 1):
        observed = visible.get(frame_index)
        if observed is not None:
            x, y = float(observed["raw_x"]), float(observed["raw_y"])
            history.append((frame_index, x, y, observed))
            history = history[-2:]
            anchors[frame_index] = {"row": dict(observed), "x": x, "y": y, "gap": False}
            continue
        if not history:
            continue
        gap = frame_index - history[-1][0]
        if gap < 1 or gap > MAX_GAP_FRAMES:
            continue
        last_frame, last_x, last_y, last_row = history[-1]
        if len(history) == 2:
            previous_frame, previous_x, previous_y, _ = history[-2]
            delta = max(1, last_frame - previous_frame)
            vx, vy = (last_x - previous_x) / delta, (last_y - previous_y) / delta
        else:
            vx = vy = 0.0
        x, y = last_x + vx * gap, last_y + vy * gap
        base_row = dict(last_row)
        base_row.update(
            frame_index=frame_index,
            timestamp_ms=int(round(frame_index * 1000 / float(sample["fps"]))),
            raw_x=x,
            raw_y=y,
            compensated_x=x,
            compensated_y=y,
            bbox_x=x - float(last_row.get("bbox_width", 0) or 0) / 2,
            bbox_y=y - float(last_row.get("bbox_height", 0) or 0) / 2,
            visible=False,
            lifecycle_state="LOST",
            tracking_source="constant_velocity_gap_prediction",
        )
        anchors[frame_index] = {"row": base_row, "x": x, "y": y, "gap": True}
    return anchors


def _detection_targets(anchors: dict[int, dict], rows: list[dict]) -> set[int]:
    visible_confidence = {
        int(row["frame_index"]): float(row.get("confidence", 0) or 0)
        for row in rows
        if str(row.get("visible", "false")).lower() == "true"
    }
    suspect = {
        frame for frame, confidence in visible_confidence.items()
        if confidence < max(D0_CONFIDENCE_GRID)
    }
    suspect.update(frame for frame, anchor in anchors.items() if anchor["gap"])
    targets = set()
    for frame in suspect:
        targets.update((frame - 1, frame, frame + 1))
    return targets.intersection(anchors)


def _infer_candidates(sample: dict, rows: list[dict], metadata: dict, model, threshold: float) -> tuple[dict, dict]:
    anchors = _candidate_anchors(sample, rows, metadata)
    target_frames = _detection_targets(anchors, rows)
    detections: dict[int, dict | None] = {}
    pending: list[tuple[int, np.ndarray, int, int, float, float]] = []

    def flush() -> None:
        if not pending:
            return
        predictions = model.predict(
            source=[item[1] for item in pending], imgsz=IMAGE_SIZE,
            conf=0.001, device="cpu", verbose=False,
        )
        for item, prediction in zip(pending, predictions):
            frame_index, _, left, top, x, y = item
            detections[frame_index] = _select_detection(prediction, (left, top), (x, y), threshold)
        pending.clear()

    cap_end = max(anchors) if anchors else -1
    for frame_index, frame in iter_tracking_video_frames(
        sample["video"], resize_width=FRAME_WIDTH, max_frames=cap_end + 1,
    ):
        if frame_index not in target_frames:
            continue
        anchor = anchors[frame_index]
        patch, left, top = _crop(frame, anchor["x"], anchor["y"])
        pending.append((frame_index, patch, left, top, anchor["x"], anchor["y"]))
        if len(pending) >= 24:
            flush()
    flush()

    supported: set[int] = set()
    for frame_index, detection in detections.items():
        if not detection or not detection["accepted"]:
            continue
        for neighbor_index in (frame_index - 1, frame_index + 1):
            neighbor = detections.get(neighbor_index)
            if not neighbor or not neighbor["accepted"]:
                continue
            if math.hypot(detection["cx"] - neighbor["cx"], detection["cy"] - neighbor["cy"]) <= CONFIRM_DISTANCE_PX:
                supported.update((frame_index, neighbor_index))
    return anchors, {
        "detections": detections,
        "supported": supported,
        "detector_threshold": threshold,
    }


def _candidate_rows(
    rows: list[dict], anchors: dict[int, dict], evidence: dict, threshold: float,
) -> tuple[list[dict], dict]:
    source_by_frame = {int(row["frame_index"]): row for row in rows}
    output = []
    replaced = recovered = accepted = 0
    for frame_index in sorted(set(source_by_frame) | {
        index for index, anchor in anchors.items() if anchor["gap"] and index in evidence["supported"]
    }):
        source = source_by_frame.get(frame_index)
        anchor = anchors.get(frame_index)
        if source is not None:
            row = dict(source)
            confidence = float(source.get("confidence", 0) or 0)
            should_try = confidence < threshold
        else:
            row = dict(anchor["row"])
            should_try = bool(anchor and anchor["gap"])
        detection = evidence["detections"].get(frame_index)
        use_detection = bool(
            should_try and detection and detection["accepted"]
            and frame_index in evidence["supported"]
        )
        row["detector_confidence"] = detection["confidence"] if detection else ""
        row["detector_dx"] = detection["cx"] - anchor["x"] if detection and anchor else ""
        row["detector_dy"] = detection["cy"] - anchor["y"] if detection and anchor else ""
        row["detector_accepted"] = use_detection
        if use_detection:
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
                tracking_source="yolo_hybrid_gap_reacquisition" if source is None else "yolo_hybrid_low_confidence",
            )
            accepted += 1
            recovered += int(source is None)
            replaced += int(source is not None)
        elif source is None:
            continue
        output.append(row)
    return output, {"detector_accepted": accepted, "pointwise_replacements": replaced, "gap_recoveries": recovered}


def _evaluate_candidate(sample: dict, rows: list[dict], metadata: dict, anchors: dict, evidence: dict,
                        threshold: float, output_dir: Path) -> dict:
    candidate, counts = _candidate_rows(rows, anchors, evidence, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trajectory.csv"
    fields = list(dict.fromkeys([
        *(rows[0].keys() if rows else ()),
        "detector_confidence", "detector_dx", "detector_dy", "detector_accepted",
    ]))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate)
    out_metadata = dict(metadata)
    out_metadata["tracker_method"] = "ard100_d0_yolo_hybrid_validation_candidate"
    out_metadata["tracking_parameters"] = {
        "base_tracker": "frozen D0",
        "local_detector": "frozen ARD100-train YOLO11n",
        "d0_confidence_threshold": threshold,
        "detector_confidence_threshold": evidence["detector_threshold"],
        "max_gap_frames": MAX_GAP_FRAMES,
        "gap_motion_model": "constant velocity from last two D0 observations",
        "confirmation": f"two adjacent accepted detections within {CONFIRM_DISTANCE_PX:.1f}px",
        "gt_used_after_initialization": False,
    }
    out_metadata["visible_frame_count"] = sum(str(row.get("visible", "false")).lower() == "true" for row in candidate)
    out_metadata["lost_frame_count"] = max(0, int(out_metadata["attempted_frame_count"]) - out_metadata["visible_frame_count"])
    metadata_path = output_dir / "metadata.json"
    _write_json(metadata_path, out_metadata)
    evaluation = evaluate_files(
        cvat_xml=sample["gt_xml"], trajectory_csv=csv_path, metadata_json=metadata_path,
        sample_id=sample["sample_id"], track_ids=(0,),
    )
    _write_json(output_dir / "tracking_evaluation.json", evaluation)
    result = {
        "sample_id": sample["sample_id"],
        "status": "completed",
        **counts,
        "d0_confidence_threshold": threshold,
        "tracking_evaluation": evaluation,
        "trajectory": str(csv_path),
        "metadata": str(metadata_path),
    }
    _write_json(output_dir / "result.json", result)
    return result


def _passes_calibration(candidate: dict, baseline: dict) -> bool:
    return (
        candidate["mean_localization"] >= baseline["mean_localization"] + MIN_VALIDATION_LOCALIZATION_GAIN
        and candidate["mean_observation"] >= baseline["mean_observation"] - 0.02
        and candidate["median_video_median_error_px"] <= baseline["median_video_median_error_px"]
        and candidate["median_video_jitter_px"] <= baseline["median_video_jitter_px"]
    )


def _select_threshold(results: dict[float, dict], baseline: dict) -> tuple[float, dict]:
    eligible = [(threshold, row) for threshold, row in results.items() if _passes_calibration(row, baseline)]
    if not eligible:
        return 0.0, {
            "selected": False,
            "reason": "no pointwise threshold met validation guardrails; disable switches and retain corroborated gap recovery only",
        }
    threshold, metrics = max(
        eligible,
        key=lambda item: (item[1]["mean_localization"], -item[1]["median_video_jitter_px"], -item[0]),
    )
    return threshold, {"selected": True, "reason": "best localization among thresholds meeting all validation guardrails"}


def _write_report(output: Path, summary: dict) -> None:
    validation_rows = "".join(
        f"<tr><td>{row['threshold']:.1f}</td><td>{row['metrics']['mean_localization']:.1%}</td>"
        f"<td>{row['metrics']['mean_observation']:.1%}</td><td>{row['metrics']['median_video_median_error_px']:.1f}</td>"
        f"<td>{row['metrics']['median_video_jitter_px']:.2f}</td><td>{'통과' if row['eligible'] else '미통과'}</td></tr>"
        for row in summary["validation_thresholds"]
    )
    dev_rows = "".join(
        f"<tr><td>{escape(row['sample_id'])}</td><td>{row['D0']['localization']:.1%}</td>"
        f"<td>{row['YOLO-every-frame+gap']['localization']:.1%}</td><td>{row['Hybrid']['localization']:.1%}</td>"
        f"<td>{row['D0']['observation']:.1%}</td><td>{row['Hybrid']['observation']:.1%}</td>"
        f"<td>{row['D0']['median_error_px']:.1f}</td><td>{row['Hybrid']['median_error_px']:.1f}</td>"
        f"<td>{row['pointwise_replacements']}</td><td>{row['gap_recoveries']}</td></tr>"
        for row in summary["development_per_video"]
    )
    aggregate_rows = "".join(
        f"<tr><th>{label}</th><td>{summary['development_aggregate'][kind][key]:.4f}</td></tr>"
        for kind in ("D0", "YOLO-every-frame+gap", "Hybrid")
        for key, label in (("mean_localization", f"{kind}: 위치 성공률"), ("mean_observation", f"{kind}: 관측률"),
                           ("median_video_median_error_px", f"{kind}: 영상별 중앙 오차 중앙값(px)"),
                           ("median_video_jitter_px", f"{kind}: 잔차 변화 중앙값(px)"))
    )
    chosen = summary["calibration"]
    gate = summary["development_gate"]
    html = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>D0 + YOLO 하이브리드 검증</title><style>
body{{font:16px/1.55 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;max-width:1240px;margin:32px auto;padding:0 20px;color:#17212b;background:#f4f6f7}}h1,h2{{line-height:1.2}}section{{background:#fff;border:1px solid #d6dfe3;border-radius:12px;padding:20px 24px;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:14px;background:#fff}}th,td{{padding:8px;border-bottom:1px solid #dce3e7;text-align:right}}th:first-child,td:first-child{{text-align:left}}thead th{{background:#eaf0f2}}.status{{font-weight:700;color:{'#19743a' if gate['passed'] else '#9a3a25'}}}.small{{font-size:13px;color:#52616a}}
</style><body><h1>D0 유지형 YOLO 하이브리드 실험</h1>
<section><p><b>개발 게이트: <span class="status">{'통과' if gate['passed'] else '미통과'}</span></b>. 운영 A 추적기는 변경하지 않았습니다.</p>
<p>D0는 기본 좌표를 계속 제공하고, YOLO는 D0 confidence가 검증에서 선택한 문턱 미만이거나 D0 궤적이 끊긴 경우에만 후보를 냅니다. 검출은 기존의 별도 영상 검증에서 정한 confidence를 넘고, 인접 프레임 검출과 24px 이내로 이어질 때만 사용합니다. 인접 프레임 확인은 다음 프레임 결과가 나온 뒤 확정하는 1프레임 지연 구조입니다. 누락은 직전 두 D0 관측으로 최대 {MAX_GAP_FRAMES}프레임까지 예측한 위치에서 국소 검색합니다.</p>
<p>독립된 원본 영상 검증 {len(summary['validation_sample_ids'])}개에서 D0 confidence 문턱을 선택하고, 고정 개발 영상 {len(summary['development_sample_ids'])}개에 한 번 적용했습니다. 개발 영상은 학습과 문턱 선택에서 제외했습니다.</p>
<p><b>선택 문턱:</b> {chosen['d0_confidence_threshold']:.1f}. {escape(chosen['reason'])}</p>
<p class="small">위치 성공률은 GT bbox 대각선 절반 이내의 비율입니다. 문턱 선택 시 관측률은 D0 대비 2%p 이내, 위치 성공률은 1%p 이상 개선, 영상별 중앙 오차와 떨림은 악화하지 않는 조건을 사전 적용했습니다.</p></section>
<section><h2>검증 세트에서의 문턱 선택</h2><table><thead><tr><th>D0 conf 미만</th><th>위치 성공률</th><th>관측률</th><th>중앙 오차(px)</th><th>떨림(px)</th><th>기준</th></tr></thead><tbody>{validation_rows}</tbody></table></section>
<section><h2>고정 개발 세트 비교</h2><table><thead><tr><th>측정</th><th>값</th></tr></thead><tbody>{aggregate_rows}</tbody></table><p>통과 조건은 D0 대비 위치 성공률 상승, 관측률 2%p 이내, 영상별 중앙 오차와 떨림 비악화입니다. 이 파일럿 통과는 운영 준비 완료를 뜻하지 않습니다.</p></section>
<section><h2>영상별 결과</h2><div style="overflow:auto"><table><thead><tr><th>영상</th><th>D0 위치</th><th>기존 YOLO 전 프레임+gap 위치</th><th>하이브리드 위치</th><th>D0 관측</th><th>하이브리드 관측</th><th>D0 오차</th><th>하이브리드 오차</th><th>전환</th><th>복구</th></tr></thead><tbody>{dev_rows}</tbody></table></div></section>
<section><h2>재현 정보</h2><p>YOLO 가중치 SHA-256: {summary['source_hashes']['weights']}<br>고정 프로토콜 SHA-256: {summary['source_hashes']['frozen_protocol']}<br>실험 스크립트 SHA-256: {summary['source_hashes']['experiment_script']}</p></section></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def run(baseline: Path, detector_run: Path, gap_run: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment: {output}")
    output.mkdir(parents=True)
    detector_summary = json.loads((detector_run / "summary.json").read_text())
    weights = Path(detector_summary["training"]["weights"])
    if not weights.is_file():
        raise FileNotFoundError(f"frozen YOLO weights not found: {weights}")
    detector_threshold = float(detector_summary["calibration"]["confidence_threshold"])
    protocol_path = baseline / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    validation_ids = detector_summary["dataset"]["validation_video_ids"]
    development = _selected_dev_samples(protocol)
    dev_ids = {sample["sample_id"] for sample in development}
    validation = [
        sample for sample in protocol["samples"]
        if sample["sample_id"] in validation_ids and sample["status"] == "ready"
    ]
    if len(validation) != len(validation_ids):
        raise ValueError("some frozen validation videos are missing from the baseline protocol")
    if dev_ids.intersection(validation_ids):
        raise ValueError("development and validation source videos overlap")

    config_dir = output / "ultralytics_config"
    (config_dir / "Ultralytics").mkdir(parents=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    cv2.setNumThreads(1)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    started = time.monotonic()
    validation_evidence = {}
    validation_rows = []
    for sample in validation:
        print(f"calibration evidence {sample['sample_id']}", flush=True)
        rows, metadata, _ = _load_baseline(sample, baseline)
        anchors, evidence = _infer_candidates(sample, rows, metadata, model, detector_threshold)
        validation_evidence[sample["sample_id"]] = (sample, rows, metadata, anchors, evidence)

    validation_metrics: dict[float, dict] = {}
    validation_by_threshold = {}
    for threshold in D0_CONFIDENCE_GRID:
        per_video = []
        for sample_id, (sample, rows, metadata, anchors, evidence) in validation_evidence.items():
            folder = output / "validation" / f"threshold_{threshold:.1f}" / "samples" / sample_id
            result = _evaluate_candidate(sample, rows, metadata, anchors, evidence, threshold, folder)
            per_video.append({"candidate": _metric_row(result["tracking_evaluation"])})
        aggregate = _aggregate([
            {"candidate": item["candidate"]} for item in per_video
        ], "candidate")
        validation_metrics[threshold] = aggregate
        validation_by_threshold[threshold] = per_video
        validation_rows.append({
            "threshold": threshold,
            "metrics": aggregate,
            "eligible": False,
        })

    validation_baseline = _aggregate([
        {"D0": _baseline_metric(sample["sample_id"], baseline)} for sample in validation
    ], "D0")
    for row in validation_rows:
        row["eligible"] = _passes_calibration(validation_metrics[row["threshold"]], validation_baseline)
    selected_threshold, selection = _select_threshold(validation_metrics, validation_baseline)

    development_per_video = []
    for sample in development:
        sample_id = sample["sample_id"]
        print(f"fixed development {sample_id}", flush=True)
        rows, metadata, _ = _load_baseline(sample, baseline)
        anchors, evidence = _infer_candidates(sample, rows, metadata, model, detector_threshold)
        candidate_dir = output / "samples" / sample_id
        result = _evaluate_candidate(sample, rows, metadata, anchors, evidence, selected_threshold, candidate_dir)
        d0_metric = _baseline_metric(sample_id, baseline)
        gap_eval_path = gap_run / "samples" / sample_id / "tracking_evaluation.json"
        gap_metric = _metric_row(json.loads(gap_eval_path.read_text())) if gap_eval_path.exists() else d0_metric
        development_per_video.append({
            "sample_id": sample_id,
            "D0": d0_metric,
            "YOLO-every-frame+gap": gap_metric,
            "Hybrid": _metric_row(result["tracking_evaluation"]),
            "pointwise_replacements": result["pointwise_replacements"],
            "gap_recoveries": result["gap_recoveries"],
            "detector_accepted": result["detector_accepted"],
        })
    aggregate = {
        name: _aggregate([{"metric": row[name]} for row in development_per_video], "metric")
        for name in ("D0", "YOLO-every-frame+gap", "Hybrid")
    }
    base, candidate = aggregate["D0"], aggregate["Hybrid"]
    dev_gate = {
        "passed": (
            candidate["mean_localization"] > base["mean_localization"]
            and candidate["mean_observation"] >= base["mean_observation"] - 0.02
            and candidate["median_video_median_error_px"] <= base["median_video_median_error_px"]
            and candidate["median_video_jitter_px"] <= base["median_video_jitter_px"]
        ),
        "interpretation": "fixed development gate only; not a production-readiness claim",
    }
    summary = {
        "protocol": "ARD100-D0-YOLO-conservative-hybrid-v1",
        "validation_sample_ids": validation_ids,
        "development_sample_ids": sorted(dev_ids),
        "detector_threshold": detector_threshold,
        "validation_baseline": validation_baseline,
        "validation_thresholds": validation_rows,
        "calibration": {"d0_confidence_threshold": selected_threshold, **selection},
        "development_aggregate": aggregate,
        "development_per_video": development_per_video,
        "development_gate": dev_gate,
        "source_hashes": {
            "experiment_script": digest(Path(__file__)),
            "frozen_protocol": digest(protocol_path),
            "evaluator": digest(ROOT / "research/tracking_evaluation.py"),
            "weights": detector_summary["training"]["weights_sha256"],
            "detector_run_summary": digest(detector_run / "summary.json"),
        },
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    _write_json(output / "protocol.json", {
        "protocol": summary["protocol"],
        "baseline": str(baseline.resolve()),
        "detector_run": str(detector_run.resolve()),
        "gap_only_run": str(gap_run.resolve()),
        "validation_sample_ids": validation_ids,
        "development_sample_ids": sorted(dev_ids),
        "d0_confidence_grid": D0_CONFIDENCE_GRID,
        "minimum_validation_localization_gain": MIN_VALIDATION_LOCALIZATION_GAIN,
        "guardrails": "observation within 2pp, median video error and jitter not worse",
        "detector_threshold": detector_threshold,
        "selected_d0_confidence_threshold": selected_threshold,
        "uses_ground_truth_after_initialization": False,
        "source_hashes": summary["source_hashes"],
    })
    _write_json(output / "summary.json", summary)
    _write_report(output, summary)
    print(json.dumps({
        "calibration": summary["calibration"],
        "validation_baseline": validation_baseline,
        "development_aggregate": aggregate,
        "development_gate": dev_gate,
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--detector-run", type=Path, default=DETECTOR_RUN)
    parser.add_argument("--gap-run", type=Path, default=GAP_RUN)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(args.baseline, args.detector_run, args.gap_run, args.output)


if __name__ == "__main__":
    main()
