"""Probe OpenCV's online MIL single-object tracker against frozen D0 clips.

The tracker receives only the first annotated bounding box. All later GT is
used only for evaluation. This is a development experiment, not a production
tracker or a blind test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import subprocess
import sys
import time
import zlib

import cv2

from ai_server.services.manual_roi_tracker import _estimate_background_affine, _gray, _safe_inverse
from research.tracking_evaluation import evaluate_files
from ai_server.services.tracking_video_io import (
    affine_to_homography,
    iter_tracking_video_frames,
    read_tracking_video_metadata,
    transform_point,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/trajectory_experiments/ard100-frozen-20261018-v3"
OUTPUT_DEFAULT = ROOT / "artifacts/trajectory_experiments/ard100-mil-sot-probe-20261018"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def selected_samples(protocol: dict) -> list[dict]:
    eligible = [s for s in protocol["samples"] if s["split"] == "train" and s["status"] == "ready"]
    return eligible[::max(1, len(eligible) // 12)][:12]


def _write_mil_track(sample: dict, output: Path, stabilize: bool) -> dict:
    video_path = Path(sample["video"])
    gt_path = Path(sample["gt_xml"])
    if digest(video_path) != sample["video_sha256"] or digest(gt_path) != sample["gt_sha256"]:
        raise ValueError(f"frozen input hash changed for {sample['sample_id']}")

    metadata = read_tracking_video_metadata(video_path, resize_width=1280).to_dict()
    metadata["num_frames_loaded"] = sample["end_frame"] + 1
    metadata["num_frames_processed"] = sample["end_frame"] + 1
    metadata["tracking_start_frame"] = sample["init_frame_index"]
    metadata["attempted_frame_count"] = sample["end_frame"] - sample["init_frame_index"] + 1
    metadata["tracking_end_frame"] = sample["end_frame"]
    metadata["coordinate_mode"] = "camera_motion_compensated" if stabilize else "frame"
    metadata["tracker_method"] = "opencv_mil_sot_probe"
    metadata["tracking_parameters"] = {"tracker": "cv2.TrackerMIL_create", "update_policy": "single initialization; no GT after init", "stabilize": stabilize}

    scale_x = metadata["processed_width"] / metadata["width"]
    scale_y = metadata["processed_height"] / metadata["height"]
    x, y, width, height = sample["target_bbox"]
    init_bbox = tuple(int(round(value)) for value in (
        x * scale_x, y * scale_y, width * scale_x, height * scale_y,
    ))
    if min(init_bbox[2:]) < 2:
        raise ValueError(f"initial bbox too small after resize: {init_bbox}")

    tracker = cv2.TrackerMIL_create()
    observations = []
    import numpy as np
    ref_to_current = np.eye(3, dtype=np.float32)
    previous_gray = None
    previous_target_center = (init_bbox[0] + init_bbox[2] / 2, init_bbox[1] + init_bbox[3] / 2)
    previous_target_size = (float(init_bbox[2]), float(init_bbox[3]))
    started = time.monotonic()
    frame_limit = sample["end_frame"] + 1
    for frame_index, frame in iter_tracking_video_frames(video_path, resize_width=1280, max_frames=frame_limit):
        if frame_index < sample["init_frame_index"]:
            continue
        curr_gray = _gray(frame)
        if frame_index > sample["init_frame_index"] and stabilize and previous_gray is not None:
            affine, _ = _estimate_background_affine(
                previous_gray,
                curr_gray,
                target_center=previous_target_center,
                target_size=previous_target_size,
            )
            ref_to_current = affine_to_homography(affine) @ ref_to_current
            current_to_reference = _safe_inverse(ref_to_current)
            tracker_frame = cv2.warpAffine(
                frame,
                current_to_reference[:2],
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        else:
            current_to_reference = np.eye(3, dtype=np.float32)
            tracker_frame = frame
        if frame_index == sample["init_frame_index"]:
            ok = tracker.init(tracker_frame, init_bbox)
            if ok is False:
                raise RuntimeError("OpenCV MIL initialization failed")
            bbox = init_bbox
            success = True
        else:
            success, bbox = tracker.update(tracker_frame)
        previous_gray = curr_gray
        if not success:
            continue
        bx, by, bw, bh = (float(value) for value in bbox)
        if bw <= 0 or bh <= 0:
            continue
        if stabilize:
            raw_x, raw_y = transform_point(ref_to_current, bx + bw / 2, by + bh / 2)
            raw_corners = [
                transform_point(ref_to_current, px, py)
                for px, py in ((bx, by), (bx + bw, by), (bx, by + bh), (bx + bw, by + bh))
            ]
            raw_left = min(point[0] for point in raw_corners)
            raw_top = min(point[1] for point in raw_corners)
            raw_width = max(point[0] for point in raw_corners) - raw_left
            raw_height = max(point[1] for point in raw_corners) - raw_top
        else:
            raw_x, raw_y = bx + bw / 2, by + bh / 2
            raw_left, raw_top, raw_width, raw_height = bx, by, bw, bh
        previous_target_center = (raw_x, raw_y)
        previous_target_size = (bw, bh)
        observations.append({
            "frame_index": frame_index,
            "timestamp_ms": round(frame_index / metadata["fps"] * 1000),
            "raw_x": raw_x,
            "raw_y": raw_y,
            "compensated_x": bx + bw / 2,
            "compensated_y": by + bh / 2,
            "bbox_x": raw_left,
            "bbox_y": raw_top,
            "bbox_width": raw_width,
            "bbox_height": raw_height,
            "confidence": 1.0,
            "visible": True,
            "lifecycle_state": "ACTIVE",
            "tracking_source": "opencv_mil",
        })

    if not observations:
        raise ValueError("MIL produced no visible positions")
    folder = output / "samples" / sample["sample_id"]
    folder.mkdir(parents=True, exist_ok=True)
    metadata["visible_frame_count"] = len(observations)
    metadata["lost_frame_count"] = metadata["attempted_frame_count"] - len(observations)
    metadata["last_visible_frame"] = observations[-1]["frame_index"]
    metadata_path = folder / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    trajectory_path = folder / "trajectory.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)

    report = evaluate_files(
        cvat_xml=gt_path,
        trajectory_csv=trajectory_path,
        metadata_json=metadata_path,
        sample_id=sample["sample_id"],
        track_ids=(0,),
    )
    return {
        "sample_id": sample["sample_id"],
        "status": "completed",
        "visible_observations": len(observations),
        "attempted_frames": metadata["attempted_frame_count"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "tracking_evaluation": report,
        "artifacts": {"metadata": str(metadata_path), "trajectory": str(trajectory_path)},
    }


def _metrics(report: dict) -> dict:
    location = report["localization"]["normalized_threshold_success"]["within_0_5_bbox_diagonal"]
    coverage = report["coverage"]
    reference = report["reference_metrics"]
    return {
        "correct_localization_ratio": location["full_gt_ratio"],
        "observation_ratio": coverage["full_gt_observation_ratio"],
        "median_error_px_1280": reference["error_px"]["median"],
        "p95_error_px_1280": reference["error_px"]["p95"],
        "median_error_bbox_width": reference["error_bbox_width"]["median"],
        "jitter_median_px_1280": reference["error_change_px"]["median"],
    }


def _aggregate(rows: list[dict], label: str) -> dict:
    from statistics import mean, median

    values = [row[label] for row in rows]
    result = {"tracker": label, "videos": len(values)}
    for key in values[0]:
        items = [value[key] for value in values if value[key] is not None]
        result[key] = mean(items) if key in {"correct_localization_ratio", "observation_ratio"} else median(items) if items else None
    return result


def _write_report(output: Path, summary: dict) -> None:
    candidate = "MIL_CMC" if summary.get("mil_uses_cmc") else "MIL"
    d0, model = summary["summaries"]
    labels = {
        "correct_localization_ratio": "위치 성공률",
        "observation_ratio": "관측률",
        "median_error_px_1280": "오차 중앙값 (1280px 기준)",
        "p95_error_px_1280": "오차 P95 (1280px 기준)",
        "median_error_bbox_width": "객체 폭 대비 오차 중앙값",
        "jitter_median_px_1280": "프레임 간 잔차 중앙값",
    }
    metric_rows = []
    for key, label in labels.items():
        left, right = d0.get(key), model.get(key)
        def display(value):
            if value is None:
                return "시간 초과"
            return f"{value * 100:.2f}%" if "ratio" in key else f"{value:.3f}"
        metric_rows.append(f"<tr><th>{escape(label)}</th><td>{display(left)}</td><td>{display(right)}</td></tr>")

    video_rows = []
    for row in summary["per_video"]:
        baseline = row["D0"]
        proposed = row.get(candidate) or {}
        loc_a = baseline.get("correct_localization_ratio")
        loc_b = proposed.get("correct_localization_ratio")
        loc_delta = "-" if loc_a is None or loc_b is None else f"{(loc_b - loc_a) * 100:+.1f} pp"
        trajectory = output / "samples" / row["sample_id"] / "trajectory.csv"
        trajectory_cell = (
            f"<a href='samples/{escape(row['sample_id'])}/trajectory.csv'>궤적 CSV</a>"
            if trajectory.exists() else "궤적 미생성"
        )
        video_rows.append(
            "<tr>"
            f"<td>{escape(row['sample_id'])}</td><td>{escape(row['status'])}</td>"
            f"<td>{loc_a * 100:.1f}%</td><td>{loc_b * 100:.1f}%</td><td>{loc_delta}</td>"
            f"<td>{baseline.get('median_error_px_1280') if baseline.get('median_error_px_1280') is not None else '-'}</td>"
            f"<td>{proposed.get('median_error_px_1280') if proposed.get('median_error_px_1280') is not None else '-'}</td>"
            f"<td>{trajectory_cell}</td></tr>"
        )

    html = f"""<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARD100 MIL 추적기 비교</title>
<style>
body{{font:16px/1.55 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;margin:36px auto;max-width:1100px;padding:0 20px;color:#17212b;background:#f5f7f8}}
h1,h2{{line-height:1.2}} .card{{background:#fff;border:1px solid #d9e1e5;border-radius:10px;padding:18px 22px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px;background:white}} th,td{{padding:9px;border-bottom:1px solid #dce3e7;text-align:right}} th:first-child,td:first-child{{text-align:left}}
thead th{{background:#e8eef1}} .bad{{color:#a32b21;font-weight:700}} .note{{color:#42515a}}
</style><body>
<h1>ARD100 온라인 MIL 추적기 비교</h1>
<p class="note">12개 개발 클립의 첫 10초. 최초 GT bbox만 초기화에 사용하고, 이후 GT는 평가에만 사용했습니다.</p>
<section class="card"><h2>전체 결과</h2><p class="bad">판정: MIL 후보 미채택. 위치 성공률과 시간 안정성 기준을 통과하지 못했습니다.</p>
<p>D0: {d0['correct_localization_ratio']*100:.2f}% 위치 성공률, {d0['observation_ratio']*100:.2f}% 관측률<br>
{escape(candidate)}: {model['correct_localization_ratio']*100:.2f}% 위치 성공률, {model['observation_ratio']*100:.2f}% 관측률<br>
MIL 위치 성공률에는 시간 초과 영상이 실패(0)로 포함됩니다.</p>
<table><thead><tr><th>지표</th><th>D0</th><th>{escape(candidate)}</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></section>
<section class="card"><h2>영상별</h2><table><thead><tr><th>영상</th><th>상태</th><th>D0 위치</th><th>MIL 위치</th><th>차이</th><th>D0 오차 중앙</th><th>MIL 오차 중앙</th><th>자료</th></tr></thead><tbody>{''.join(video_rows)}</tbody></table></section>
<section class="card"><h2>해석과 한계</h2><ul>
<li>관측률은 높았지만, 위치가 정답 객체를 따라간다는 뜻은 아닙니다.</li>
<li>5×3px 수준의 초기 ROI에서 MIL이 45초 제한을 넘었습니다.</li>
<li>CMC 포함 후보는 누적 보정으로 큰 오차가 난 영상이 있었고, 프레임 간 잔차 중앙값도 D0보다 커졌습니다.</li>
<li>12개는 개발 자료입니다. 이 후보를 전체 train/test로 확장하거나 C 분류에 연결하지 않았습니다.</li>
<li>OpenCV MIL은 고전 온라인 SOT이며, 최신 학습 기반 탐지기/추적기 비교 결과를 뜻하지 않습니다.</li>
</ul><p>세부 결과: <a href="comparison.csv">comparison.csv</a>, <a href="summary.json">summary.json</a>, <a href="protocol.json">protocol.json</a></p></section>
</body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def worker(baseline: Path, output: Path, sample_id: str, stabilize: bool) -> dict:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(zlib.crc32(sample_id.encode("utf-8")) & 0x7FFFFFFF)
    protocol = json.loads((baseline / "protocol.json").read_text())
    sample = next(s for s in selected_samples(protocol) if s["sample_id"] == sample_id)
    result_path = output / "samples" / sample_id / "result.json"
    try:
        result = _write_mil_track(sample, output, stabilize)
        (output / "samples" / sample_id / "mil_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2))
        d0_result_path = baseline / "samples" / sample["sample_id"] / "result.json"
        d0_result = json.loads(d0_result_path.read_text())
        d0_report = d0_result["tracking_evaluation"]
        d0_values = _metrics(d0_report)
        mil_values = _metrics(result["tracking_evaluation"])
        candidate = "MIL_CMC" if stabilize else "MIL"
        row = {"sample_id": sample_id, "status": "completed", "D0": d0_values, candidate: mil_values}
    except Exception as error:
        row = {"sample_id": sample_id, "status": "failed", "reason": str(error)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(row, ensure_ascii=False, indent=2))
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def run(baseline: Path, output: Path, timeout_seconds: float = 45.0, stabilize: bool = False) -> dict:
    protocol = json.loads((baseline / "protocol.json").read_text())
    samples = selected_samples(protocol)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps({
        "protocol": "ARD100-MIL-probe-v3",
        "baseline": str(baseline.resolve()),
        "selection": "same lexicographic-stride 12 ready train clips as prior ARD development subset",
        "initialization": "first annotated full-object bbox only",
        "post_initialization_gt": "evaluation only; never passed to MIL",
        "coordinate_space": "raw processed video frame; MIL and D0 evaluated in source pixels",
        "d0_uses_cmc_for_internal_prediction": True,
        "mil_uses_cmc": stabilize,
        "mil_candidate": "MIL_CMC" if stabilize else "MIL",
        "tracker": "OpenCV TrackerMIL_create",
        "per_sample_timeout_seconds": timeout_seconds,
        "timeout_policy": "timed-out videos remain in localization and observation denominators with zero success",
        "samples": [{k: s[k] for k in ("sample_id", "video", "video_sha256", "gt_xml", "gt_sha256", "target_bbox", "init_frame_index", "end_frame", "fps", "width", "height")} for s in samples],
    }, ensure_ascii=False, indent=2))

    rows = []
    for sample in samples:
        print(f"MIL {sample['sample_id']} starting", flush=True)
        command = [sys.executable, "-m", "research.ard100_sot_probe", "--worker-sample-id", sample["sample_id"],
                   "--baseline", str(baseline), "--output", str(output)]
        if stabilize:
            command.append("--stabilize")
        try:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                       timeout=timeout_seconds, check=False)
            if completed.returncode == 0:
                row = json.loads((output / "samples" / sample["sample_id"] / "result.json").read_text())
            else:
                row = {"sample_id": sample["sample_id"], "status": "failed",
                       "reason": completed.stderr[-2000:] or completed.stdout[-2000:]}
        except subprocess.TimeoutExpired:
            row = {"sample_id": sample["sample_id"], "status": "timeout",
                   "reason": f"exceeded {timeout_seconds:.0f}s per-sample limit"}
        if row["status"] != "completed":
            d0_result = json.loads((baseline / "samples" / sample["sample_id"] / "result.json").read_text())
            row["D0"] = _metrics(d0_result["tracking_evaluation"])
            candidate = "MIL_CMC" if stabilize else "MIL"
            row[candidate] = {"correct_localization_ratio": 0.0, "observation_ratio": 0.0,
                          "median_error_px_1280": None, "p95_error_px_1280": None,
                          "median_error_bbox_width": None, "jitter_median_px_1280": None}
            (output / "samples" / sample["sample_id"]).mkdir(parents=True, exist_ok=True)
            (output / "samples" / sample["sample_id"] / "result.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2))
        rows.append(row)
        print(f"MIL {sample['sample_id']} {row['status']}", flush=True)

    candidate = "MIL_CMC" if stabilize else "MIL"
    summaries = [_aggregate(rows, key) for key in ("D0", candidate)]
    summary = {
        "protocol": "ARD100-MIL-probe-v3",
        "sample_count": len(rows),
        "mil_uses_cmc": stabilize,
        "mil_rng_seed": "CRC32 of sample_id, fixed independently per clip",
        "completed_count": sum(row["status"] == "completed" for row in rows),
        "timed_out_count": sum(row["status"] == "timeout" for row in rows),
        "failed_count": sum(row["status"] == "failed" for row in rows),
        "summaries": summaries,
        "per_video": rows,
        "limitations": [
            "12 development clips only; no candidate has proceeded to full-train or test validation.",
            "MIL is an installed classical OpenCV tracker, not a modern transformer SOT.",
            "D0 and MIL use the same CMC setting for this run; both outputs are scored in source video coordinates.",
            "GT is used only to initialize the first bbox and evaluate later frames.",
            "Timeouts and failures count as zero localization and zero observation in macro averages.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    _write_report(output, summary)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ["sample_id", "tracker", "correct_localization_ratio", "observation_ratio", "median_error_px_1280", "p95_error_px_1280", "median_error_bbox_width", "jitter_median_px_1280"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            for tracker in ("D0", candidate):
                writer.writerow({"sample_id": row["sample_id"], "tracker": tracker, **row[tracker]})
    print(json.dumps({"output": str(output), "summaries": summaries}, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--worker-sample-id")
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--stabilize", action="store_true")
    args = parser.parse_args()
    if args.worker_sample_id:
        worker(args.baseline.resolve(), args.output.resolve(), args.worker_sample_id, args.stabilize)
    else:
        run(args.baseline.resolve(), args.output.resolve(), args.timeout_seconds, args.stabilize)


if __name__ == "__main__":
    main()
