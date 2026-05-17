from __future__ import annotations

import argparse
from pathlib import Path
import sys
from traceback import format_exception_only

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_server.schemas import StabilizationInfo, StabilizationMethod
from ai_server.services.detector_eval import (
    DEFAULT_DATASET_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VIDEO_CASES,
    DetectorAdapter,
    OpenCVMotionDebugAdapter,
    UltralyticsYOLOAdapter,
    YOLOMGAdapter,
    build_manifest,
    build_tracks_for_tracker,
    iter_limited_video_frames,
    max_frames_for_case,
    sanitize_id,
    summarize_failure,
    summarize_run,
    write_detection_json,
    write_json,
    write_summary_csv,
    write_track_visualization,
    write_tracks_json,
)
from ai_server.services.stabilization import run_stabilization_stage
from ai_server.services.video_io import load_video_metadata


def main() -> int:
    args = _parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(dataset_dir=dataset_dir)
    write_json(output_root / "manifest.json", manifest)

    detectors = _build_detectors(args)
    rows: list[dict[str, object]] = []

    cases = _select_cases(args)
    for case in cases:
        input_path = case.path(dataset_dir)
        if not input_path.exists():
            rows.extend(
                _failure_rows_for_case(
                    case=case,
                    detectors=detectors,
                    trackers=args.trackers,
                    methods=args.stabilization_methods,
                    error=f"Video file does not exist: {input_path}",
                    start_sec=args.start_sec,
                )
            )
            continue

        for method in args.stabilization_methods:
            try:
                stabilization_result = run_stabilization_stage(
                    video_path=str(input_path),
                    method=method,
                )
                eval_video_path = stabilization_result.stabilized_video_path
                stabilization = stabilization_result.stabilization
                metadata = load_video_metadata(eval_video_path)
                max_frames = max_frames_for_case(
                    metadata,
                    case,
                    args.eval_clip_sec,
                )
            except Exception as exc:
                rows.extend(
                    _failure_rows_for_case(
                        case=case,
                        detectors=detectors,
                        trackers=args.trackers,
                        methods=(method,),
                        error=_short_error(exc),
                        start_sec=args.start_sec,
                    )
                )
                continue

            for detector in detectors:
                run_key_base = sanitize_id(
                    f"{case.video_id}{_start_suffix(args.start_sec)}__{method}__{detector.name}"
                )
                try:
                    frames = iter_limited_video_frames(
                        eval_video_path,
                        max_frames,
                        start_sec=args.start_sec,
                    )
                    frame_detections = list(detector.detect_frames(frames))
                    detections_path = output_root / "detections" / f"{run_key_base}.json"
                    write_detection_json(
                        detections_path,
                        source_video_id=run_key_base,
                        detector_name=detector.name,
                        frame_detections=frame_detections,
                    )
                except Exception as exc:
                    for tracker_name in args.trackers:
                        rows.append(
                            summarize_failure(
                                case=case,
                                detector_name=detector.name,
                                tracker_name=tracker_name,
                                stabilization=stabilization,
                                error=_short_error(exc),
                                start_sec=args.start_sec,
                            )
                        )
                    continue

                for tracker_name in args.trackers:
                    run_key = sanitize_id(f"{run_key_base}__{tracker_name}")
                    try:
                        tracks = build_tracks_for_tracker(
                            tracker_name=tracker_name,
                            frame_detections=frame_detections,
                            metadata=metadata,
                            source_video_id=run_key,
                            stabilization=stabilization,
                        )
                        write_tracks_json(
                            output_root / "tracks" / f"{run_key}.json",
                            tracks,
                        )
                        if not args.skip_viz:
                            write_track_visualization(
                                input_video_path=eval_video_path,
                                output_video_path=output_root / "viz" / f"{run_key}.mp4",
                                tracks=tracks,
                                max_frames=max_frames,
                                start_sec=args.start_sec,
                            )
                        rows.append(
                            summarize_run(
                                case=case,
                                detector_name=detector.name,
                                tracker_name=tracker_name,
                                stabilization=stabilization,
                                frame_detections=frame_detections,
                                tracks=tracks,
                                start_sec=args.start_sec,
                            )
                        )
                    except Exception as exc:
                        rows.append(
                            summarize_failure(
                                case=case,
                                detector_name=detector.name,
                                tracker_name=tracker_name,
                                stabilization=stabilization,
                                error=_short_error(exc),
                                start_sec=args.start_sec,
                            )
                        )

    summary_path = output_root / "summary.csv"
    write_summary_csv(summary_path, rows)
    _print_done(output_root, summary_path, rows)
    return 0 if any(row["status"] == "ok" for row in rows) else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOMG A-part detector evaluation matrix.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Directory containing the 11 evaluation videos.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for manifest, detections, tracks, viz, and summary.csv.",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["yolomg", "baseline"],
        choices=["yolomg", "baseline", "debug_motion"],
        help="Detector adapters to run.",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["nn", "sort"],
        choices=["nn", "sort"],
        help="Tracker adapters to run.",
    )
    parser.add_argument(
        "--stabilization-methods",
        nargs="+",
        default=["none", "ffmpeg_vidstab"],
        choices=["none", "ffmpeg_vidstab"],
        help="Stabilization modes to evaluate.",
    )
    parser.add_argument(
        "--yolomg-repo",
        help="Local path to the Irisky123/YOLOMG checkout.",
    )
    parser.add_argument(
        "--yolomg-weights",
        help="Local path to YOLOMG pretrained best.pt.",
    )
    parser.add_argument(
        "--baseline-model",
        default="yolov8n.pt",
        help="Ultralytics baseline model path/name.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detector confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Detector image size.",
    )
    parser.add_argument(
        "--eval-clip-sec",
        type=float,
        help="Optional fixed clip length for every video.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Start evaluation at this timestamp in seconds.",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip mp4 visualization generation.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        help="Optional number of videos to evaluate from the fixed manifest order.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="Optional video_id to run. Repeat the flag to target multiple specific cases.",
    )
    return parser.parse_args()


def _select_cases(args: argparse.Namespace):
    if args.video_ids:
        requested_ids = list(dict.fromkeys(args.video_ids))
        case_by_id = {case.video_id: case for case in DEFAULT_VIDEO_CASES}
        missing_ids = [video_id for video_id in requested_ids if video_id not in case_by_id]
        if missing_ids:
            available = ", ".join(case_by_id)
            raise SystemExit(
                "Unknown --video-id value(s): "
                + ", ".join(missing_ids)
                + f". Available ids: {available}"
            )
        return [case_by_id[video_id] for video_id in requested_ids]
    if args.case_limit:
        return list(DEFAULT_VIDEO_CASES[: args.case_limit])
    return list(DEFAULT_VIDEO_CASES)


def _build_detectors(args: argparse.Namespace) -> list[DetectorAdapter]:
    detectors: list[DetectorAdapter] = []
    for name in args.detectors:
        if name == "yolomg":
            detectors.append(
                YOLOMGAdapter(
                    repo_path=args.yolomg_repo,
                    weights_path=args.yolomg_weights,
                    confidence_threshold=args.conf,
                    image_size=args.imgsz,
                )
            )
        elif name == "baseline":
            detectors.append(
                UltralyticsYOLOAdapter(
                    model_path=args.baseline_model,
                    confidence_threshold=args.conf,
                    image_size=args.imgsz,
                )
            )
        elif name == "debug_motion":
            detectors.append(OpenCVMotionDebugAdapter())
    return detectors


def _failure_rows_for_case(
    *,
    case,
    detectors: list[DetectorAdapter],
    trackers: list[str],
    methods: tuple[StabilizationMethod, ...] | list[str],
    error: str,
    start_sec: float = 0.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in methods:
        stabilization = StabilizationInfo(
            applied=method != "none",
            method=method,  # type: ignore[arg-type]
        )
        for detector in detectors:
            for tracker in trackers:
                rows.append(
                    summarize_failure(
                        case=case,
                        detector_name=detector.name,
                        tracker_name=tracker,
                        stabilization=stabilization,
                        error=error,
                        start_sec=start_sec,
                    )
                )
    return rows


def _short_error(exc: BaseException) -> str:
    return "".join(format_exception_only(type(exc), exc)).strip()


def _start_suffix(start_sec: float) -> str:
    if start_sec <= 0:
        return ""
    return f"__start_{start_sec:g}s"


def _print_done(output_root: Path, summary_path: Path, rows: list[dict[str, object]]) -> None:
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    error_count = len(rows) - ok_count
    print("Detector evaluation complete")
    print(f"- output_root: {output_root}")
    print(f"- summary: {summary_path}")
    print(f"- ok runs: {ok_count}")
    print(f"- error runs: {error_count}")


if __name__ == "__main__":
    sys.exit(main())
