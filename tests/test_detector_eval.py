from pathlib import Path
import argparse

import pytest

from ai_server.schemas import StabilizationInfo
from ai_server.services.detector import Detection, FrameDetections
from ai_server.services.detector_eval import (
    DEFAULT_VIDEO_CASES,
    build_tracks_for_tracker,
    read_detection_json,
    summarize_run,
    write_detection_json,
)
from ai_server.services.video_io import VideoMetadata
from scripts.run_detector_eval import _select_cases, _start_suffix


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        path="synthetic.mp4",
        fps=30.0,
        total_frames=4,
        width=100,
        height=100,
    )


def _frame(frame_index: int, detections: list[Detection]) -> FrameDetections:
    return FrameDetections(
        frame_index=frame_index,
        timestamp_ms=round((frame_index / 30.0) * 1000),
        detections=detections,
    )


def test_detection_json_round_trip(tmp_path: Path) -> None:
    frame_detections = [
        _frame(0, [Detection(left=10, top=20, width=5, height=6, confidence=0.7)]),
        _frame(1, []),
    ]
    output_path = tmp_path / "detections.json"

    write_detection_json(
        output_path,
        source_video_id="video_01",
        detector_name="yolomg",
        frame_detections=frame_detections,
    )

    restored = read_detection_json(output_path)
    assert restored == frame_detections


def test_sort_tracker_keeps_track_across_short_detection_gap() -> None:
    frame_detections = [
        _frame(0, [Detection(left=10, top=10, width=8, height=8, confidence=0.8)]),
        _frame(1, []),
        _frame(2, [Detection(left=12, top=10, width=8, height=8, confidence=0.8)]),
    ]

    tracks = build_tracks_for_tracker(
        tracker_name="sort",
        frame_detections=frame_detections,
        metadata=_metadata(),
        source_video_id="synthetic_sort",
        stabilization=StabilizationInfo(applied=False, method="none"),
    )

    assert len(tracks) == 1
    assert [point.frame_index for point in tracks[0].history] == [0, 2]
    assert tracks[0].quality is not None
    assert tracks[0].quality.missing_ratio == 0.333


def test_summary_includes_detector_and_track_metrics() -> None:
    case = DEFAULT_VIDEO_CASES[0]
    frame_detections = [
        _frame(0, [Detection(left=10, top=10, width=8, height=8, confidence=0.8)]),
        _frame(1, [Detection(left=11, top=10, width=8, height=8, confidence=0.6)]),
    ]
    tracks = build_tracks_for_tracker(
        tracker_name="nn",
        frame_detections=frame_detections,
        metadata=_metadata(),
        source_video_id="synthetic_nn",
        stabilization=StabilizationInfo(applied=False, method="none"),
    )

    row = summarize_run(
        case=case,
        detector_name="yolomg",
        tracker_name="nn",
        stabilization=StabilizationInfo(applied=False, method="none"),
        frame_detections=frame_detections,
        tracks=tracks,
    )

    assert row["video_id"] == "drone_fix_stab_01"
    assert row["detector"] == "yolomg"
    assert row["tracker"] == "nn"
    assert row["detected_frame_ratio"] == 1.0
    assert row["detector_mean_conf"] == 0.7
    assert row["track_count"] == 1
    assert row["main_track_num_points"] == 2
    assert row["eval_start_sec"] == 0.0


def test_select_cases_by_video_id_preserves_request_order() -> None:
    args = argparse.Namespace(video_ids=["bird_fix_stab_01", "drone_fix_stab_01"], case_limit=None)

    cases = _select_cases(args)

    assert [case.video_id for case in cases] == ["bird_fix_stab_01", "drone_fix_stab_01"]


def test_select_cases_rejects_unknown_video_id() -> None:
    args = argparse.Namespace(video_ids=["missing_case"], case_limit=None)

    with pytest.raises(SystemExit, match="Unknown --video-id value"):
        _select_cases(args)


def test_start_suffix_only_marks_offset_runs() -> None:
    assert _start_suffix(0.0) == ""
    assert _start_suffix(34.0) == "__start_34s"
    assert _start_suffix(33.5) == "__start_33.5s"
