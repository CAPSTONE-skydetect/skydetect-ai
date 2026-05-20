from pathlib import Path
import argparse

import pytest

from ai_server.schemas import StabilizationInfo
from ai_server.services.detector import Detection, FrameDetections
from ai_server.services.detector_eval import (
    DetectionArtifact,
    DEFAULT_VIDEO_CASES,
    build_tracks_for_tracker,
    detection_gap_stats,
    read_detection_artifact,
    read_detection_json,
    summarize_run,
    write_detection_json,
)
from ai_server.services.video_io import VideoMetadata
from scripts.run_detector_eval import (
    _conf_suffix,
    _select_cases,
    _stabilization_for_source_id,
    _start_sec_for_source_id,
    _start_suffix,
)


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        path="synthetic.mp4",
        fps=30.0,
        total_frames=4,
        width=100,
        height=100,
    )


def _wide_metadata() -> VideoMetadata:
    return VideoMetadata(
        path="synthetic_wide.mp4",
        fps=30.0,
        total_frames=40,
        width=1000,
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


def test_detection_artifact_round_trip_includes_source_metadata(tmp_path: Path) -> None:
    frame_detections = [
        _frame(0, [Detection(left=10, top=20, width=5, height=6, confidence=0.7)]),
    ]
    output_path = tmp_path / "detections.json"

    write_detection_json(
        output_path,
        source_video_id="video_01__none__yolomg__conf020",
        detector_name="yolomg",
        frame_detections=frame_detections,
    )

    restored = read_detection_artifact(output_path)

    assert restored == DetectionArtifact(
        source_video_id="video_01__none__yolomg__conf020",
        detector_name="yolomg",
        frame_detections=frame_detections,
    )


def test_detection_gap_stats_counts_gap_and_detection_runs() -> None:
    detection = Detection(left=10, top=10, width=8, height=8, confidence=0.8)
    frame_detections = [
        _frame(0, []),
        _frame(1, []),
        _frame(2, [detection]),
        _frame(3, [detection]),
        _frame(4, [detection]),
        _frame(5, []),
        _frame(6, [detection]),
        _frame(7, [detection]),
        _frame(8, []),
        _frame(9, []),
        _frame(10, []),
    ]

    assert detection_gap_stats(frame_detections) == {
        "num_detection_gaps": 3,
        "max_detection_gap": 3,
        "median_detection_gap": 2.0,
        "mean_detection_gap": 2.0,
        "detection_gap_p90": 3,
        "longest_detection_run": 3,
    }


def test_detection_gap_stats_handles_all_detected_frames() -> None:
    detection = Detection(left=10, top=10, width=8, height=8, confidence=0.8)
    frame_detections = [
        _frame(0, [detection]),
        _frame(1, [detection]),
        _frame(2, [detection]),
    ]

    assert detection_gap_stats(frame_detections) == {
        "num_detection_gaps": 0,
        "max_detection_gap": 0,
        "median_detection_gap": 0.0,
        "mean_detection_gap": 0.0,
        "detection_gap_p90": 0,
        "longest_detection_run": 3,
    }


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


def test_sort_long_memory_keeps_track_across_long_detection_gap() -> None:
    frame_detections = [
        _frame(0, [Detection(left=10, top=10, width=8, height=8, confidence=0.8)]),
        _frame(20, [Detection(left=10, top=10, width=8, height=8, confidence=0.8)]),
    ]

    sort_tracks = build_tracks_for_tracker(
        tracker_name="sort",
        frame_detections=frame_detections,
        metadata=_metadata(),
        source_video_id="synthetic_sort",
        stabilization=StabilizationInfo(applied=False, method="none"),
    )
    long_memory_tracks = build_tracks_for_tracker(
        tracker_name="sort_long_memory",
        frame_detections=frame_detections,
        metadata=_metadata(),
        source_video_id="synthetic_sort_long_memory",
        stabilization=StabilizationInfo(applied=False, method="none"),
    )

    assert len(sort_tracks) == 2
    assert len(long_memory_tracks) == 1
    assert [point.frame_index for point in long_memory_tracks[0].history] == [0, 20]


def test_sort_center_matches_small_boxes_by_normalized_center_distance() -> None:
    frame_detections = [
        _frame(0, [Detection(left=10, top=10, width=4, height=4, confidence=0.8)]),
        _frame(1, [Detection(left=30, top=10, width=4, height=4, confidence=0.8)]),
    ]

    sort_tracks = build_tracks_for_tracker(
        tracker_name="sort",
        frame_detections=frame_detections,
        metadata=_wide_metadata(),
        source_video_id="synthetic_sort",
        stabilization=StabilizationInfo(applied=False, method="none"),
    )
    center_tracks = build_tracks_for_tracker(
        tracker_name="sort_center",
        frame_detections=frame_detections,
        metadata=_wide_metadata(),
        source_video_id="synthetic_sort_center",
        stabilization=StabilizationInfo(applied=False, method="none"),
    )

    assert len(sort_tracks) == 2
    assert len(center_tracks) == 1
    assert [point.frame_index for point in center_tracks[0].history] == [0, 1]


def test_upper_bound_links_single_detections_within_long_memory_window() -> None:
    frame_detections = [
        _frame(0, [Detection(left=10, top=10, width=4, height=4, confidence=0.8)]),
        _frame(20, [Detection(left=400, top=10, width=4, height=4, confidence=0.8)]),
    ]

    long_memory_tracks = build_tracks_for_tracker(
        tracker_name="sort_long_memory",
        frame_detections=frame_detections,
        metadata=_wide_metadata(),
        source_video_id="synthetic_sort_long_memory",
        stabilization=StabilizationInfo(applied=False, method="none"),
    )
    upper_bound_tracks = build_tracks_for_tracker(
        tracker_name="upper_bound",
        frame_detections=frame_detections,
        metadata=_wide_metadata(),
        source_video_id="synthetic_upper_bound",
        stabilization=StabilizationInfo(applied=False, method="none"),
    )

    assert len(long_memory_tracks) == 2
    assert len(upper_bound_tracks) == 1
    assert [point.frame_index for point in upper_bound_tracks[0].history] == [0, 20]


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
    assert row["num_detection_gaps"] == 0
    assert row["max_detection_gap"] == 0
    assert row["median_detection_gap"] == 0.0
    assert row["mean_detection_gap"] == 0.0
    assert row["detection_gap_p90"] == 0
    assert row["longest_detection_run"] == 2
    assert row["track_count"] == 1
    assert row["main_track_num_points"] == 2
    assert row["main_track_ratio_total"] == 1.0
    assert row["main_track_ratio_recoverable"] == 1.0
    assert row["eval_start_sec"] == 0.0


def test_summary_uses_zero_track_ratios_without_main_track() -> None:
    row = summarize_run(
        case=DEFAULT_VIDEO_CASES[0],
        detector_name="yolomg",
        tracker_name="sort",
        stabilization=StabilizationInfo(applied=False, method="none"),
        frame_detections=[
            _frame(0, []),
            _frame(1, [Detection(left=10, top=10, width=8, height=8, confidence=0.8)]),
        ],
        tracks=[],
    )

    assert row["main_track_num_points"] == 0
    assert row["main_track_ratio_total"] == 0.0
    assert row["main_track_ratio_recoverable"] == 0.0


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


def test_conf_suffix_uses_three_digit_percent_format() -> None:
    assert _conf_suffix(0.05) == "__conf005"
    assert _conf_suffix(0.2) == "__conf020"
    assert _conf_suffix(0.45) == "__conf045"


def test_source_id_helpers_parse_reused_detection_context() -> None:
    source_video_id = "drone_fix_stab_far_01__start_4s__ffmpeg_vidstab__yolomg__conf020"

    assert _start_sec_for_source_id(source_video_id) == 4.0
    stabilization = _stabilization_for_source_id(source_video_id)
    assert stabilization.applied is True
    assert stabilization.method == "ffmpeg_vidstab"
