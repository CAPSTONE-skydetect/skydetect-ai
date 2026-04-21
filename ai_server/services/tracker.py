from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ai_server.schemas import (
    AnalyzeResponse,
    StabilizationInfo,
    StabilizationMethod,
    TrackPoint,
    TrackSequence,
)
from ai_server.services.detector import Detection, FrameDetections, detect_flying_objects
from ai_server.services.stabilization import (
    StabilizationStageError,
    run_stabilization_stage,
)
from ai_server.services.video_io import VideoIOError, VideoMetadata, iter_video_frames, load_video_metadata
from ai_server.utils.quality import build_track_quality


class AnalyzePipelineError(ValueError):
    """Raised when the analyze pipeline cannot start from the given input."""


@dataclass
class _TrackBuilderState:
    track_id: int
    history: list[TrackPoint] = field(default_factory=list)

    @property
    def last_point(self) -> TrackPoint:
        return self.history[-1]


def run_track_sequence_pipeline(
    video_path: str,
    source_video_id: str,
    stabilization_method: StabilizationMethod,
) -> AnalyzeResponse:
    input_path = Path(video_path).expanduser()

    if not input_path.exists():
        raise AnalyzePipelineError(f"Video path does not exist: {input_path}")
    if not input_path.is_file():
        raise AnalyzePipelineError(f"Video path is not a file: {input_path}")

    try:
        stabilization_result = run_stabilization_stage(
            video_path=str(input_path),
            method=stabilization_method,
        )
        metadata = load_video_metadata(stabilization_result.stabilized_video_path)
        frame_detections = list(
            detect_flying_objects(
                iter_video_frames(stabilization_result.stabilized_video_path),
            )
        )
    except (StabilizationStageError, VideoIOError) as exc:
        raise AnalyzePipelineError(str(exc)) from exc

    if not frame_detections:
        raise AnalyzePipelineError(
            "Video contains no readable frames after stabilization stage: "
            f"{stabilization_result.stabilized_video_path}"
        )

    tracks = build_track_sequences(
        frame_detections=frame_detections,
        metadata=metadata,
        source_video_id=source_video_id,
        stabilization=stabilization_result.stabilization,
    )
    first_frame_detection_count = len(frame_detections[0].detections)
    total_detection_count = sum(len(frame.detections) for frame in frame_detections)
    first_frame_timestamp_ms = frame_detections[0].timestamp_ms

    return AnalyzeResponse(
        source_video_id=source_video_id,
        tracks=tracks,
        message=(
            "Pipeline entry initialized from the requested video path "
            f"with stabilization mode '{stabilization_result.stabilization.method}'. "
            f"Loaded {metadata.total_frames} frames at {metadata.fps:.3f} FPS "
            f"({metadata.width}x{metadata.height}); first frame timestamp is "
            f"{first_frame_timestamp_ms} ms. "
            f"Detector contract produced {first_frame_detection_count} candidate boxes "
            f"on the first frame and {total_detection_count} total detections. "
            f"Tracker assembled {len(tracks)} track sequences. "
            + " ".join(stabilization_result.messages)
            + " Feature extraction and classification stages are not connected yet."
        ),
    )


def build_track_sequences(
    frame_detections: list[FrameDetections],
    metadata: VideoMetadata,
    source_video_id: str,
    stabilization: StabilizationInfo,
) -> list[TrackSequence]:
    active_tracks: list[_TrackBuilderState] = []
    next_track_id = 1

    for frame in frame_detections:
        matched_track_ids: set[int] = set()
        for detection in frame.detections:
            point = _detection_to_track_point(
                detection=detection,
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                metadata=metadata,
            )
            track = _match_track(
                active_tracks=active_tracks,
                point=point,
                matched_track_ids=matched_track_ids,
            )
            if track is None:
                track = _TrackBuilderState(track_id=next_track_id)
                active_tracks.append(track)
                next_track_id += 1
            track.history.append(point)
            matched_track_ids.add(track.track_id)

    return [
        TrackSequence(
            track_id=track.track_id,
            source_video_id=source_video_id,
            stabilization=stabilization,
            history=track.history,
            quality=build_track_quality(track.history),
        )
        for track in active_tracks
        if track.history
    ]


def _detection_to_track_point(
    detection: Detection,
    frame_index: int,
    timestamp_ms: int,
    metadata: VideoMetadata,
) -> TrackPoint:
    return TrackPoint(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        cx=(detection.left + (detection.width / 2)) / metadata.width,
        cy=(detection.top + (detection.height / 2)) / metadata.height,
        w=detection.width / metadata.width,
        h=detection.height / metadata.height,
        conf=detection.confidence,
    )


def _match_track(
    active_tracks: list[_TrackBuilderState],
    point: TrackPoint,
    matched_track_ids: set[int],
) -> _TrackBuilderState | None:
    best_track: _TrackBuilderState | None = None
    best_distance_sq: float | None = None

    for track in active_tracks:
        if track.track_id in matched_track_ids:
            continue
        previous = track.last_point
        if point.frame_index - previous.frame_index > 1:
            continue

        dx = point.cx - previous.cx
        dy = point.cy - previous.cy
        distance_sq = (dx * dx) + (dy * dy)
        if distance_sq > 0.0025:
            continue

        if best_distance_sq is None or distance_sq < best_distance_sq:
            best_track = track
            best_distance_sq = distance_sq

    return best_track
