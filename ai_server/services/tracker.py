from pathlib import Path

from ai_server.schemas import AnalyzeResponse
from ai_server.services.manual_roi_tracker import (
    ManualTrackingResult,
    process_manual_roi_video,
)
from ai_server.services.tracking_video_io import TrackingVideoError
from ai_server.tracking_schemas import ManualTrackingRequest


class AnalyzePipelineError(ValueError):
    """Raised when a valid API request cannot be processed as a video track."""


def run_manual_tracking_pipeline(
    payload: ManualTrackingRequest,
) -> AnalyzeResponse:
    result = execute_manual_tracking(payload)

    return AnalyzeResponse(
        source_video_id=payload.source_video_id,
        tracks=[result.track],
        message=(
            "Manual ROI tracking complete: "
            f"{len(result.track.history)} observed frames."
        ),
    )


def execute_manual_tracking(
    payload: ManualTrackingRequest,
    *,
    output_dir: str | Path | None = None,
) -> ManualTrackingResult:
    try:
        return process_manual_roi_video(
            payload.video_path,
            source_video_id=payload.source_video_id,
            target_bbox=payload.target_bbox,
            init_frame_index=payload.init_frame_index,
            max_seconds=payload.max_seconds,
            stabilize=payload.stabilize,
            track_id=payload.track_id,
            tuning=payload.tuning,
            output_dir=output_dir,
            resize_width=payload.resize_width,
            write_overlay=payload.write_overlay,
        )
    except (TrackingVideoError, ValueError) as exc:
        raise AnalyzePipelineError(str(exc)) from exc
