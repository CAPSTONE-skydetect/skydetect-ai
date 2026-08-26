from ai_server.schemas import AnalyzeResponse
from ai_server.services.manual_roi_tracker import process_manual_roi_video
from ai_server.services.tracking_video_io import TrackingVideoError
from ai_server.tracking_schemas import ManualTrackingRequest


class AnalyzePipelineError(ValueError):
    """Raised when a valid API request cannot be processed as a video track."""


def run_manual_tracking_pipeline(
    payload: ManualTrackingRequest,
) -> AnalyzeResponse:
    try:
        result = process_manual_roi_video(
            payload.video_path,
            source_video_id=payload.source_video_id,
            target_bbox=payload.target_bbox,
            init_frame_index=payload.init_frame_index,
            max_seconds=payload.max_seconds,
            stabilize=payload.stabilize,
            track_id=payload.track_id,
            tuning=payload.tuning,
            resize_width=payload.resize_width,
            write_overlay=payload.write_overlay,
        )
    except (TrackingVideoError, ValueError) as exc:
        raise AnalyzePipelineError(str(exc)) from exc

    return AnalyzeResponse(
        source_video_id=payload.source_video_id,
        tracks=[result.track],
        message=(
            "Manual ROI tracking complete: "
            f"{len(result.track.history)} observed frames."
        ),
    )
