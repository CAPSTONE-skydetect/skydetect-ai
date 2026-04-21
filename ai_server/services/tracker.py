from pathlib import Path

from ai_server.schemas import AnalyzeResponse, StabilizationMethod
from ai_server.services.stabilization import (
    StabilizationStageError,
    run_stabilization_stage,
)
from ai_server.services.video_io import VideoIOError, iter_video_frames, load_video_metadata


class AnalyzePipelineError(ValueError):
    """Raised when the analyze pipeline cannot start from the given input."""


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
        first_frame = next(iter_video_frames(stabilization_result.stabilized_video_path), None)
    except (StabilizationStageError, VideoIOError) as exc:
        raise AnalyzePipelineError(str(exc)) from exc

    if first_frame is None:
        raise AnalyzePipelineError(
            "Video contains no readable frames after stabilization stage: "
            f"{stabilization_result.stabilized_video_path}"
        )

    return AnalyzeResponse(
        source_video_id=source_video_id,
        tracks=[],
        message=(
            "Pipeline entry initialized from the requested video path "
            f"with stabilization mode '{stabilization_result.stabilization.method}'. "
            f"Loaded {metadata.total_frames} frames at {metadata.fps:.3f} FPS "
            f"({metadata.width}x{metadata.height}); first frame timestamp is "
            f"{first_frame.timestamp_ms} ms. "
            + " ".join(stabilization_result.messages)
            + " Detection and tracking stages are not connected yet."
        ),
    )
