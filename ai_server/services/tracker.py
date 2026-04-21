from pathlib import Path

from ai_server.schemas import AnalyzeResponse, StabilizationInfo


class AnalyzePipelineError(ValueError):
    """Raised when the analyze pipeline cannot start from the given input."""


def run_track_sequence_pipeline(
    video_path: str,
    source_video_id: str,
    stabilization: StabilizationInfo,
) -> AnalyzeResponse:
    input_path = Path(video_path).expanduser()

    if not input_path.exists():
        raise AnalyzePipelineError(f"Video path does not exist: {input_path}")
    if not input_path.is_file():
        raise AnalyzePipelineError(f"Video path is not a file: {input_path}")

    return AnalyzeResponse(
        source_video_id=source_video_id,
        tracks=[],
        message=(
            "Pipeline entry initialized from the requested video path "
            f"with stabilization mode '{stabilization.method}'. "
            "Frame loading, detection, and tracking stages are not connected yet."
        ),
    )
