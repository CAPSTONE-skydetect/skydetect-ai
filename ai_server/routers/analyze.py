from fastapi import APIRouter, HTTPException

from ai_server.schemas import AnalyzeRequest, AnalyzeResponse
from ai_server.services.stabilization import build_stabilization_info
from ai_server.services.tracker import AnalyzePipelineError, run_track_sequence_pipeline

router = APIRouter(tags=["analyze"])


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_video(payload: AnalyzeRequest) -> AnalyzeResponse:
    stabilization = build_stabilization_info(payload.stabilization_method)
    try:
        return run_track_sequence_pipeline(
            video_path=payload.video_path,
            source_video_id=payload.source_video_id,
            stabilization=stabilization,
        )
    except AnalyzePipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
