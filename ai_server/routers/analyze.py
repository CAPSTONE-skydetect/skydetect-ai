from fastapi import APIRouter, HTTPException, status

from ai_server.schemas import AnalyzeResponse
from ai_server.services.tracker import (
    AnalyzePipelineError,
    run_manual_tracking_pipeline,
)
from ai_server.tracking_schemas import ManualTrackingRequest

router = APIRouter(tags=["analyze"])


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_video(payload: ManualTrackingRequest) -> AnalyzeResponse:
    try:
        return run_manual_tracking_pipeline(payload)
    except AnalyzePipelineError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
