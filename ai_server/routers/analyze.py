from fastapi import APIRouter

from ai_server.schemas import AnalyzeRequest, AnalyzeResponse
from ai_server.services.stabilization import build_stabilization_info
from ai_server.services.tracker import build_bootstrap_track

router = APIRouter(tags=["analyze"])


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_video(payload: AnalyzeRequest) -> AnalyzeResponse:
    stabilization = build_stabilization_info(payload.stabilization_method)
    track = build_bootstrap_track(
        track_id=1,
        source_video_id=payload.source_video_id,
        stabilization=stabilization,
    )
    return AnalyzeResponse(
        source_video_id=payload.source_video_id,
        tracks=[track],
        message="Bootstrap response. Replace detector/tracker stubs with real pipeline stages.",
    )
