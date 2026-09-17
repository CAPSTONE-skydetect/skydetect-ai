from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from ai_server.services.tracker import AnalyzePipelineError, execute_manual_tracking
from ai_server.services.tracking_video_io import (
    TrackingVideoError,
    read_tracking_video_metadata,
)
from ai_server.tracking_schemas import ManualTrackingRequest

router = APIRouter(tags=["tracking-ui"])

UPLOAD_DIR = Path(os.environ.get("SKYDETECT_UPLOAD_DIR", "storage/uploads"))
OUTPUT_DIR = Path(
    os.environ.get("SKYDETECT_TRACK_OUTPUT_DIR", "artifacts/manual_tracks")
)
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@router.post("/videos/upload")
async def upload_video(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="지원하지 않는 영상 형식입니다.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = (UPLOAD_DIR / f"{uuid4().hex}{suffix}").resolve()
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                handle.write(chunk)
        metadata = read_tracking_video_metadata(target, resize_width=1280).to_dict()
    except (OSError, TrackingVideoError) as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()

    return {
        "message": "video uploaded",
        "source_video_id": target.stem,
        "video_path": str(target),
        "metadata": metadata,
        "download_urls": {"source_video": _download_url(target)},
    }


@router.post("/tracks/manual")
def create_manual_track(payload: ManualTrackingRequest) -> dict[str, object]:
    safe_video_path = _safe_file_path(payload.video_path, roots=[UPLOAD_DIR])
    safe_payload = payload.model_copy(update={"video_path": str(safe_video_path)})
    try:
        result = execute_manual_tracking(safe_payload, output_dir=OUTPUT_DIR)
    except AnalyzePipelineError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return {
        "message": "manual ROI tracking complete",
        "source_video_id": payload.source_video_id,
        "tracks": [result.track.model_dump(mode="json")],
        "metadata": result.metadata,
        "metrics": result.metrics,
        "artifacts": result.artifacts,
        "download_urls": {
            "source_video": _download_url(safe_video_path),
            "track_sequence": _download_url(result.artifacts["track_sequence"]),
            "trajectory": _download_url(result.artifacts["trajectory"]),
            "overlay": _download_url(result.artifacts["overlay"]),
            "metrics": _download_url(result.artifacts["metrics"]),
            "foreground_mask": _download_url(result.artifacts["foreground_mask"]),
        },
    }


@router.get("/files")
def download_file(path: str) -> FileResponse:
    target = _safe_file_path(path, roots=[UPLOAD_DIR, OUTPUT_DIR])
    return FileResponse(str(target), filename=target.name)


def _download_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    return f"/api/files?path={quote(str(path), safe='')}"


def _safe_file_path(path: str | Path, *, roots: list[Path]) -> Path:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.resolve()
    resolved_roots = [
        (Path.cwd() / root).resolve() if not root.is_absolute() else root.resolve()
        for root in roots
    ]
    if not any(target.is_relative_to(root) for root in resolved_roots):
        raise HTTPException(status_code=403, detail="접근할 수 없는 파일 경로입니다.")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return target
