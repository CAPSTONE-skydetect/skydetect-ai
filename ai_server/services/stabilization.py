from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from tempfile import gettempdir
from uuid import uuid4

from ai_server.schemas import StabilizationInfo, StabilizationMethod


class StabilizationStageError(ValueError):
    """Raised when the stabilization stage cannot run successfully."""


@dataclass(frozen=True)
class StabilizationStageResult:
    source_video_path: str
    stabilized_video_path: str
    stabilization: StabilizationInfo
    messages: tuple[str, ...]


def run_stabilization_stage(
    video_path: str,
    method: StabilizationMethod,
) -> StabilizationStageResult:
    input_path = Path(video_path).expanduser()
    if not input_path.exists():
        raise StabilizationStageError(f"Video path does not exist: {input_path}")
    if not input_path.is_file():
        raise StabilizationStageError(f"Video path is not a file: {input_path}")

    if method == "none":
        return StabilizationStageResult(
            source_video_path=str(input_path),
            stabilized_video_path=str(input_path),
            stabilization=StabilizationInfo(applied=False, method="none"),
            messages=("Stabilization bypassed; using the original video stream.",),
        )

    if method == "ffmpeg_vidstab":
        return _run_ffmpeg_vidstab(input_path)

    raise StabilizationStageError(
        f"Stabilization method '{method}' is not wired into the pipeline yet."
    )


def _run_ffmpeg_vidstab(input_path: Path) -> StabilizationStageResult:
    ffmpeg_bin = which("ffmpeg")
    if ffmpeg_bin is None:
        raise StabilizationStageError(
            "ffmpeg is required for 'ffmpeg_vidstab' stabilization but was not found "
            "on PATH."
        )

    work_dir = Path(gettempdir()) / "skydetect_ai_stabilization"
    work_dir.mkdir(parents=True, exist_ok=True)

    job_id = uuid4().hex
    safe_stem = _safe_ffmpeg_stem(input_path.stem)
    transforms_path = work_dir / f"{safe_stem}_{job_id}.trf"
    output_path = work_dir / f"{safe_stem}_{job_id}_stabilized.mp4"

    detect_cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"vidstabdetect=shakiness=5:accuracy=15:result={transforms_path}",
        "-f",
        "null",
        "-",
    ]
    transform_cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"vidstabtransform=input={transforms_path}:zoom=0:smoothing=20",
        "-an",
        str(output_path),
    ]

    detect_result = subprocess.run(
        detect_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if detect_result.returncode != 0:
        stderr = _last_stderr_line(detect_result.stderr)
        raise StabilizationStageError(
            "ffmpeg vidstabdetect failed"
            + (f": {stderr}" if stderr else ".")
        )

    transform_result = subprocess.run(
        transform_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if transform_result.returncode != 0:
        stderr = _last_stderr_line(transform_result.stderr)
        raise StabilizationStageError(
            "ffmpeg vidstabtransform failed"
            + (f": {stderr}" if stderr else ".")
        )

    if not output_path.exists():
        raise StabilizationStageError(
            "ffmpeg reported success but no stabilized output file was created."
        )

    return StabilizationStageResult(
        source_video_path=str(input_path),
        stabilized_video_path=str(output_path),
        stabilization=StabilizationInfo(applied=True, method="ffmpeg_vidstab"),
        messages=(
            "Stabilization completed with ffmpeg vidstab.",
            f"Generated stabilized video at {output_path}.",
        ),
    )


def _last_stderr_line(stderr: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _safe_ffmpeg_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._") or "video"
