"""업로드 영상을 브라우저에서 재생 가능한 H.264 mp4 로 맞춘다.

웹 프론트는 업로드한 영상을 프레임 단위로 보면서 ROI 를 지정한다. 그런데 확장자가
mp4 여도 안의 코덱이 MPEG-2 / HEVC 등이면 브라우저가 재생하지 못해 화면이 검게 나온다.
서버(OpenCV)는 읽을 수 있어도 사람이 대상을 고를 수가 없다.

그래서 업로드 시점에 재생 가능 여부를 보고, 안 되면 H.264(yuv420p) mp4 로 변환한다.
추적도 변환한 파일로 하므로 화면에서 고른 프레임 번호와 추적 프레임이 어긋나지 않는다.
프레임을 버리거나 복제하지 않도록 타임스탬프를 그대로 넘긴다(-fps_mode passthrough).

ffmpeg 가 없는 환경(로컬 개발 등)에서는 변환을 건너뛰고 원본을 그대로 쓴다.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# 브라우저(Chrome/Edge/Firefox/Safari)가 공통으로 재생하는 조합
_PLAYABLE_SUFFIX = ".mp4"
_PLAYABLE_CODEC = "h264"
_PLAYABLE_PIX_FMT = "yuv420p"

_TIMEOUT_SEC = 600


class VideoTranscodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscodeResult:
    path: Path
    transcoded: bool
    original_codec: str | None
    reason: str | None = None


def ensure_browser_playable(path: Path) -> TranscodeResult:
    """재생 가능하면 그대로, 아니면 같은 이름의 .mp4 로 변환한 뒤 원본을 지운다."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        return TranscodeResult(path, False, None, reason="ffmpeg_not_installed")

    codec, pix_fmt = _probe_video_stream(ffprobe, path)
    if (
        path.suffix.lower() == _PLAYABLE_SUFFIX
        and codec == _PLAYABLE_CODEC
        and pix_fmt == _PLAYABLE_PIX_FMT
    ):
        return TranscodeResult(path, False, codec)

    # 원본이 .mp4 면 이름이 같아지므로 임시 파일에 쓰고 나중에 바꿔 단다.
    target = path.with_suffix(".mp4")
    temporary = path.with_name(f"{path.stem}.transcoding.mp4")
    command = [
        ffmpeg, "-y", "-v", "error",
        "-i", str(path),
        "-map", "0:v:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", _PLAYABLE_PIX_FMT,
        # yuv420p 는 가로세로가 짝수여야 한다. 홀수면 1px 줄인다.
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        # 프레임 수를 원본과 같게 유지한다. ROI 프레임 번호가 그대로 맞아야 한다.
        "-fps_mode", "passthrough",
        "-an",
        "-movflags", "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=_TIMEOUT_SEC)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        temporary.unlink(missing_ok=True)
        stderr = getattr(exc, "stderr", b"") or b""
        raise VideoTranscodeError(
            f"영상을 H.264 로 변환하지 못했습니다: {stderr.decode(errors='replace')[-300:]}"
        ) from exc

    path.unlink(missing_ok=True)
    temporary.replace(target)
    return TranscodeResult(target, True, codec)


def _probe_video_stream(ffprobe: str, path: Path) -> tuple[str | None, str | None]:
    command = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt",
        "-of", "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, timeout=60,
        )
        streams = json.loads(completed.stdout or b"{}").get("streams") or []
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None, None
    if not streams:
        return None, None
    return streams[0].get("codec_name"), streams[0].get("pix_fmt")
