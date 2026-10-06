import shutil
import subprocess

import cv2
import pytest

from ai_server.services.video_transcode import ensure_browser_playable

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg 가 없는 환경",
)


def _make_video(path, codec_args, frames=45):
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=size=321x241:rate=30:duration={frames / 30}",
            *codec_args,
            str(path),
        ],
        check=True,
    )


def _codec_and_frames(path):
    # 헤더의 프레임 수는 컨테이너에 따라 추정값이다 (.mpg 는 길이 × fps).
    # 추적기가 실제로 읽는 수와 비교해야 하므로 끝까지 디코딩해서 센다.
    capture = cv2.VideoCapture(str(path))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    return "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)), frames


def test_mpeg2_mp4_is_transcoded_to_h264_with_same_frame_count(tmp_path):
    source = tmp_path / "clip.mp4"
    _make_video(source, ["-c:v", "mpeg2video"])
    _, original_frames = _codec_and_frames(source)

    result = ensure_browser_playable(source)

    assert result.transcoded is True
    assert result.original_codec == "mpeg2video"
    assert result.path == tmp_path / "clip.mp4"
    assert result.path.exists()
    assert not (tmp_path / "clip.transcoding.mp4").exists()
    codec, frames = _codec_and_frames(result.path)
    assert codec in {"avc1", "h264"}
    assert frames == original_frames


def test_avi_is_transcoded_to_mp4_and_original_removed(tmp_path):
    source = tmp_path / "clip.avi"
    _make_video(source, ["-c:v", "mjpeg"])

    result = ensure_browser_playable(source)

    assert result.transcoded is True
    assert result.path == tmp_path / "clip.mp4"
    assert not source.exists()


def test_mpg_program_stream_is_transcoded_to_mp4(tmp_path):
    source = tmp_path / "clip.mpg"
    _make_video(source, ["-c:v", "mpeg2video", "-f", "mpeg"])
    _, original_frames = _codec_and_frames(source)

    result = ensure_browser_playable(source)

    assert result.transcoded is True
    assert result.path == tmp_path / "clip.mp4"
    assert not source.exists()
    codec, frames = _codec_and_frames(result.path)
    assert codec in {"avc1", "h264"}
    assert frames == original_frames


def test_h264_yuv420p_mp4_is_left_untouched(tmp_path):
    source = tmp_path / "clip.mp4"
    _make_video(
        source,
        ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=320:240"],
    )
    before = source.stat().st_mtime_ns

    result = ensure_browser_playable(source)

    assert result.transcoded is False
    assert result.original_codec == "h264"
    assert source.stat().st_mtime_ns == before
