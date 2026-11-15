"""Run inside the standalone clip image, without AI packages or live services."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        os.environ.update(
            PI_BRIDGE_DB=str(root / "jobs.sqlite3"),
            PI_CLIPS_DIR=str(root / "clips"),
            PI_STAGE_DIR=str(root / "staging"),
            PI_PIN_DIR=str(root / "pins"),
        )
        sys.path.insert(0, "/bridge")
        import server
        assert importlib.util.find_spec("sklearn") is None
        assert importlib.util.find_spec("cv2") is None
        server.initialize_db()
        part = root / "2026-11-06_16-00-00-000000_+0900.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "color=c=black:s=1920x1080:r=30:d=4", "-c:v", "libx264",
            "-preset", "ultrafast", str(part),
        ], check=True, timeout=30)
        start = server.parse_segment_time(part) + 1.123
        result = server._extract_clip([part], root, root / "result.mp4", start, start + 2)
        assert result["metadata"]["frameCount"] == 60
        assert len(result["frame_map"]) == 60
        print("PASS: non-root standalone worker, SQLite, 1080p FFmpeg extraction, 60-frame PTS map")


if __name__ == "__main__":
    main()
