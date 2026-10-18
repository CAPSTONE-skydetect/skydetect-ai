from pathlib import Path

import cv2
import numpy as np

from research.verified_video_prefix import prepare


def test_prefix_keeps_exact_decoded_pixels_and_frame_count(tmp_path: Path) -> None:
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"FFV1"), 25, (64, 48))
    for index in range(6):
        writer.write(np.full((48, 64, 3), index * 30, dtype=np.uint8))
    writer.release()
    result = prepare(source, tmp_path / "prefix.avi", 4)
    assert result["frame_count"] == 4
    assert result["frame_range"] == [0, 3]
    assert len(set(result["frame_sha256"])) == 4
