"""Create an exact decoded-frame prefix without altering the source or GT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from research.trajectory_experiments.manifest import sha256


def prepare(source: Path, output: Path, frame_count: int) -> dict:
    if frame_count <= 0 or output.exists():
        raise ValueError("positive frame_count and a new output path are required")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"cannot open source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
    if not writer.isOpened():
        cap.release()
        raise ValueError("lossless FFV1 writer is unavailable")
    hashes = []
    try:
        for index in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"source ends before requested frame {index}")
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    verify = cv2.VideoCapture(str(output))
    try:
        for index, expected in enumerate(hashes):
            ok, frame = verify.read()
            if not ok or hashlib.sha256(frame.tobytes()).hexdigest() != expected:
                raise ValueError(f"decoded pixel verification failed at frame {index}")
        if verify.read()[0]:
            raise ValueError("output contains more frames than requested")
        if not np.isclose(verify.get(cv2.CAP_PROP_FPS), fps, atol=1e-6):
            raise ValueError("output FPS differs from source")
    finally:
        verify.release()
    result = {
        "source": str(source.resolve()), "source_sha256": sha256(source),
        "output": str(output.resolve()), "output_sha256": sha256(output),
        "frame_range": [0, frame_count - 1], "frame_count": frame_count,
        "fps": fps, "size": list(size), "codec": "FFV1",
        "verification": "every decoded BGR pixel matches the source prefix",
        "frame_sha256": hashes,
        "limitation": "Preserves source frame indices; CVAT-to-source identity still requires annotation review.",
    }
    output.with_suffix(".verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.frames)
    print(json.dumps({key: value for key, value in result.items() if key != "frame_sha256"}, indent=2))


if __name__ == "__main__":
    main()
