from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ai_server.services.detector_eval import _motion_compensate_to_reference
from ai_server.services.video_io import load_video_metadata


def main() -> int:
    args = _parse_args()

    import cv2

    video_path = str(Path(args.video_path).expanduser())
    metadata = load_video_metadata(video_path)
    center_index = _resolve_center_frame(args, metadata.fps, metadata.total_frames)
    older_index = max(0, center_index - args.temporal_radius)
    newer_index = min(metadata.total_frames - 1, center_index + args.temporal_radius)

    frames = _read_frames(
        video_path=video_path,
        frame_indices=(older_index, center_index, newer_index),
        cv2=cv2,
    )
    older_frame = frames[older_index]
    center_frame = frames[center_index]
    newer_frame = frames[newer_index]

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / f"frame_{older_index:06d}_older.jpg"), older_frame)
    cv2.imwrite(str(output_dir / f"frame_{center_index:06d}_center.jpg"), center_frame)
    cv2.imwrite(str(output_dir / f"frame_{newer_index:06d}_newer.jpg"), newer_frame)

    summary: dict[str, Any] = {
        "video_path": video_path,
        "fps": metadata.fps,
        "center_frame_index": center_index,
        "older_frame_index": older_index,
        "newer_frame_index": newer_index,
        "temporal_radius": args.temporal_radius,
        "variants": {},
    }

    for blur_kernel in args.blur_kernel:
        gray = {
            "older": _gray(older_frame, cv2=cv2, blur_kernel=blur_kernel),
            "center": _gray(center_frame, cv2=cv2, blur_kernel=blur_kernel),
            "newer": _gray(newer_frame, cv2=cv2, blur_kernel=blur_kernel),
        }

        raw_diff = _average_diff(gray["center"], gray["older"], gray["newer"], cv2=cv2)
        compensated_older = _motion_compensate_to_reference(
            moving_gray=gray["older"],
            reference_gray=gray["center"],
            cv2=cv2,
        )
        compensated_newer = _motion_compensate_to_reference(
            moving_gray=gray["newer"],
            reference_gray=gray["center"],
            cv2=cv2,
        )
        compensated_diff = _average_diff(
            gray["center"],
            compensated_older,
            compensated_newer,
            cv2=cv2,
        )

        suffix = "none" if blur_kernel == 0 else str(blur_kernel)
        _write_variant(output_dir / f"raw_diff_blur_{suffix}.jpg", raw_diff, cv2=cv2)
        _write_variant(
            output_dir / f"compensated_diff_blur_{suffix}.jpg",
            compensated_diff,
            cv2=cv2,
        )
        _write_variant(output_dir / f"raw_diff_blur_{suffix}_stretched.jpg", raw_diff, cv2=cv2, stretch=True)
        _write_variant(
            output_dir / f"compensated_diff_blur_{suffix}_stretched.jpg",
            compensated_diff,
            cv2=cv2,
            stretch=True,
        )

        summary["variants"][f"raw_diff_blur_{suffix}"] = _stats(raw_diff)
        summary["variants"][f"compensated_diff_blur_{suffix}"] = _stats(compensated_diff)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote motion variants to {output_dir}")
    print(f"Wrote summary to {summary_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump YOLOMG motion-map variants for one center video frame.",
    )
    parser.add_argument("video_path", help="Video file to inspect.")
    parser.add_argument("--output-dir", required=True, help="Directory for dumped images.")
    parser.add_argument("--frame-index", type=int, help="Center frame index to inspect.")
    parser.add_argument("--time-sec", type=float, help="Center timestamp in seconds.")
    parser.add_argument(
        "--temporal-radius",
        type=int,
        default=2,
        help="Frame distance from center to older/newer frames.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        action="append",
        default=[],
        help="Gaussian blur kernel to test. Use 0 for no blur. Repeat for multiple values.",
    )
    return parser.parse_args()


def _resolve_center_frame(
    args: argparse.Namespace,
    fps: float,
    total_frames: int,
) -> int:
    if args.frame_index is not None and args.time_sec is not None:
        raise SystemExit("Pass only one of --frame-index or --time-sec.")
    if args.frame_index is not None:
        center_index = args.frame_index
    elif args.time_sec is not None:
        center_index = round(args.time_sec * fps)
    else:
        center_index = 2
    return min(total_frames - 1, max(0, center_index))


def _read_frames(
    *,
    video_path: str,
    frame_indices: tuple[int, int, int],
    cv2: Any,
) -> dict[int, Any]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    try:
        frames = {}
        for frame_index in sorted(set(frame_indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Failed to read frame {frame_index} from {video_path}")
            frames[frame_index] = frame
        return frames
    finally:
        capture.release()


def _gray(frame: Any, *, cv2: Any, blur_kernel: int) -> Any:
    if blur_kernel > 0:
        if blur_kernel % 2 == 0:
            raise ValueError(f"Blur kernel must be odd or 0: {blur_kernel}")
        frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _average_diff(center_gray: Any, older_gray: Any, newer_gray: Any, *, cv2: Any) -> Any:
    diff_older = cv2.absdiff(center_gray, older_gray)
    diff_newer = cv2.absdiff(center_gray, newer_gray)
    return ((diff_older.astype(np.float32) + diff_newer.astype(np.float32)) / 2).astype(np.uint8)


def _write_variant(path: Path, image: Any, *, cv2: Any, stretch: bool = False) -> None:
    if stretch:
        max_value = int(image.max())
        if max_value > 0:
            image = np.clip((image.astype(np.float32) / max_value) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


def _stats(image: Any) -> dict[str, int | float]:
    return {
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
        "nonzero": int(np.count_nonzero(image)),
        "pixels": int(image.size),
    }


if __name__ == "__main__":
    raise SystemExit(main())
