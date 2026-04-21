from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_server.schemas import StabilizationMethod
from ai_server.services.tracker import AnalyzePipelineError, run_track_sequence_pipeline


def main() -> int:
    args = _parse_args()
    video_path = str(Path(args.video_path).expanduser().resolve())
    source_video_id = args.source_video_id or Path(video_path).stem

    methods: tuple[StabilizationMethod, ...] = ("none", "ffmpeg_vidstab")
    results: list[dict[str, object]] = []
    had_failure = False

    for method in methods:
        run_id = f"{source_video_id}__{method}"
        output_path = _tracks_output_path(run_id)

        try:
            response = run_track_sequence_pipeline(
                video_path=video_path,
                source_video_id=run_id,
                stabilization_method=method,
            )
            result = {
                "method": method,
                "status": "ok",
                "source_video_id": run_id,
                "track_count": len(response.tracks),
                "output_path": str(output_path),
                "message": response.message,
            }
        except AnalyzePipelineError as exc:
            had_failure = True
            result = {
                "method": method,
                "status": "error",
                "source_video_id": run_id,
                "output_path": str(output_path),
                "error": str(exc),
            }

        results.append(result)

    summary_path = _write_summary(source_video_id, results)
    _print_summary(results, summary_path)
    return 1 if had_failure else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A pipeline once with no stabilization and once with ffmpeg vidstab.",
    )
    parser.add_argument("video_path", help="Absolute or relative path to the sample video.")
    parser.add_argument(
        "--source-video-id",
        help="Base source video id. Defaults to the input filename stem.",
    )
    return parser.parse_args()


def _tracks_output_path(run_id: str) -> Path:
    return _repo_root() / "artifacts" / "tracks" / f"{run_id}.json"


def _write_summary(source_video_id: str, results: list[dict[str, object]]) -> Path:
    summary_path = _repo_root() / "artifacts" / "tracks" / f"{source_video_id}__compare_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "source_video_id": source_video_id,
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path


def _print_summary(results: list[dict[str, object]], summary_path: Path) -> None:
    print("A pipeline comparison summary")
    for result in results:
        status = result["status"]
        method = result["method"]
        output_path = result["output_path"]
        if status == "ok":
            print(
                f"- {method}: ok, tracks={result['track_count']}, json={output_path}",
            )
        else:
            print(
                f"- {method}: error, json={output_path}, reason={result['error']}",
            )
    print(f"- summary: {summary_path}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    sys.exit(main())
