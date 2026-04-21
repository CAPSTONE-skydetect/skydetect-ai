from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from ai_server.schemas import AnalyzeResponse


@dataclass(frozen=True)
class PersistenceResult:
    ok: bool
    output_path: str
    error: str | None = None


def persist_analyze_response(
    response: AnalyzeResponse,
    output_root: str | None = None,
) -> PersistenceResult:
    root = Path(output_root) if output_root else _default_output_root()
    root.mkdir(parents=True, exist_ok=True)

    filename = f"{_sanitize_filename(response.source_video_id)}.json"
    output_path = root / filename

    try:
        output_path.write_text(
            response.model_dump_json(indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        return PersistenceResult(
            ok=False,
            output_path=str(output_path),
            error=str(exc),
        )

    return PersistenceResult(
        ok=True,
        output_path=str(output_path),
    )


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "tracks"


def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "track_sequence"
