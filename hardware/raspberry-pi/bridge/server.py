"""Authenticated local bridge for Pi clip jobs and official AI APIs.

This is a local integration service, not a Spring Boot feature. It listens only
inside the Compose network; the frontend reverse proxy is the sole route in.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
from email.utils import formatdate
import hashlib
import http.server
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


DB_PATH = Path(os.environ.get("PI_BRIDGE_DB", "/runtime/local-bridge/jobs.sqlite3"))
RECORDING_DIR = Path(os.environ.get("PI_RECORDING_DIR", "/runtime/recordings/pi"))
CLIPS_DIR = Path(os.environ.get("PI_CLIPS_DIR", "/runtime/clips"))
STAGE_DIR = Path(os.environ.get("PI_STAGE_DIR", "/runtime/local-bridge/staging"))
PIN_DIR = Path(os.environ.get("PI_PIN_DIR", "/runtime/pins/segments"))
RELAY_STATE = Path(os.environ.get("PI_RELAY_STATE", "/runtime/status/state.json"))
BACKEND_URL = os.environ.get("PI_BACKEND_URL", "http://backend:8080").rstrip("/")
AI_URLS = {"rf": os.environ.get("PI_RF_URL", "http://ai-rf:8000").rstrip("/"),
           "minirocket": os.environ.get("PI_MINIROCKET_URL", "http://ai-minirocket:8000").rstrip("/")}
CLIP_MAX_BYTES = int(os.environ.get("PI_CLIP_MAX_BYTES", str(4 * 1024**3)))
CLIP_RESERVATION_BYTES = 48 * 1024**2
CLIP_PRE_SECONDS = 3.0
CLIP_POST_SECONDS = 10.0
SEGMENT_NAME = re.compile(r"^\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d-\d{6}_[+-]\d{4}\.mp4$")
CLIP_ID = re.compile(r"^clip_[0-9a-f]{32}$")
SPRING_CLIP_ID = re.compile(r"^spring_clip_[0-9a-f]{32}$")
ANALYSIS_ID = re.compile(r"^local_anl_[0-9a-f]{32}$")
EXPECTED_ORIGINS = {os.environ.get("PI_FRONTEND_ORIGIN", "http://127.0.0.1:5178")}
_DB_LOCK = threading.RLock()
_SPACE_LOCK = threading.Lock()
_SPRING_CLIP_SLOTS = threading.BoundedSemaphore(2)


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    # Runtime DB lives on a macOS/Docker bind mount. Avoid WAL's shared-memory
    # coordination across host/VM boundaries; keep one service as the DB owner.
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_db() -> None:
    for path in (DB_PATH.parent, CLIPS_DIR, STAGE_DIR, PIN_DIR):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    with connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS clips (
                clip_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                ready_at REAL NOT NULL,
                trigger_server_epoch REAL NOT NULL,
                client_click_epoch_ms REAL,
                displayed_media_time_s REAL,
                display_mapping TEXT NOT NULL,
                requested_start_epoch REAL NOT NULL,
                requested_end_epoch REAL NOT NULL,
                video_path TEXT,
                video_sha256 TEXT,
                video_bytes INTEGER,
                metadata_json TEXT,
                frame_map_json TEXT,
                timing_json TEXT NOT NULL,
                error TEXT,
                latest_analysis_id TEXT
            );
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id TEXT PRIMARY KEY,
                clip_id TEXT NOT NULL REFERENCES clips(clip_id),
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                request_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL,
                result_json TEXT,
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS analyses_clip_idx ON analyses(clip_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS clips_created_idx ON clips(created_at DESC);
        """)
        db.execute("UPDATE clips SET status='FAILED', error='bridge_restarted_before_clip_ready' WHERE status='PENDING'")
        db.execute("UPDATE analyses SET status='FAILED', error='bridge_restarted_during_analysis', finished_at=? WHERE status IN ('PENDING','RUNNING')", (time.time(),))
    try:
        os.chmod(DB_PATH, 0o600)
    except OSError:
        pass


def iso(epoch: float | None) -> str | None:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z") if epoch else None


def parse_segment_time(path: Path) -> float | None:
    if not SEGMENT_NAME.fullmatch(path.name):
        return None
    try:
        value = datetime.strptime(path.name[:-4], "%Y-%m-%d_%H-%M-%S-%f_%z")
        return value.timestamp()
    except ValueError:
        return None


def probe_video(path: Path, include_frames: bool = False) -> dict:
    entries = "format=duration,start_time:stream=codec_name,width,height,avg_frame_rate,r_frame_rate,time_base,start_time,duration,nb_read_frames"
    if include_frames:
        entries += ":frame=best_effort_timestamp_time,pkt_duration_time,key_frame"
    command = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
               "-show_entries", entries]
    if include_frames:
        command.append("-show_frames")
    command += ["-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError("ffprobe_failed: " + result.stderr[-500:])
    return json.loads(result.stdout)


def _rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return None


def _safe_error(exc: Exception) -> str:
    text = str(exc).replace("\n", " ").strip()
    # Keep errors useful without persisting cookies, authorization headers or URLs.
    text = re.sub(r"(?i)(cookie|authorization|token|password)[=: ]+[^ ,;]+", r"\1=[redacted]", text)
    return text[:1000] or type(exc).__name__


def backend_session_ok(cookie_header: str) -> bool:
    if not cookie_header:
        return False
    request = urllib.request.Request(BACKEND_URL + "/api/auth/me", headers={"Cookie": cookie_header})
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            return response.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return False


def csrf_ok(cookie_header: str, header_token: str | None) -> bool:
    if not cookie_header or not header_token:
        return False
    cookies = {}
    for pair in cookie_header.split(";"):
        if "=" in pair:
            key, value = pair.strip().split("=", 1)
            cookies[key] = urllib.parse.unquote(value)
    return bool(cookies.get("XSRF-TOKEN")) and cookies["XSRF-TOKEN"] == header_token


def _db_clip(clip_id: str) -> dict | None:
    with connect() as db:
        row = db.execute("SELECT * FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
    return dict(row) if row else None


def _clip_dto(row: dict) -> dict:
    metadata = json.loads(row["metadata_json"]) if row.get("metadata_json") else None
    frame_pts = json.loads(row["frame_map_json"]) if row.get("frame_map_json") else None
    timing = json.loads(row["timing_json"])
    result = {
        "clipId": row["clip_id"], "status": row["status"], "readyAt": iso(row["ready_at"]),
        "triggeredAt": iso(row["trigger_server_epoch"]), "serverReceivedEpochMs": round(row["trigger_server_epoch"] * 1000),
        "clientClickEpochMs": row["client_click_epoch_ms"], "displayedMediaTimeS": row["displayed_media_time_s"],
        "displayTimeMapping": row["display_mapping"], "timing": timing,
        "analysisId": row["latest_analysis_id"], "error": row["error"],
    }
    if row["status"] == "READY" and metadata:
        result["meta"] = {**metadata, "videoUrl": f"/local-api/clips/{row['clip_id']}/video",
                           "framePts": frame_pts}
        result["videoUrl"] = result["meta"]["videoUrl"]
        result.update({k: metadata.get(k) for k in ("fps", "width", "height", "frameCount", "durationMs")})
    return result


def stable_file(path: Path, min_age: float = 1.5) -> bool:
    try:
        first = path.stat()
        if not path.is_file() or path.is_symlink() or time.time() - first.st_mtime < min_age:
            return False
        time.sleep(0.25)
        second = path.stat()
        return first.st_size == second.st_size and first.st_mtime_ns == second.st_mtime_ns and second.st_size > 0
    except OSError:
        return False


def segments_in_window(start_epoch: float, end_epoch: float) -> list[Path]:
    found = []
    if not RECORDING_DIR.exists():
        return found
    for path in RECORDING_DIR.glob("*.mp4"):
        segment_start = parse_segment_time(path)
        if segment_start is None:
            continue
        # Bounded wide range includes a prior GOP/segment and the last future segment.
        if start_epoch - 12 <= segment_start <= end_epoch + 12:
            found.append(path)
    return sorted(found, key=lambda p: parse_segment_time(p) or 0)


def _write_pin(path: Path, clip_id: str) -> Path:
    marker = PIN_DIR / f"{path.name}.pin"
    marker.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker.write_text(clip_id + "\n", encoding="ascii")
    try:
        os.chmod(marker, 0o600)
    except OSError:
        pass
    return marker


def _copy_stable_segment(source: Path, destination: Path) -> bool:
    # fMP4 is flushed in parts; a temporarily quiet file is not finalized.
    # MediaMTX opens the next segment only after finishing its predecessor.
    start = parse_segment_time(source)
    if start is None or not any(
        p.is_file() and not p.is_symlink() and
        (parse_segment_time(p) or 0) > start
        for p in source.parent.glob("*.mp4")
    ):
        return False
    if not stable_file(source):
        return False
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copyfile(source, destination)
    after = source.stat()
    copied = destination.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or copied.st_size != after.st_size:
        destination.unlink(missing_ok=True)
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffprobe_frames(path: Path) -> tuple[dict, list[dict]]:
    raw = probe_video(path, include_frames=True)
    streams = raw.get("streams") or []
    if not streams:
        raise RuntimeError(f"no_video_stream:{path.name}")
    stream = streams[0]
    frames = []
    for frame in raw.get("frames", []):
        value = frame.get("best_effort_timestamp_time")
        if value is None:
            continue
        pts = float(value)
        duration = float(frame.get("pkt_duration_time") or 0)
        frames.append({"pts": pts, "duration": duration, "key": int(frame.get("key_frame", 0))})
    return {**stream, "format": raw.get("format", {})}, frames


def inspect_segments(parts: list[Path], *, allow_forward_gaps: bool = False) -> tuple[list[dict], list[dict]]:
    entries, source_frames = [], []
    previous_end = None
    prior_frame_epoch = None
    dt_values = []
    for index, part in enumerate(parts):
        start_epoch = parse_segment_time(part)
        if start_epoch is None:
            raise RuntimeError(f"unrecognized_segment_time:{part.name}")
        media, frames = _ffprobe_frames(part)
        if int(media.get("width") or 0) != 1920 or int(media.get("height") or 0) != 1080:
            raise RuntimeError(f"unexpected_recording_dimensions:{part.name}")
        if not frames:
            raise RuntimeError(f"no_frame_timestamps:{part.name}")
        timestamps = [frame["pts"] for frame in frames]
        for left, right in zip(timestamps, timestamps[1:]):
            delta = right - left
            if delta <= 0:
                raise RuntimeError(f"nonmonotonic_segment_pts:{part.name}")
            dt_values.append(delta)
            if delta > 0.12 and not allow_forward_gaps:
                raise RuntimeError(f"intra_segment_pts_gap_{delta:.3f}s:{part.name}")
        first_pts = float(media.get("start_time") or timestamps[0])
        duration = float((media.get("format") or {}).get("duration") or media.get("duration") or 0)
        if duration <= 0:
            duration = max(timestamps) + (frames[-1]["duration"] or 1 / 30) - first_pts
        if previous_end is not None:
            gap = start_epoch - previous_end
            if gap > 0.12 and not allow_forward_gaps:
                raise RuntimeError(f"inter_segment_time_gap_{gap:.3f}s:{part.name}")
            if gap < -0.12:
                raise RuntimeError(f"overlapping_segment_timeline_{gap:.3f}s:{part.name}")
        previous_end = start_epoch + duration
        entries.append({"name": part.name, "start_epoch": start_epoch, "duration_s": duration,
                        "first_pts_s": first_pts, "width": int(media["width"]),
                        "height": int(media["height"]), "fps": _rate(media.get("avg_frame_rate")),
                        "frame_count": len(frames), "sha256": _sha256(part)})
        for frame_index, frame in enumerate(frames):
            absolute = start_epoch + frame["pts"]
            if prior_frame_epoch is not None and absolute <= prior_frame_epoch:
                raise RuntimeError(f"nonmonotonic_source_timeline:{part.name}")
            source_frames.append({"server_media_epoch": absolute, "segment": part.name,
                                  "segment_frame_index": frame_index, "key_frame": frame["key"]})
            prior_frame_epoch = absolute
    median_dt = sorted(dt_values)[len(dt_values) // 2] if dt_values else 1 / 30
    for left, right in zip(source_frames, source_frames[1:]):
        delta = right["server_media_epoch"] - left["server_media_epoch"]
        if delta > max(0.12, median_dt * 2.5) and not allow_forward_gaps:
            raise RuntimeError(f"source_frame_gap_{delta:.3f}s")
    return entries, source_frames


def segment_window_coverage(parts: list[Path]) -> tuple[float, float] | None:
    """Return coarse finalized segment bounds without judging PTS continuity.

    The strict frame/PTS validator runs only after stable segments cover the
    requested future endpoint; otherwise a not-yet-recorded tail could look
    like a permanent gap.
    """
    if not parts:
        return None
    bounds = []
    for part in parts:
        start_epoch = parse_segment_time(part)
        if start_epoch is None:
            raise RuntimeError(f"unrecognized_segment_time:{part.name}")
        raw = probe_video(part)
        streams = raw.get("streams") or []
        if not streams:
            raise RuntimeError(f"no_video_stream:{part.name}")
        stream = streams[0]
        duration = float((raw.get("format") or {}).get("duration") or stream.get("duration") or 0)
        first_pts = float(stream.get("start_time") or (raw.get("format") or {}).get("start_time") or 0)
        if duration <= 0:
            continue
        bounds.append((start_epoch + first_pts, start_epoch + duration))
    if not bounds:
        return None
    return min(item[0] for item in bounds), max(item[1] for item in bounds)


def requested_window_covered(parts: list[Path], start_epoch: float, end_epoch: float) -> bool:
    coverage = segment_window_coverage(parts)
    return bool(coverage and coverage[0] <= start_epoch + 0.06 and coverage[1] >= end_epoch - 0.08)


def parts_overlapping_window(parts: list[Path], start_epoch: float, end_epoch: float) -> list[Path]:
    selected = []
    for part in parts:
        start = parse_segment_time(part)
        if start is None or start >= end_epoch:
            continue
        raw = probe_video(part)
        stream = (raw.get("streams") or [{}])[0]
        duration = float((raw.get("format") or {}).get("duration") or stream.get("duration") or 0)
        if start + duration > start_epoch:
            selected.append(part)
    return selected


def _concat_file(parts: list[Path], work_dir: Path) -> Path:
    target = work_dir / "segments.ffconcat"
    lines = ["ffconcat version 1.0"]
    for path in parts:
        escaped = str(path).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _select_contiguous_frames(frames: list[dict], trigger_epoch: float | None) -> tuple[list[dict], float | None]:
    cuts = [(index, right["server_media_epoch"] - left["server_media_epoch"])
            for index, (left, right) in enumerate(zip(frames, frames[1:]), start=1)
            if right["server_media_epoch"] - left["server_media_epoch"] > 0.12]
    if not cuts:
        return frames, None
    largest_gap = max(gap for _, gap in cuts)
    boundaries = [0, *(index for index, _ in cuts), len(frames)]
    for start, end in zip(boundaries, boundaries[1:]):
        first = frames[start]["server_media_epoch"]
        last = frames[end - 1]["server_media_epoch"]
        if trigger_epoch is not None and first <= trigger_epoch <= last and last - first >= 3:
            return frames[start:end], largest_gap
    raise RuntimeError("recording_gap_no_contiguous_trigger_window")


def _extract_clip(parts: list[Path], stage: Path, output: Path, start_epoch: float, end_epoch: float,
                  *, trigger_epoch: float | None = None) -> dict:
    parts = parts_overlapping_window(parts, start_epoch, end_epoch)
    segment_meta, all_frames = inspect_segments(parts, allow_forward_gaps=True)
    selected = [frame for frame in all_frames
                if start_epoch <= frame["server_media_epoch"] < end_epoch]
    if len(selected) < 2 or selected[0]["server_media_epoch"] > start_epoch + 0.06:
        raise RuntimeError("requested_pre_roll_not_covered_by_source_frames")
    if selected[-1]["server_media_epoch"] < end_epoch - 0.08:
        raise RuntimeError("requested_post_roll_not_covered_by_source_frames")
    selected, largest_gap = _select_contiguous_frames(selected, trigger_epoch)
    selected_start = selected[0]["server_media_epoch"]
    selected_end = selected[-1]["server_media_epoch"]
    first_index = all_frames.index(selected[0])
    last_index = first_index + len(selected) - 1
    # Re-encode at the original 1080p with the installed software H.264 encoder
    # to make -3s/+10s frame-accurate and seekable. No scaling is applied.
    concat = _concat_file(parts, stage)
    temporary = output.with_suffix(".partial.mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
               "-i", str(concat),
               "-vf", f"select=between(n\\,{first_index}\\,{last_index}),setpts=PTS-STARTPTS",
               "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-fps_mode", "passthrough", "-video_track_timescale", "90000", "-movflags", "+faststart"]
    if largest_gap is not None:
        # Do not carry the removed gap into the final sample's display duration.
        command += ["-bf", "0", "-bsf:v", "setts=duration=1/(30*TB)"]
    command.append(str(temporary))
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg_clip_extract_failed: " + completed.stderr[-700:])
    try:
        output_probe = probe_video(temporary, include_frames=True)
        stream = output_probe["streams"][0]
        out_frames = [float(frame["best_effort_timestamp_time"]) for frame in output_probe.get("frames", [])
                      if frame.get("best_effort_timestamp_time") is not None]
        if not out_frames or len(out_frames) != len(selected):
            raise RuntimeError(f"output_frame_mapping_count_mismatch:{len(out_frames)}:{len(selected)}")
        if any(right <= left for left, right in zip(out_frames, out_frames[1:])):
            raise RuntimeError("output_pts_nonmonotonic")
        if any((right - left) > 0.12 for left, right in zip(out_frames, out_frames[1:])):
            raise RuntimeError("output_pts_gap_over_120ms")
        duration = float((output_probe.get("format") or {}).get("duration") or 0)
        if largest_gap is not None and duration - out_frames[-1] > 0.12:
            raise RuntimeError("partial_clip_final_frame_duration_over_120ms")
        if int(stream.get("width") or 0) != 1920 or int(stream.get("height") or 0) != 1080:
            raise RuntimeError("output_resolution_changed")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    map_rows = []
    for index, (frame, clip_pts) in enumerate(zip(selected, out_frames)):
        map_rows.append({"frame_index": index, "clip_pts_s": clip_pts,
                         "source_server_media_epoch": frame["server_media_epoch"],
                         "source_server_media_iso": iso(frame["server_media_epoch"]),
                         "source_segment": frame["segment"],
                         "source_segment_frame_index": frame["segment_frame_index"]})
    fps = _rate(stream.get("avg_frame_rate")) or 30.0
    metadata = {"width": 1920, "height": 1080, "fps": fps, "frameCount": len(map_rows),
                "durationMs": round(duration * 1000), "timeBase": stream.get("time_base"),
                "frameMapVersion": 1}
    timing = {"requestedStartEpoch": start_epoch, "requestedEndEpoch": end_epoch,
              "actualFirstSourceFrameEpoch": selected_start, "actualLastSourceFrameEpoch": selected_end,
              "requestedSpanSeconds": end_epoch - start_epoch,
              "actualSampledSpanSeconds": selected_end - selected_start,
              "shortenedByGap": largest_gap is not None,
              "largestExcludedGapSeconds": largest_gap,
              "sourceClock": "MediaMTX segment wall-time + per-segment PTS; not verified sensor capture wall-clock",
              "browserDisplayedMediaTimeIsCaptureTime": False,
              "serverReceivedEpoch": trigger_epoch if trigger_epoch is not None else start_epoch + CLIP_PRE_SECONDS,
              "frameCount": len(map_rows)}
    return {"metadata": metadata, "frame_map": map_rows, "timing": timing,
            "segments": segment_meta, "video_sha256": _sha256(output), "video_bytes": output.stat().st_size}


def _materialize_clip(clip_id: str, trigger_epoch: float) -> dict:
    start_epoch = trigger_epoch - CLIP_PRE_SECONDS
    end_epoch = trigger_epoch + CLIP_POST_SECONDS
    stage = STAGE_DIR / clip_id
    part_dir = stage / "parts"
    markers: set[Path] = set()
    copied: dict[str, Path] = {}
    deadline = end_epoch + 35
    try:
        stage.mkdir(parents=True, exist_ok=not SPRING_CLIP_ID.fullmatch(clip_id), mode=0o700)
        part_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        _atomic = stage / "job.json"
        _atomic.write_text(json.dumps({"clip_id": clip_id, "requested_start_epoch": start_epoch,
                                       "requested_end_epoch": end_epoch,
                                       "trigger_server_epoch": trigger_epoch}, indent=2), encoding="utf-8")
        while time.time() < deadline:
            for source in segments_in_window(start_epoch, end_epoch):
                marker = _write_pin(source, clip_id)
                markers.add(marker)
                destination = part_dir / source.name
                if source.name not in copied and _copy_stable_segment(source, destination):
                    copied[source.name] = destination
                    manifest_path = stage / "pinned_segments.json"
                    manifest_path.write_text(json.dumps([{"name": p.name, "sha256": _sha256(copied[p.name])}
                                                         for p in sorted(copied.values())], indent=2), encoding="utf-8")
            candidates = sorted(copied.values(), key=lambda p: parse_segment_time(p) or 0)
            if candidates:
                if requested_window_covered(candidates, start_epoch, end_epoch):
                    break
            time.sleep(0.5)
        else:
            raise RuntimeError("recording_segments_did_not_cover_requested_minus3_plus10_before_timeout")
        candidates = sorted(copied.values(), key=lambda p: parse_segment_time(p) or 0)
        # Keep one previous segment for decoder preroll; its frame mapping is
        # retained, while only the requested interval is emitted to the user.
        first_needed = 0
        for index, item in enumerate(candidates):
            if parse_segment_time(item) is not None and parse_segment_time(item) <= start_epoch:
                first_needed = max(0, index - 1)
        candidates = candidates[first_needed:]
        output = CLIPS_DIR / f"{clip_id}.mp4"
        with _SPACE_LOCK:
            used = sum(p.stat().st_size for p in CLIPS_DIR.glob("*.mp4") if p.is_file())
            if used + CLIP_RESERVATION_BYTES > CLIP_MAX_BYTES:
                raise RuntimeError("completed_clip_storage_cap_reached; existing clips were preserved")
            result = _extract_clip(candidates, stage, output, start_epoch, end_epoch, trigger_epoch=trigger_epoch)
        (stage / "source_frame_map.json").write_text(json.dumps(result["frame_map"], indent=2), encoding="utf-8")
        (stage / "timing_and_source.json").write_text(json.dumps({"timing": result["timing"],
            "segments": result["segments"], "video_sha256": result["video_sha256"]}, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        (stage / "failure.json").write_text(json.dumps({"error": _safe_error(exc), "time": iso(time.time()),
                                                       "copied_segment_names": sorted(copied)}, indent=2), encoding="utf-8")
        raise
    finally:
        for marker in markers:
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                pass


def _clip_worker(clip_id: str) -> None:
    row = _db_clip(clip_id)
    if not row:
        return
    try:
        result = _materialize_clip(clip_id, row["trigger_server_epoch"])
        with connect() as db:
            db.execute("UPDATE clips SET status='READY', video_path=?, video_sha256=?, video_bytes=?, metadata_json=?, frame_map_json=?, timing_json=?, error=NULL WHERE clip_id=?",
                       (str(CLIPS_DIR / f"{clip_id}.mp4"), result["video_sha256"], result["video_bytes"],
                        json.dumps(result["metadata"]), json.dumps([r["clip_pts_s"] for r in result["frame_map"]]),
                        json.dumps(result["timing"]), clip_id))
    except Exception as exc:
        with connect() as db:
            db.execute("UPDATE clips SET status='FAILED', error=? WHERE clip_id=?", (_safe_error(exc), clip_id))


def _relay_is_live() -> bool:
    try:
        state = json.loads(RELAY_STATE.read_text(encoding="utf-8"))
        age = time.time() - float(state["last_media_progress_at"])
        return state.get("status") == "running" and -2 <= age <= 20
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _multipart_upload(base_url: str, video_path: Path, filename: str) -> dict:
    boundary = "SkyDetect" + uuid.uuid4().hex
    prefix = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
              "Content-Type: video/mp4\r\n\r\n").encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()
    data = prefix + video_path.read_bytes() + suffix
    request = urllib.request.Request(base_url + "/api/videos/upload", data=data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def _post_json(url: str, payload: dict, timeout: int = 600) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _fetch_trajectory(base: str, url: str, track: dict) -> list[dict] | None:
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    request_url = base + parsed.path + ("?" + parsed.query if parsed.query else "")
    try:
        with urllib.request.urlopen(request_url, timeout=20) as response:
            text = response.read().decode("utf-8")
    except Exception:
        return None
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    header = lines[0].split(",")
    result = []
    width = float(track.get("processed_width") or 1)
    height = float(track.get("processed_height") or 1)
    for line in lines[1:]:
        values = line.split(",")
        row = dict(zip(header, values))
        try:
            result.append({"frameIndex": int(row["frame_index"]),
                           "cx": float(row["raw_x"]) / width,
                           "cy": float(row["raw_y"]) / height,
                           "w": float(row["bbox_width"]) / width,
                           "h": float(row["bbox_height"]) / height,
                           "conf": float(row["confidence"]),
                           "visible": row.get("visible", "True").lower() == "true",
                           "source": row.get("tracking_source")})
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _analysis_worker(analysis_id: str) -> None:
    with connect() as db:
        row = db.execute("SELECT a.*, c.video_path, c.status AS clip_status, c.metadata_json FROM analyses a JOIN clips c USING(clip_id) WHERE analysis_id=?",
                         (analysis_id,)).fetchone()
    if not row:
        return
    request_data = json.loads(row["request_json"])
    base = AI_URLS[row["model"]]
    started = time.time()
    with connect() as db:
        db.execute("UPDATE analyses SET status='RUNNING', started_at=? WHERE analysis_id=?", (started, analysis_id))
    try:
        if row["clip_status"] != "READY" or not Path(row["video_path"]).is_file():
            raise RuntimeError("clip_file_missing_or_not_ready")
        uploaded = _multipart_upload(base, Path(row["video_path"]), row["clip_id"] + ".mp4")
        payload = {
            "source_video_id": uploaded["source_video_id"], "video_path": uploaded["video_path"],
            "init_frame_index": request_data["initFrameIndex"], "target_bbox": request_data["targetBbox"],
            "stabilize": True, "resize_width": 1280, "write_overlay": True,
            "max_seconds": request_data["maxSeconds"],
            "tuning": {"klt_accept_conf": 0.40, "recovery_conf": 0.58, "update_conf": 0.76,
                       "tracker_variant": request_data.get("trackerVariant", "d0"),
                       "search_radius_multiplier": 2.5, "online_update_enabled": False},
        }
        output = _post_json(base + "/api/tracks/manual", payload)
        track = (output.get("tracks") or [None])[0]
        if track is None:
            raise RuntimeError("AI response did not contain TrackSequence")
        urls = output.get("download_urls") or {}
        prefix = "/ai/" if row["model"] == "rf" else "/ai-minirocket/"
        trajectory_url = prefix + (urls.get("trajectory") or "").lstrip("/")
        overlay_url = prefix + (urls.get("overlay") or "").lstrip("/")
        track_sequence_url = prefix + (urls.get("track_sequence") or "").lstrip("/")
        points = _fetch_trajectory(base, urls.get("trajectory"), track)
        prediction = output.get("prediction") or {}
        result = {
            "analysisId": analysis_id, "clipId": row["clip_id"], "status": "DONE", "model": row["model"],
            "prediction": prediction, "track": track, "metadata": output.get("metadata"),
            "metrics": output.get("metrics"), "features": output.get("features"),
            "trajectoryUrl": trajectory_url, "overlayUrl": overlay_url, "points": points,
            "trackSequenceUrl": track_sequence_url if urls.get("track_sequence") else None,
            "request": request_data, "elapsedMs": round((time.time() - started) * 1000),
        }
        with connect() as db:
            db.execute("UPDATE analyses SET status='DONE', finished_at=?, result_json=?, error=NULL WHERE analysis_id=?",
                       (time.time(), json.dumps(result, ensure_ascii=False), analysis_id))
    except Exception as exc:
        with connect() as db:
            db.execute("UPDATE analyses SET status='FAILED', finished_at=?, error=? WHERE analysis_id=?",
                       (time.time(), _safe_error(exc), analysis_id))


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "SkyDetectLocalBridge/1.0"

    def log_message(self, _format: str, *_args) -> None:
        # Never log request headers, cookies, query strings, or arbitrary paths.
        print(f"{self.log_date_time_string()} {self.command} route={self.path.split('?', 1)[0][:80]}", flush=True)

    def _send_json(self, status: int, value: dict) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"message": message})

    def _authorized(self, mutate: bool = False) -> bool:
        cookie = self.headers.get("Cookie", "")
        if mutate:
            origin = self.headers.get("Origin")
            if origin not in EXPECTED_ORIGINS:
                self._error(403, "요청 출처를 확인할 수 없습니다.")
                return False
            if not csrf_ok(cookie, self.headers.get("X-XSRF-TOKEN")):
                self._error(403, "CSRF 검증에 실패했습니다.")
                return False
        if not backend_session_ok(cookie):
            self._error(401, "로그인이 필요합니다.")
            return False
        return True

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > 64 * 1024:
            raise ValueError("요청 크기가 허용 범위를 넘었습니다.")
        parsed = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(parsed, dict):
            raise ValueError("JSON 객체가 필요합니다.")
        return parsed

    def do_GET(self) -> None:
        if not self._authorized():
            return
        path = urllib.parse.urlsplit(self.path).path
        if path == "/healthz":
            self._send_json(200, {"status": "ok"})
            return
        if path == "/clips":
            with connect() as db:
                rows = db.execute("SELECT * FROM clips ORDER BY created_at DESC LIMIT 100").fetchall()
            self._send_json(200, {"clips": [_clip_dto(dict(row)) for row in rows]})
            return
        if path == "/analyses":
            with connect() as db:
                rows = db.execute("SELECT analysis_id,clip_id,status,model,created_at,error FROM analyses ORDER BY created_at DESC LIMIT 100").fetchall()
            self._send_json(200, {"analyses": [{**dict(row), "createdAt": iso(row["created_at"])} for row in rows]})
            return
        match = re.fullmatch(r"/clips/([^/]+)(?:/(video))?", path)
        if match:
            clip_id, video = match.groups()
            if not CLIP_ID.fullmatch(clip_id):
                self._error(404, "클립을 찾을 수 없습니다.")
                return
            row = _db_clip(clip_id)
            if not row:
                self._error(404, "클립을 찾을 수 없습니다.")
                return
            if video:
                self._serve_video(row)
            else:
                self._send_json(200, _clip_dto(row))
            return
        match = re.fullmatch(r"/analyses/(local_anl_[0-9a-f]{32})", path)
        if match:
            with connect() as db:
                row = db.execute("SELECT * FROM analyses WHERE analysis_id=?", (match.group(1),)).fetchone()
            if not row:
                self._error(404, "분석 이력을 찾을 수 없습니다.")
                return
            result = json.loads(row["result_json"]) if row["result_json"] else None
            self._send_json(200, {"analysisId": row["analysis_id"], "clipId": row["clip_id"],
                "status": row["status"], "model": row["model"], "result": result,
                "message": row["error"], "request": json.loads(row["request_json"])})
            return
        self._error(404, "요청한 경로를 찾을 수 없습니다.")

    def do_POST(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        internal = re.fullmatch(r"/internal/clips/(spring_clip_[0-9a-f]{32})", path)
        if internal:
            # This route is not forwarded by nginx; only the Compose network can reach it.
            self._materialize_spring_clip(internal.group(1))
            return
        if not self._authorized(mutate=True):
            return
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, _safe_error(exc))
            return
        if path == "/clips":
            self._create_clip(body)
            return
        match = re.fullmatch(r"/clips/([^/]+)/analysis", path)
        if match:
            self._create_analysis(match.group(1), body)
            return
        self._error(404, "요청한 경로를 찾을 수 없습니다.")

    def _materialize_spring_clip(self, clip_id: str) -> None:
        try:
            trigger = float(self._body()["triggerEpoch"])
            if not math.isfinite(trigger) or not -2 <= time.time() - trigger <= 30:
                raise ValueError("invalid clip trigger time")
        except (KeyError, TypeError, ValueError):
            self._error(400, "invalid clip trigger time")
            return
        if not _relay_is_live():
            self._error(503, "카메라 라이브 수신이 확인되지 않습니다.")
            return
        if not _SPRING_CLIP_SLOTS.acquire(blocking=False):
            self._error(503, "클립 저장 작업이 진행 중입니다.")
            return
        try:
            if (STAGE_DIR / clip_id).exists() or (CLIPS_DIR / f"{clip_id}.mp4").exists():
                self._error(409, "clip ID already used")
                return
            result = _materialize_clip(clip_id, trigger)
            self._send_json(200, {
                "metadata": result["metadata"],
                "framePts": [frame["clip_pts_s"] for frame in result["frame_map"]],
                "timing": result["timing"],
                "sha256": result["video_sha256"],
                "bytes": result["video_bytes"],
            })
        except Exception as exc:
            self._error(422, _safe_error(exc))
        finally:
            _SPRING_CLIP_SLOTS.release()

    def _create_clip(self, body: dict) -> None:
        try:
            client_click = body.get("clientClickEpochMs")
            display_time = body.get("displayedMediaTimeS")
            if client_click is not None and (not isinstance(client_click, (int, float)) or not math.isfinite(client_click)):
                raise ValueError("clientClickEpochMs must be finite")
            if display_time is not None and (not isinstance(display_time, (int, float)) or not math.isfinite(display_time) or display_time < 0):
                raise ValueError("displayedMediaTimeS must be finite and nonnegative")
            if not _relay_is_live():
                self._error(503, "카메라 릴레이의 실제 영상 진행이 확인되지 않습니다. 상태를 확인해 주세요.")
                return
            received = time.time()
            start_epoch = received - CLIP_PRE_SECONDS
            end_epoch = received + CLIP_POST_SECONDS
            clip_id = "clip_" + uuid.uuid4().hex
            ready_at = end_epoch + 13
            with _SPACE_LOCK, connect() as db:
                used = sum(p.stat().st_size for p in CLIPS_DIR.glob("*.mp4") if p.is_file())
                pending = db.execute("SELECT COUNT(*) AS n FROM clips WHERE status='PENDING'").fetchone()["n"]
                if used + (pending + 1) * CLIP_RESERVATION_BYTES > CLIP_MAX_BYTES:
                    self._error(507, "완료 클립 저장공간 상한에 도달했습니다. 기존 파일은 삭제하지 않았습니다.")
                    return
                db.execute("INSERT INTO clips(clip_id,status,created_at,ready_at,trigger_server_epoch,client_click_epoch_ms,displayed_media_time_s,display_mapping,requested_start_epoch,requested_end_epoch,timing_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (clip_id, "PENDING", received, ready_at, received, client_click, display_time,
                            "HLS media time recorded separately; no sensor-wall-clock mapping claimed",
                            start_epoch, end_epoch,
                            json.dumps({"anchor": "server_received_at", "client_click_epoch_ms": client_click,
                                "displayed_media_time_s": display_time,
                                "display_clock_mapping": "unknown_not_sensor_capture_time",
                                "requested_pre_seconds": CLIP_PRE_SECONDS, "requested_post_seconds": CLIP_POST_SECONDS,
                                "server_received_at": iso(received)})))
            # Pin currently existing candidate segments synchronously before a
            # worker starts waiting for the +10s tail. Future segments are pinned
            # by the worker as soon as they appear.
            for segment in segments_in_window(start_epoch, end_epoch):
                _write_pin(segment, clip_id)
            threading.Thread(target=_clip_worker, args=(clip_id,), name=f"clip-{clip_id[-8:]}", daemon=True).start()
            self._send_json(202, {"clipId": clip_id, "status": "PENDING", "readyAt": iso(ready_at),
                "triggeredAt": iso(received), "serverReceivedEpochMs": round(received * 1000),
                "clientClickEpochMs": client_click, "displayedMediaTimeS": display_time,
                "displayTimeMapping": "unknown_not_sensor_capture_time"})
        except Exception as exc:
            self._error(503, "클립 작업을 시작할 수 없습니다: " + _safe_error(exc))

    def _create_analysis(self, clip_id: str, body: dict) -> None:
        if not CLIP_ID.fullmatch(clip_id):
            self._error(404, "클립을 찾을 수 없습니다.")
            return
        row = _db_clip(clip_id)
        if not row or row["status"] != "READY":
            self._error(409, "준비된 클립에서만 분석을 시작할 수 있습니다.")
            return
        model = body.get("model")
        bbox = body.get("targetBbox")
        init_frame = body.get("initFrameIndex")
        max_seconds = body.get("maxSeconds")
        tracker_variant = body.get("trackerVariant", "d0")
        if tracker_variant not in ("d0", "central_v1", "central_handoff_v1"):
            self._error(400, "지원하지 않는 추적 방식입니다.")
            return
        metadata = json.loads(row["metadata_json"])
        if model not in AI_URLS:
            self._error(400, "model은 rf 또는 minirocket이어야 합니다.")
            return
        if not isinstance(init_frame, int) or init_frame < 0 or init_frame >= metadata["frameCount"]:
            self._error(400, "초기 프레임이 클립 범위 밖입니다.")
            return
        if (not isinstance(bbox, list) or len(bbox) != 4 or
            any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in bbox)):
            self._error(400, "targetBbox는 유한한 [x,y,w,h] 배열이어야 합니다.")
            return
        x, y, width, height = bbox
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > metadata["width"] or y + height > metadata["height"]:
            self._error(400, "ROI가 원본 영상 밖에 있습니다.")
            return
        if not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or max_seconds <= 0:
            self._error(400, "분석 구간 길이를 지정해 주세요.")
            return
        start_s = float(json.loads(row["frame_map_json"])[init_frame])
        duration_s = float(max_seconds)
        if start_s + duration_s > metadata["durationMs"] / 1000 + 0.05:
            self._error(400, "선택한 분석 구간이 저장 클립 범위를 벗어납니다.")
            return
        analysis_id = "local_anl_" + uuid.uuid4().hex
        request_data = {"initFrameIndex": init_frame, "targetBbox": [int(round(v)) for v in bbox],
                        "maxSeconds": duration_s, "intervalStartClipSeconds": start_s,
                        "intervalEndClipSeconds": start_s + duration_s, "model": model,
                        "trackerVariant": tracker_variant}
        with connect() as db:
            db.execute("INSERT INTO analyses(analysis_id,clip_id,status,model,request_json,created_at) VALUES(?,?,?,?,?,?)",
                       (analysis_id, clip_id, "PENDING", model, json.dumps(request_data), time.time()))
            db.execute("UPDATE clips SET latest_analysis_id=? WHERE clip_id=?", (analysis_id, clip_id))
        threading.Thread(target=_analysis_worker, args=(analysis_id,), name=f"analysis-{analysis_id[-8:]}", daemon=True).start()
        self._send_json(202, {"analysisId": analysis_id, "clipId": clip_id, "status": "PENDING",
                              "model": model, "request": request_data})

    def _serve_video(self, row: dict) -> None:
        if row["status"] != "READY" or not row["video_path"]:
            self._error(409, "클립 영상이 아직 준비되지 않았습니다.")
            return
        path = Path(row["video_path"]).resolve()
        if path.parent != CLIPS_DIR.resolve() or path.name != f"{row['clip_id']}.mp4" or not path.is_file():
            self._error(404, "클립 영상을 찾을 수 없습니다.")
            return
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self._error(416, "지원하지 않는 byte range입니다.")
                return
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), size - 1) if right else size - 1
            elif right:
                start = max(0, size - int(right))
            if start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Last-Modified", formatdate(path.stat().st_mtime, usegmt=True))
        self.send_header("Cache-Control", "private, no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    initialize_db()
    server = Server(("0.0.0.0", 8080), Handler)
    print("local bridge ready; internal Compose network only", flush=True)
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
