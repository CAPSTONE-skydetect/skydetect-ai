"""Explicit, loopback-only supervisor for the SkyDetect Pi camera relay."""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid

AI_REPO = Path(__file__).resolve().parents[2]
DOCKER_DIR = AI_REPO / "docker"
PI_ENV = DOCKER_DIR / ".env.pi"


def _pi_settings(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    settings = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip().startswith("PI_"):
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            settings[key.strip()] = value
    return settings


SETTINGS = _pi_settings(PI_ENV)


def _setting(name: str, default: str) -> str:
    return os.environ.get(name, SETTINGS.get(name, default))


ROOT = (DOCKER_DIR / Path(_setting("PI_RELAY_RUNTIME", "../../.skydetect-pi")).expanduser()).resolve()
CAMERA_SSH = _setting("PI_CAMERA_SSH", "")
CAMERA_HOST_KEY_ALIAS = _setting("PI_CAMERA_HOST_KEY_ALIAS", "")
FFMPEG = _setting("PI_FFMPEG", "ffmpeg")
STATE = ROOT / "status/state.json"
LOCK = ROOT / ".supervisor.lock"
MEDIA = (DOCKER_DIR / Path(_setting("PI_MEDIAMTX", str(ROOT / "bin/mediamtx"))).expanduser()).resolve()
MEDIA_CONFIG = Path(__file__).resolve().parent / "mediamtx.yml"
RECORDINGS = ROOT / "recordings/pi"
PIN_DIR = ROOT / "pins/segments"
OWNED_MANIFEST = ROOT / "owned-rolling-files.json"
LOG_DIR = ROOT / "logs"
PORTS = (18554, 18888, 19997)
HLS_URL = "http://127.0.0.1:18888/pi/index.m3u8"
BACKOFF = (1, 2, 5, 10, 20)
HEARTBEAT_SECONDS = 4
PROGRESS_TIMEOUT = 18
MAX_RAW_BYTES = int(os.environ.get("PI_RELAY_MAX_RAW_BYTES", str(2 * 1024**3)))
SEGMENT_FILE = re.compile(r"^\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d-\d{6}_[+-]\d{4}\.mp4$")

# A long hard ceiling is a last-resort orphan safeguard, not a per-minute restart.
# The 15-second renewable lease normally stops capture when SSH disappears.
REMOTE_CAMERA = r'''import fcntl, os, pathlib, select, signal, subprocess, sys, time
root = pathlib.Path.home() / ".local/state/skydetect-pi-local"
root.mkdir(parents=True, exist_ok=True)
lock_fd = os.open(root / "camera.lock", os.O_CREAT | os.O_RDWR, 0o600)
try:
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("CAMERA_LEASE_BUSY", file=sys.stderr, flush=True)
    raise SystemExit(42)
os.set_inheritable(lock_fd, True)
stopping = False
def stop(_sig, _frame):
    global stopping
    stopping = True
for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, stop)
command = ["rpicam-vid", "--nopreview", "--width", "1920", "--height", "1080",
           "--framerate", "30", "--timeout", "21600000", "--codec", "libav",
           "--libav-video-codec", "h264_v4l2m2m", "--libav-format", "mpegts",
           "--bitrate", "8000000", "--intra", "30", "--low-latency", "--output", "-"]
camera = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=sys.stdout.buffer,
                          stderr=sys.stderr, start_new_session=True, pass_fds=(lock_fd,))
deadline = time.monotonic() + 15
try:
    while not stopping and camera.poll() is None:
        ready, _, _ = select.select([sys.stdin], [], [], 1)
        if ready:
            line = sys.stdin.readline()
            if not line:
                break
            if line.strip() == "renew":
                deadline = time.monotonic() + 15
        if time.monotonic() > deadline:
            print("CAMERA_LEASE_EXPIRED", file=sys.stderr, flush=True)
            break
finally:
    if camera.poll() is None:
        try:
            os.killpg(camera.pid, signal.SIGTERM)
            camera.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(camera.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            camera.wait()
    os.close(lock_fd)
raise SystemExit(camera.returncode or 0)
'''


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _read_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _pid_is_ours(pid: int) -> bool:
    try:
        command = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "command="],
                                 capture_output=True, text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return str(Path(__file__).resolve()) in command and "_supervisor" in command


def _ports_available() -> None:
    for port in PORTS:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"Required loopback port {port} is occupied; no process was stopped") from exc


def _fetch_playlist(url: str, depth: int = 0) -> tuple[str | None, str | None]:
    if depth > 2:
        return None, None
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            text = response.read(256_000).decode("utf-8", "replace")
    except Exception:
        return None, None
    media_lines = [line.strip() for line in text.splitlines() if line.strip()]
    parts = [line for line in media_lines if line.startswith("#EXT-X-PART:")]
    sequences = [line for line in media_lines if line.startswith("#EXT-X-MEDIA-SEQUENCE:")]
    if parts or sequences:
        last_part = parts[-1] if parts else ""
        uri = ""
        if 'URI="' in last_part:
            uri = last_part.split('URI="', 1)[1].split('"', 1)[0]
        elif media_lines:
            uri = next((line for line in reversed(media_lines) if not line.startswith("#")), "")
        # A new HLS reader gets a new session query even if media is stalled.
        return sequences[-1] if sequences else None, urllib.parse.urlsplit(uri).path if uri else None
    child = next((line for line in reversed(media_lines)
                  if not line.startswith("#") and
                  urllib.parse.urlsplit(line).path.endswith(".m3u8")), None)
    if child:
        return _fetch_playlist(urllib.parse.urljoin(url, child), depth + 1)
    return None, None


def _media_fingerprint() -> str | None:
    sequence, uri = _fetch_playlist(HLS_URL)
    if not uri:
        return None
    return f"{sequence or ''}|{uri}"


def _load_owned() -> set[str]:
    try:
        values = json.loads(OWNED_MANIFEST.read_text(encoding="utf-8"))
        return set(values if isinstance(values, list) else [])
    except (OSError, json.JSONDecodeError):
        return set()


def _rolling_gc(state: dict, now: float) -> None:
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    PIN_DIR.mkdir(parents=True, exist_ok=True)
    owned = _load_owned()
    candidates = [p for p in RECORDINGS.glob("*.mp4") if p.is_file() and not p.is_symlink()]
    started_at = float(state.get("started_at_epoch", now))
    # Only files first observed after this supervisor began are eligible for
    # our disk-ceiling cleanup. Existing recordings are never adopted/deleted.
    for item in candidates:
        if not SEGMENT_FILE.fullmatch(item.name):
            continue
        try:
            if item.stat().st_mtime >= started_at - 2:
                owned.add(item.name)
        except OSError:
            continue
    _atomic_json(OWNED_MANIFEST, sorted(owned))
    total = sum(p.stat().st_size for p in candidates if p.exists())
    state["rolling_bytes"] = total
    state["rolling_limit_bytes"] = MAX_RAW_BYTES
    if total <= MAX_RAW_BYTES:
        return
    # Avoid the newest potentially active segment and any segment pinned by a
    # clip job. MediaMTX's configured five-minute retention remains in place.
    stable_before = now - 15
    ordered = sorted(candidates, key=lambda p: p.stat().st_mtime if p.exists() else now)
    for item in ordered:
        if total <= MAX_RAW_BYTES:
            break
        if item.name not in owned or item.stat().st_mtime > stable_before:
            continue
        if (PIN_DIR / f"{item.name}.pin").exists():
            continue
        size = item.stat().st_size
        try:
            item.unlink()
            owned.discard(item.name)
            total -= size
            state.setdefault("rolling_gc_deleted", []).append(item.name)
        except OSError as exc:
            state["rolling_gc_error"] = type(exc).__name__
    _atomic_json(OWNED_MANIFEST, sorted(owned))
    state["rolling_bytes"] = total
    if total > MAX_RAW_BYTES:
        state["rolling_limit_blocked_by_pins_or_unowned"] = True


def _stop_process_group(process: subprocess.Popen | None, timeout: float = 5) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


def _launch_capture(state: dict) -> tuple[subprocess.Popen, subprocess.Popen, threading.Event, threading.Thread]:
    camera_log = (LOG_DIR / "camera.log").open("ab", buffering=0)
    publisher_log = (LOG_DIR / "publisher.log").open("ab", buffering=0)
    code = base64.b64encode(REMOTE_CAMERA.encode()).decode("ascii")
    remote_command = f"python3 -c 'import base64;exec(base64.b64decode(\"{code}\"))'"
    source = subprocess.Popen(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
         "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
         *(["-o", f"HostKeyAlias={CAMERA_HOST_KEY_ALIAS}"] if CAMERA_HOST_KEY_ALIAS else []),
         CAMERA_SSH, remote_command],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=camera_log,
        start_new_session=True,
    )
    stop_heartbeat = threading.Event()

    def renew() -> None:
        while not stop_heartbeat.wait(HEARTBEAT_SECONDS):
            try:
                source.stdin.write(b"renew\n")
                source.stdin.flush()
            except (BrokenPipeError, OSError, AttributeError):
                return

    heartbeat = threading.Thread(target=renew, name="camera-lease-heartbeat", daemon=True)
    heartbeat.start()
    publisher = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "warning",
         "-analyzeduration", "1000000", "-probesize", "65536",
         "-f", "mpegts", "-i", "pipe:0", "-map", "0:v:0", "-an", "-c:v", "copy",
         "-f", "rtsp", "-rtsp_transport", "tcp", "rtsp://127.0.0.1:18554/pi"],
        stdin=source.stdout, stdout=publisher_log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    source.stdout.close()
    camera_log.close()
    publisher_log.close()
    state["capture_children"] = [
        {"pid": source.pid, "role": "ssh_camera"},
        {"pid": publisher.pid, "role": "publisher"},
    ]
    return source, publisher, stop_heartbeat, heartbeat


def _stop_capture(capture: tuple | None) -> None:
    if not capture:
        return
    source, publisher, heartbeat_stop, heartbeat = capture
    heartbeat_stop.set()
    # Stop the consumer then close SSH; EOF/keepalive failure expires the remote lease.
    _stop_process_group(publisher)
    _stop_process_group(source)
    heartbeat.join(timeout=1)
    for handle in (source.stdin, source.stderr, publisher.stdout, publisher.stderr):
        try:
            if handle:
                handle.close()
        except OSError:
            pass


def _supervisor() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        stop_requested = threading.Event()

        def request_stop(_signum, _frame):
            stop_requested.set()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        instance_id = uuid.uuid4().hex
        now = time.time()
        state = {
            "instance_id": instance_id, "controller_pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
            "started_at_epoch": now, "status": "starting", "camera_host": CAMERA_SSH,
            "hls_url": HLS_URL, "transport": "H264 MPEG-TS over SSH lease -> localhost RTSP stream copy",
            "capture_profile": {"width": 1920, "height": 1080, "fps": 30, "bitrate": 8000000},
            "raw_retention_seconds": 300, "raw_disk_ceiling_bytes": MAX_RAW_BYTES,
            "reconnect_backoff_seconds": list(BACKOFF), "last_media_progress_at": None,
            "last_media_fingerprint": None, "capture_children": [],
        }
        _atomic_json(STATE, state)
        capture = None
        media = None
        backoff_index = 0
        last_fingerprint = None
        last_progress = time.monotonic()
        last_update = 0.0
        try:
            _ports_available()
            if not MEDIA.is_file() or not os.access(MEDIA, os.X_OK):
                raise RuntimeError("Pinned local MediaMTX binary is missing or not executable")
            media_log = (LOG_DIR / "mediamtx.log").open("ab", buffering=0)
            media = subprocess.Popen([str(MEDIA), str(MEDIA_CONFIG)], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=media_log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(100):
                if media.poll() is not None:
                    raise RuntimeError("MediaMTX exited; see mediamtx.log")
                try:
                    with socket.create_connection(("127.0.0.1", PORTS[0]), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("MediaMTX RTSP listener did not become ready")

            state.update(status="connecting", media_pid=media.pid)
            _atomic_json(STATE, state)
            while not stop_requested.wait(1):
                now_mono = time.monotonic()
                if media.poll() is not None:
                    raise RuntimeError("MediaMTX stopped unexpectedly")
                if capture is None:
                    try:
                        capture = _launch_capture(state)
                        last_fingerprint = None
                        last_progress = time.monotonic()
                        state.update(status="connecting", last_capture_start_epoch=time.time(), last_error=None)
                        _atomic_json(STATE, state)
                    except Exception as exc:
                        state["last_error"] = f"capture_start:{type(exc).__name__}"
                        state["status"] = "reconnecting"
                        _atomic_json(STATE, state)
                        stop_requested.wait(BACKOFF[backoff_index])
                        backoff_index = min(backoff_index + 1, len(BACKOFF) - 1)
                        continue

                source, publisher, _, _ = capture
                exited = source.poll() is not None or publisher.poll() is not None
                fingerprint = _media_fingerprint()
                if fingerprint and fingerprint != last_fingerprint:
                    last_fingerprint = fingerprint
                    last_progress = now_mono
                    state.update(status="running", last_media_progress_at=time.time(),
                                 last_media_fingerprint=fingerprint, last_error=None,
                                 last_progress_check="hls_playlist_part")
                    backoff_index = 0
                stalled = now_mono - last_progress > PROGRESS_TIMEOUT
                if exited or stalled:
                    state["last_error"] = ("capture_child_exited" if exited else
                                            f"media_progress_stalled_{PROGRESS_TIMEOUT}s")
                    state["status"] = "reconnecting"
                    state["last_interruption_at"] = time.time()
                    _atomic_json(STATE, state)
                    _stop_capture(capture)
                    capture = None
                    delay = BACKOFF[backoff_index]
                    state["next_retry_in_seconds"] = delay
                    _atomic_json(STATE, state)
                    if stop_requested.wait(delay):
                        break
                    backoff_index = min(backoff_index + 1, len(BACKOFF) - 1)
                _rolling_gc(state, time.time())
                if time.monotonic() - last_update >= 2:
                    _atomic_json(STATE, state)
                    last_update = time.monotonic()
        except Exception as exc:
            state["status"] = "failed"
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            _atomic_json(STATE, state)
        finally:
            _stop_capture(capture)
            _stop_process_group(media)
            state.update(status="stopped", stopped_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                         stop_reason="user_requested" if stop_requested.is_set() else state.get("last_error", "supervisor_exit"),
                         capture_children=[])
            _atomic_json(STATE, state)


def relay_start() -> dict:
    state = _read_state()
    pid = state.get("controller_pid")
    if state.get("status") in {"starting", "connecting", "running", "reconnecting"} and isinstance(pid, int) and _pid_is_ours(pid):
        return {"status": "already_running", "controller_pid": pid, "hls_url": state.get("hls_url")}
    _validate_camera_setting()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "relay-supervisor.log").open("ab", buffering=0) as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_supervisor"],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    for _ in range(60):
        time.sleep(0.25)
        state = _read_state()
        if state.get("controller_pid") == process.pid:
            if state.get("status") in {"failed", "stopped"}:
                raise RuntimeError(state.get("last_error", "Relay start failed"))
            return {"status": state.get("status"), "controller_pid": process.pid,
                    "instance_id": state.get("instance_id"), "hls_url": state.get("hls_url")}
        if process.poll() is not None:
            raise RuntimeError("Supervisor exited during startup; see relay-supervisor.log")
    raise RuntimeError("Supervisor did not publish state within 15 seconds")


def relay_stop() -> dict:
    state = _read_state()
    pid = state.get("controller_pid")
    if not isinstance(pid, int) or not _pid_is_ours(pid):
        return {"status": "not_running", "note": "No owned supervisor process was signaled"}
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        time.sleep(0.1)
        updated = _read_state()
        if updated.get("controller_pid") == pid and updated.get("status") == "stopped":
            return {"status": "stopped", "stop_reason": updated.get("stop_reason")}
        if not _pid_is_ours(pid):
            return {"status": "stopped"}
    return {"status": "stop_requested", "controller_pid": pid}


def _validate_camera_setting() -> None:
    user, separator, host = CAMERA_SSH.partition("@")
    valid_host = bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host))
    if ":" in host:
        address, scope_separator, scope = host.partition("%")
        try:
            socket.inet_pton(socket.AF_INET6, address)
            valid_host = not scope_separator or bool(re.fullmatch(r"[A-Za-z0-9_.-]+", scope))
        except OSError:
            valid_host = False
    if not separator or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", user) or not valid_host:
        raise RuntimeError("Set PI_CAMERA_SSH to user@hostname, user@IPv4, or user@IPv6%interface.")
    if CAMERA_HOST_KEY_ALIAS and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", CAMERA_HOST_KEY_ALIAS):
        raise RuntimeError("PI_CAMERA_HOST_KEY_ALIAS must name an existing known_hosts entry.")


def _compose_parameters() -> tuple[list[str], dict[str, str]]:
    command = ["docker", "compose", "--project-name", "skydetect-pi-local", "--env-file", ".env"]
    if PI_ENV.is_file():
        command += ["--env-file", str(PI_ENV)]
    command += ["-f", "docker-compose.yml", "-f", "docker-compose.pi.yml"]
    env = {**os.environ, "PI_RELAY_RUNTIME": str(ROOT),
           "PI_LOCAL_UID": _setting("PI_LOCAL_UID", str(os.getuid())),
           "PI_LOCAL_GID": _setting("PI_LOCAL_GID", str(os.getgid()))}
    return command, env


def _compose(*args: str) -> subprocess.CompletedProcess:
    command, env = _compose_parameters()
    return subprocess.run([*command, *args], cwd=DOCKER_DIR, env=env, check=True, text=True)


def _prepare_runtime_directories() -> None:
    for folder in ("recordings", "status", "clips", "local-bridge", "pins"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)


def stack_status() -> dict:
    command, env = _compose_parameters()
    result = subprocess.run([*command, "ps", "--format", "json"], cwd=DOCKER_DIR, env=env,
                            capture_output=True, text=True, timeout=10)
    services = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                item = json.loads(line)
                services.append({k: item.get(k) for k in ("Service", "State", "Health", "Ports")})
            except json.JSONDecodeError:
                continue
    return {"relay": _read_state(), "compose_services": services,
            "compose_error": None if result.returncode == 0 else result.stderr.strip()[:300]}


def main(argv: list[str]) -> int:
    if argv == ["_supervisor"]:
        _supervisor()
        return 0
    if argv == ["relay", "start"]:
        print(json.dumps(relay_start(), ensure_ascii=False, indent=2))
    elif argv == ["relay", "stop"]:
        print(json.dumps(relay_stop(), ensure_ascii=False, indent=2))
    elif argv == ["relay", "status"]:
        print(json.dumps(_read_state(), ensure_ascii=False, indent=2))
    elif argv == ["stack", "start"]:
        _validate_camera_setting()
        _prepare_runtime_directories()
        _compose("up", "-d", "--build", "frontend", "clip-bridge")
        print(json.dumps(relay_start(), ensure_ascii=False, indent=2))
    elif argv == ["stack", "stop"]:
        print(json.dumps(relay_stop(), ensure_ascii=False, indent=2))
        _compose("stop")
    elif argv == ["stack", "status"]:
        print(json.dumps(stack_status(), ensure_ascii=False, indent=2))
    else:
        print("Usage: control.py relay|stack start|stop|status", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
