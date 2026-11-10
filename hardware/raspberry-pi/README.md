# Raspberry Pi local integration (experimental)

This module belongs to `codex/pi-live-integration`. It does not change D0,
RF or MiniRocket. Spring owns new clip jobs, authorization, and MySQL history.
It calls the private Python worker to extract clips with FFmpeg. The frontend
can still read existing SQLite clips from the bridge; their records are not
migrated to MySQL. New clip analyses use the existing browser-to-AI flow, not
Spring/MySQL analysis history.

## Compatible revisions

Use the sibling repository and model worktree layout in
[`docker/README.md`](../../docker/README.md). These revisions reproduce the
Pi integration candidate, not a hardware-validated final release:

| Repository or worktree | Branch | Commit |
| --- | --- | --- |
| `skydetect-ai` | `codex/pi-live-integration` | `dee654daaa24858dc499a1bf687741d16ae75597` (runtime baseline) |
| `skydetect-backend` | `codex/pi-live-integration` | `f6c64943f353eb75d292405abd0740d028b257c9` |
| `skydetect-frontend` | `codex/pi-live-integration` | `24927194590c6134a6e05e87dc5ce8caee7e9549` |
| `skydetect-ai-rf` | `model/rf` | `16246fb028e836c451c95fc80fe68ef978d42505` |
| `skydetect-ai-minirocket` | `model/minirocket` | `d31a8fe9606e6adc71e10f798aff837b4a708f36` |

For a fresh checkout, run these commands from an empty parent directory with
access to the three GitHub repositories. A documentation-only commit on the AI
branch can be used in place of the runtime baseline if the runtime files match.

```sh
git clone --branch codex/pi-live-integration https://github.com/CAPSTONE-skydetect/skydetect-ai.git
git clone --branch codex/pi-live-integration https://github.com/CAPSTONE-skydetect/skydetect-backend.git
git clone --branch codex/pi-live-integration https://github.com/CAPSTONE-skydetect/skydetect-frontend.git
git -C skydetect-ai switch --detach dee654daaa24858dc499a1bf687741d16ae75597
git -C skydetect-backend switch --detach f6c64943f353eb75d292405abd0740d028b257c9
git -C skydetect-frontend switch --detach 24927194590c6134a6e05e87dc5ce8caee7e9549
git -C skydetect-ai worktree add --detach ../skydetect-ai-rf 16246fb028e836c451c95fc80fe68ef978d42505
git -C skydetect-ai worktree add --detach ../skydetect-ai-minirocket d31a8fe9606e6adc71e10f798aff837b4a708f36
```

Before starting, create `skydetect-ai/docker/.env` from `.env.example` with
local DB and operator credentials. Create `docker/.env.pi` in the AI repository
from [`.env.pi.example`](../../docker/.env.pi.example), set `PI_CAMERA_SSH` and
verify that SSH to the camera works with a trusted host key. Set
`PI_RELAY_RUNTIME` to reuse an existing recordings/clip directory if needed.
Relative runtime paths are resolved from `skydetect-ai/docker`.
The controller supplies the invoking UID/GID unless overridden; direct Compose
invocations must set both. The example binds the stack ports to loopback.

## Explicit operation

From the sibling `skydetect-live-integration` installation:

```sh
python3 skydetect-ai/hardware/raspberry-pi/control.py stack start
python3 skydetect-ai/hardware/raspberry-pi/control.py stack status
python3 skydetect-ai/hardware/raspberry-pi/control.py stack stop
```

Relay-only controls use `relay start|status|stop`. No boot/login services are
installed. The Mac must remain awake, with Docker and the Pi network available.
The default installation layout is documented in `control.py`; `PI_RELAY_RUNTIME`
can override the data folder. Native MediaMTX and FFmpeg must already exist on
the host, and `rpicam-vid` must work on the Pi. Binary installation is not
automated. With the example config, open `http://127.0.0.1:5178/` and log in
using the operator account configured in `docker/.env`.

Supervisor reconnects on camera child exit or missing media progress, with capped
backoff. Remote heartbeat expires after 15 seconds; a separate six-hour capture
ceiling bounds orphan lifetime if the guardian dies. This can cause a reconnect
at six hours: uninterrupted 24/7 capture has NOT been demonstrated.

Raw recordings retain five minutes in MediaMTX. Additional GC has a 2GiB target
and only deletes eligible supervisor-owned completed files. Pinned/unowned files
can prevent reaching the target. Completed clips have a separate 4GiB reservation
check and are never auto-deleted. This is not a whole-disk/log/staging quota.

## API and timing

New clip creation, listing, status, and playback use Spring's `/api/clips`.
Writes require an authenticated Spring session and CSRF token. Interrupted
pending jobs are marked failed on Spring restart. The legacy proxy
`/local-api/` still exposes the bridge's authenticated SQLite records. Legacy
POST also checks Origin; the internal worker is not published on a host port.

Clips target server receipt time minus 3 seconds / plus 10 seconds. Client click
and displayed HLS media time are recorded separately; sensor capture time and
display-wall-clock alignment are NOT established. Gaps are rejected, not filled.
The extracted clip retains a source-frame sidecar. The model runtime still uses
its existing FPS-based tracking timestamp behavior; this work does not validate
precision motion features on variable-rate camera recordings.

## Verification

```sh
python3 skydetect-ai/hardware/raspberry-pi/test_local.py
python3 skydetect-ai/hardware/raspberry-pi/verify_spring_clips.py
python3 skydetect-ai/hardware/raspberry-pi/verify_spring_clips.py --browser
```

The disposable Compose verifier requires the local stack images to be built
and FFmpeg on PATH. It uses synthetic recordings and a separate MySQL volume,
checks creation, failure when offline, range playback, and persistence after
Spring restart, and writes evidence under ignored `reports/pi-integration/`.
`--browser` also provides a temporary browser session to check clip playback
and ROI analysis manually. Neither mode validates the physical Pi connection.

SQLite uses DELETE journaling with explicitly closed per-request connections;
do not inspect a running bind-mounted database from a second host SQLite process.
Use authenticated APIs for live inspection. A stopped-service backup is preferred
before maintenance. See `STORAGE_FIX_20261106.md` for the save-race regression.

Camera clip/ROI/classification/restart/soak checks on the new Spring path remain
required. Keep real recordings, model weights, local credentials, databases
and logs outside commits.

### Final submission check

1. Confirm the live feed from the physical Pi and capture a clip in the UI.
   Check that the Spring job reaches `READY`, the clip plays and seeks, and its
   history survives a backend restart.
2. In the clip, select an airborne object's ROI and analyze a segment with each
   model. Confirm the trajectory and overlay downloads and the displayed
   prediction or abstention. This checks integration, not classification accuracy.
3. Record the date, hardware, test outcomes, and exact Git SHAs for the AI,
   backend, frontend, RF, and MiniRocket checkouts that passed. Share the AI
   README at the submission revision that documents those verified versions.

### Legacy bridge-only integration checks (2026-11-06)

One real Pi clip reached READY (390 mapped frames, about 13 seconds), byte-range
seek returned 206. Real stored indoor footage was then analyzed once per model:
both jobs DONE with 91 points at frames 30..120; TrackSequence JSON and overlay
downloads succeeded. RF abstained (`high_noise`), MiniRocket returned `bird` on
ceiling texture. This is a known input-domain limitation, NOT a correct result:
neither service proves that the selected ROI contains a flying bird/drone.
The UI warns users; weights and thresholds are unchanged.

Bridge restart preserved the READY clip and both DONE job result hashes. New
authenticated API sessions can list and retrieve those records. Actual browser
ROI drawing, refresh interaction, long soak and true airborne accuracy remain
unverified. Completed clips/results survive restart; this is local SQLite, not
Spring/MySQL analysis integration. Root validation scripts store evidence outside
Git. Keep original videos, models, runtime database, .env and logs out of commits.
