# Raspberry Pi local integration (experimental)

This module belongs to `codex/pi-live-integration`. It does not change D0,
RF or MiniRocket. The Python bridge is a LOCAL integration service using Spring
session authentication; these endpoints are not implemented in Spring itself.

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
can override the data folder. Native MediaMTX and Homebrew FFmpeg must already
exist. Binary downloading/installing is not automated.

Supervisor reconnects on camera child exit or missing media progress, with capped
backoff. Remote heartbeat expires after 15 seconds; a separate six-hour capture
ceiling bounds orphan lifetime if the guardian dies. This can cause a reconnect
at six hours: uninterrupted 24/7 capture has NOT been demonstrated.

Raw recordings retain five minutes in MediaMTX. Additional GC has a 2GiB target
and only deletes eligible supervisor-owned completed files. Pinned/unowned files
can prevent reaching the target. Completed clips have a separate 4GiB reservation
check and are never auto-deleted. This is not a whole-disk/log/staging quota.

## API and timing

Frontend proxy `/local-api/` exposes authenticated clip and analysis jobs.
POST requires matching Origin and XSRF token plus a valid Spring login session.
Jobs persist in SQLite, interrupted jobs become FAILED on bridge restart.
No bridge port is published to the host.

Clips target server receipt time minus 3 seconds / plus 10 seconds. Client click
and displayed HLS media time are recorded separately; sensor capture time and
display-wall-clock alignment are NOT established. Gaps are rejected, not filled.
The extracted clip retains a source-frame sidecar. The model runtime still uses
its existing FPS-based tracking timestamp behavior; this work does not validate
precision motion features on variable-rate camera recordings.

## Verification

```sh
python3 skydetect-ai/hardware/raspberry-pi/test_local.py
```

Nine tests cover HLS session query handling, progress identity, remote source
syntax, segment names, CSRF, interrupted-job persistence and an actual FFmpeg
non-frame-aligned cut of synthetic 1080p video, invalid ROI and interval rejection.
These unit tests are not real-Pi validation.

Actual camera clip/ROI/classification/restart/soak checks remain required before
calling this complete. See installation-root `PI_FLOW_01_RESULTS.md`.

### Observed integration checks (2026-11-06)

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
