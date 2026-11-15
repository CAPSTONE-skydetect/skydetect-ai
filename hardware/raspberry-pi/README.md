# Raspberry Pi local integration (experimental)

## Wired preview check (2026-11-11)

The Pi's DHCP-based Ethernet profile repeatedly failed activation on the direct
Mac cable. The backed-up link-local configuration passed 25 network probes over
four minutes. Network verification is not a long-duration reliability guarantee.
`direct_link_setup.sh` requires the operator to enter sudo credentials on the Pi;
it preserves a root-owned profile backup and uses an independent rollback timer.
This maintenance helper is specific to the inspected Pi profile UUID, Mac peer,
and interface names embedded in the scripts. It refuses a non-auto/auto baseline;
do not rerun it as a routine start command or use it unchanged on another setup.

Preview changes: reuse the supervisor's media-playlist session, rebuild the HLS
player from the master URL after fatal network errors, and provide stable/fast
preview modes plus a live-edge return button. Stable mode remains the HLS default.
The UI delay is hls.js's estimated distance to the live edge, NOT sensor latency.

Manual clock-in-camera screenshots compared the current on-screen clock with
the same clock inside the Pi feed. Both clocks originate on the Mac, avoiding
the Pi clock offset. Values below are short development observations, not a
latency SLA; the clock updates every 50 ms and camera/display scanning add error.

| Mode | Current clock / filmed clock | Approximate end-to-end delay |
| --- | --- | --- |
| Stable | 16:33:54.432 / 16:33:52.933 | 1.499 s |
| Stable | 16:34:21.232 / 16:34:19.732 | 1.500 s |
| Stable | 16:34:43.781 / 16:34:42.282 | 1.499 s |
| Fast | 16:35:19.180 / 16:35:17.981 | 1.199 s |
| Fast | 16:35:54.929 / 16:35:53.680 | 1.249 s |
| Fast | 16:36:53.027 / 16:36:51.778 | 1.249 s |

The fast-mode UI showed about 0.95 s estimated live-edge delay and zero buffer
stalls during the observed interval. Automatic recovery after one intentional
relay restart was observed without pressing reconnect. Reader count fell from
dozens to two (browser plus supervisor) after idle sessions expired. This does
not establish that session churn caused the entire prior delay.

1080p30 capture, recording, and analysis settings were unchanged. A separate
FFmpeg HLS check still reports duplicated MOOV and non-monotonic DTS warnings,
including when using a fixed media-playlist session. These are NOT fixed by the
session-cache change. MediaMTX also logged variable part-duration warnings;
Safari/iOS compatibility and long-duration fast-mode stability remain unverified.

### WebRTC preview (2026-11-11)

The local Compose overlay now supplies optional `VITE_WHEP_URL`, selecting
WebRTC preview by default only in that configured build. Without it the frontend
keeps HLS. The UI offers an explicit HLS/WebRTC switch. MediaMTX exposes WHEP
on `127.0.0.1:18889` and ICE UDP/TCP on `127.0.0.1:18189`; no external ICE
servers or interface candidates are configured. This is a same-Mac setup, not
a remotely reachable deployment configuration.

The Pi H.264 stream still feeds the same stream-copy recording path. No preview
transcoding, capture-resolution reduction, or recording-bitrate reduction was
introduced. Completed recording samples before/after the change were H.264 High,
1920x1080, approximately 30 fps, no B frames, and 7.989/7.996 Mbps respectively.

Observed checks: WebRTC decoded 1920x1080 at 30 fps; the selected ICE connection
used loopback UDP; switching to HLS removed the WebRTC server session, and
switching back restored live playback. The sampled server session reported zero
outbound discarded frames. Connection statistics alone do not establish
end-to-end latency or long-duration reliability.

After the user aimed the camera at the clock, three same-screenshot comparisons
on 2026-11-11 measured approximately 0.30-0.35 seconds glass-to-glass:

| Current clock | Filmed clock | Displayed-time difference |
| --- | --- | --- |
| 18:03:11.528 | 18:03:11.178 | 0.350 s |
| 18:03:18.277 | 18:03:17.978 | 0.299 s |
| 18:03:21.977 | 18:03:21.677 | 0.300 s |

These manually read short samples support sub-second preview in this wired
same-Mac setup, not a latency SLA. The clock refreshes every 50 ms; screenshot,
display and camera scanning introduce uncertainty, so millisecond precision is
not claimed. Earlier fast-HLS samples were approximately 1.20-1.25 seconds;
the observations were sequential, not simultaneous controlled trials.

Frontend HLS/WebRTC unit tests: 10 passed. Frontend container build passed.
Preview media time is not a calibrated sensor wall-clock timestamp.

### Interrupted stability check (2026-11-11)

A 30-minute read-only check was stopped at the user's request to charge the
laptop, after about 16.8 minutes and 201 polls (including deliberate outages and
final shutdown). It is NOT a completed uninterrupted 30-minute soak.

- A single owned camera SSH process was terminated. The supervisor relaunched
  capture and the browser returned to WebRTC 1080p30 without a refresh or retry
  click. Estimated recording gap: 4.16 seconds.
- The user physically disconnected/reconnected the cable. Pi logs show link
  return at 18:23:13; Mac recorder resumed at 18:23:26 and WebRTC connected at
  18:23:28. The approximate 15-second recovery is not a precise cross-device
  measurement. The recording gap including the disconnection was about 262.62s.
- 68 finalized recording containers passed ffprobe. This is container readability,
  not full-frame decoding or visual quality validation. Other than the two induced
  outages, no >0.25s gaps were found using filename wall-clock versus duration;
  frame-PTS continuity is not established by that check.
- The user reported successful clip capture and analysis during WebRTC playback;
  this is a workflow check, not evidence of bird/drone classification accuracy.

Raw samples and the resume note remain local under
`reports/pi-stability-20261111/` in the parent workspace, outside this repository.
Next: a fresh uninterrupted 30-minute soak and a post-reconnection clip check.
Relay/recording stopped normally; Pi power-off required the user's interactive
sudo command and was not independently confirmed. No model settings changed.

This module belongs to `codex/pi-live-integration`. It does not change D0,
RF or MiniRocket. Spring owns new clip jobs, authorization, and MySQL history.
It calls the private Python worker to extract clips with FFmpeg. The frontend
can still read existing SQLite clips from the bridge; their records are not
migrated to MySQL. New clip analyses use the existing browser-to-AI flow, not
Spring/MySQL analysis history.

## Unified source checkout

On `codex/ai-model-unification`, all hardware files from the published Pi branch
are retained. Both AI images now build from this checkout with separate model
targets. The Pi overlay no longer replaces A source files via bind mounts.
The clip worker has its own Python+FFmpeg Dockerfile; it does not use the RF image.
See [model unification](../../docs/model_unification.md) and
[Compose setup](../../docker/README.md). Existing running stacks are not migrated
automatically. Validate on a separate stack before replacing a working installation.

## Historical compatible revisions (before unification)

The following old revisions used sibling model worktrees. They reproduce the
Pi integration candidate, not a hardware-validated final release:

| Repository or worktree | Branch | Commit |
| --- | --- | --- |
| `skydetect-ai` | `codex/pi-live-integration` | `dee654daaa24858dc499a1bf687741d16ae75597` (runtime baseline) |
| `skydetect-backend` | `codex/pi-live-integration` | `f6c64943f353eb75d292405abd0740d028b257c9` |
| `skydetect-frontend` | `codex/pi-live-integration` | `24927194590c6134a6e05e87dc5ce8caee7e9549` |
| `skydetect-ai-rf` | `model/rf` | `16246fb028e836c451c95fc80fe68ef978d42505` |
| `skydetect-ai-minirocket` | `model/minirocket` | `d31a8fe9606e6adc71e10f798aff837b4a708f36` |

For a historical checkout only, run these commands from an empty parent directory with
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
