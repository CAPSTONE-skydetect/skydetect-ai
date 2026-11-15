# Clip save regression repair — 2026-11-06

## Confirmed primary cause

Active fMP4 recordings flush in parts. The old 1.5-second mtime/250ms size check
could copy an unfinished segment during a pause and never refresh that copy.
Failed job clip_92bb316ffd824534805786acbb50322b had a staged segment lasting
1.032967 seconds; the finalized original of the same name lasted 10.996100 seconds.
The resulting spurious 9.963-second inter-segment gap rejected the clip.

## Repair

- Require a newer MediaMTX segment before copying a quiet predecessor.
- Exclude non-overlapping segments from strict clip validation.
- Select decoded frame ordinals rather than ffmpeg -ss/-t rounding; rebase PTS
  without synthesizing frames, retaining output/source count validation.
- Keep genuine PTS-gap rejection. No threshold sweep or gap-filling workaround.
- One additional `disk I/O error` was observed on the macOS bind-mounted SQLite
  WAL setup during validation. Disk had >300GiB free and integrity_check was ok.
  The exact low-level cause is not proven. Backed up the stopped database,
  migrated to DELETE journaling, and close every request's connection explicitly.
  Do not access live SQLite from both host and VM; inspect through APIs instead.

## Actual verification

- 12 tests passed: quiet/active file cannot copy early, successor permits complete
  copy, irrelevant parts excluded, real PTS gap still rejected, actual FFmpeg
  non-frame-aligned extraction, DB restart and closed handles, existing tests.
- Real clip after file fix: clip_2eba59e92a284764bd17d82c0c40d993 READY, 390 frames.
- After DB migration: clip_a2f24a6d11e944ffb91b54b3308945ba READY, 390 frames,
  duration 12962 ms, HTTP Range 206/1024 bytes.
- Final service with connection cleanup: clip_1fd3c36423124f3db570932f9b3cf9be
  READY, 391 frames, duration 12995 ms, HTTP Range 206/1024 bytes.
- Counts reflect actual selected frames, not padding to a fixed count.
- Failed user jobs and existing saved clips were not rewritten or removed.
- Local runtime backup: outputs/pi-live-20261106/local-bridge/
  backup-save-fix-20261106/jobs.sqlite3 (outside Git).

This verifies local save plumbing, not browser interaction, uninterrupted network
operation, sensor wall-clock accuracy or bird/drone classification performance.
