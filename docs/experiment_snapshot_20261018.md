# Trajectory Experiment Snapshot (2026-10-18)

This branch snapshot publishes the A-tracker experiments, CVAT/GT evaluator,
ARD100 probes, frozen MiniRocket comparison tools, and their focused tests. It
does not change the production tracker defaults or replace any deployed model.

## What Is Included

- A tracking refinements and opt-in controls in `ai_server/services/` and
  `ai_server/tracking_schemas.py`.
- CVAT-to-TrackSequence evaluation, temporal/size-normalized metrics, B-feature
  comparison, and comparison-video tooling in `research/tracking_evaluation.py`.
- ARD100, bird-screening, MiniRocket, and trajectory-window research runners,
  with tests and experiment protocols/results in `research/` and `docs/`.
- `research/requirements-common-c-training.txt` for the optional common-C data
  preparation/training utilities.

## Local Inputs and Excluded Artifacts

Generated data and model outputs under `artifacts/trajectory_experiments/` are
intentionally ignored by Git. This snapshot therefore does **not** include raw
videos, CVAT XML exports, generated TrackSequences, overlays, evaluation CSV/JSON
reports, MiniRocket checkpoints, or training arrays. Those files are needed to
reproduce data-dependent results; their manifests and hashes are recorded in the
corresponding experiment documents when available.

The current author's local runs referenced these external locations:

- ARD100 videos: `/Users/yuchan/Desktop/ARD100/`.
- User-collected CVAT/video material: `/Users/yuchan/Desktop/라벨링/`.
- Bird screening clips: `/Users/yuchan/Desktop/데이터셋/`.
- MiniRocket runtime and deployed checkpoint:
  `/Users/yuchan/Documents/형주플젝/skydetect-ai-minirocket-runtime/`.
- The received training package:
  `/Users/yuchan/Downloads/real_reference_comparison_v1.zip`.
- Some result documents refer to the Stack checkpoint inside the ignored
  `artifacts/trajectory_experiments/minirocket-stack-20261018/` directory.

These are machine-specific paths, not credentials. Several runner scripts accept
explicit input/output/model paths; consult each protocol before rerunning. The
small per-run manifests and outputs under the ignored artifact directory were
not published, so a fresh clone must first obtain the corresponding data and
recreate a manifest from the documented protocol. No external videos or model
weights were force-added.

## Verification Scope

The publish snapshot is validated with focused tests for the changed tracker,
tracking evaluator, trajectory refinement/windowing, ARD100/MiniRocket runners,
and EXP-CONTINUITY-01. Data/model-dependent integrations require the local inputs
and checkpoints above and are not implied to run from a clean clone without
provisioning them.

On the author's current machine, the focused tracker suite passed **31 tests**;
the research suite passed **281 tests with 1 skipped**, with the unrelated
notebook-kernel test deselected because this sandbox cannot bind its local
kernel port. All 71 changed/new Python files compiled. The notebook test itself
was attempted and failed at kernel startup with `PermissionError` while binding
`127.0.0.1`; this was an environment restriction, not an assertion failure.
