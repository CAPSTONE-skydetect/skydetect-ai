# EXP-CONTINUITY-01 Protocol

## Identity and starting state

- Repository: `/Users/yuchan/Documents/형주플젝/skydetect-ai-trajectory-experiments`
- Requested baseline HEAD: `a240387b269f80460785bc305aea5eef6ad6d00c`
- Start working tree was dirty. Existing tracked modifications: `.gitignore`,
  `ai_server/services/foreground_motion.py`,
  `ai_server/services/manual_roi_tracker.py`,
  `ai_server/services/online_appearance.py`, `ai_server/tracking_schemas.py`,
  `research/tests/test_tracking_evaluation.py`, `research/tracking_evaluation.py`,
  and `tests/test_manual_roi_tracking.py`.
- Existing untracked work included prior ARD100/MiniRocket scripts and tests,
  complementary-center/motion-lock/patch-registration work, trajectory experiment
  modules, and reports under `docs/`. These are pre-existing and must not be
  staged, edited, or included in EXP-CONTINUITY-01 commits.
- First commit scope: this protocol document only. Final commit scope: only files
  created for EXP-CONTINUITY-01 under `research/`, `docs/`, and its artifact folder.

## Parent evidence and frozen inputs

Hashes are SHA-256 at protocol creation.

| Item | Path | SHA-256 |
|---|---|---|
| Parent downstream summary | `artifacts/trajectory_experiments/ard100-yolo-hybrid-downstream-20261018-v4/summary.json` | `4a8226f0004a8ce609269d0af0ba9d503fbef2a85213f8abb54e1f2e246b6a7b` |
| Parent predictions | `artifacts/trajectory_experiments/ard100-yolo-hybrid-downstream-20261018-v4/classification.csv` | `96eeafa592ab385da4de194b05477a38ebf702b6d57b37e3c49209433c1772cb` |
| Parent per-window details | `artifacts/trajectory_experiments/ard100-yolo-hybrid-downstream-20261018-v4/classification_details.json` | `b0820c9439d174dc30c8b3f951bd123f0b612cd56a4f9d5bc73d37d2d9f7b494` |
| Parent splice audit | `artifacts/trajectory_experiments/ard100-yolo-hybrid-downstream-20261018-v4/splice_audit.csv` | `da3c6aff908e2cbf925abc025f7c312e5c951fda1058bcc28c9523f17e625b3a` |
| Parent runner | `research/ard100_yolo_hybrid_downstream.py` | `06802bbce9c19ee583c46208e3d9a2f67af349d5e54765099fb9100591efc0e8` |
| Parent hybrid gate protocol | `docs/ard100_yolo_hybrid_gate_20261018.md` | `e874648659c37b09b46b20b925a1c3755663f8e1615d9b4205af82b4a754e818` |
| Parent MiniRocket training report | `docs/minirocket_training_results_20261018.md` | `f4436831e371f760684e0d39a92c7b7bd296303fc98079e6ec4bb2792a4e9780` |
| Parent bird screening report | `docs/bird_false_positive_check_20261018.md` | `bc7264e6960d69d046cb974006a2bfd19287e703577ff0aba492942b19ce79ea` |
| Frozen deployed baseline model | `../skydetect-ai-minirocket-runtime/models/minirocket_classifier.joblib` | `9024731a649b5e0a84ead3fcc45c9dc0a692ce305cea6e5ec8b7d720867a1e5a` |
| Frozen Stack candidate | `artifacts/trajectory_experiments/minirocket-stack-20261018/stack.joblib` | `5ba8fc4ac843f30c8a2a590add3f5854f6cf23093d1b8ed1f3b0731b95ec8aab` |
| Reproduced training baseline, not downstream checkpoint | `artifacts/trajectory_experiments/minirocket-retrain-20261018/baseline.joblib` | `fd08d555575f744f6ab460e9971f87b5ca0d0be0e6085140e0d356e0c2f7bce8` |
| Real reference training package | `/Users/yuchan/Downloads/real_reference_comparison_v1.zip` | `445b4d0cac9339a24775841f11efe0aa230719a945e544868ecd9d3adbc299e4` |
| Existing C window/preprocessing contract | `research/trajectory_sequence.py` | `0fa02ce7629b8c8fc556c5581123b425582aae41eb9c7cc8fcd68bd4135a678a` |
| Existing Stack adapter | `research/minirocket_stack.py` | `1177a146234c8c5a685fdfabb2c0d61d0da412b14a0e5e0bf01e8695aff0c8ba` |

The 12-object cohort, source videos, GT XML, D0 and Hybrid TrackSequence files,
recovery provenance, and model metadata are selected by the parent protocol and
summary. Per-file hashes for the exact selected inputs will be written to the
experiment artifact manifest before any result is computed. Do not infer that
HEAD alone reproduces this evidence: parent sources and artifacts were already
untracked or modified at start.

## Questions

1. Does excluding any two-second classifier window whose observed support crosses
   a provenance-confirmed D0/YOLO source transition reduce downstream harm, and
   what is the change in decision coverage?
2. Do local/package sources contain enough paired, source-disjoint GT and A
   trajectories for a later GT-only/A-only/GT+A training comparison?

## Experiment 1: continuity segmentation

- Reuse the 12 fixed ARD100 development objects, the saved D0 and Hybrid tracks,
  the accepted recovery audit, the deployed baseline checkpoint, the frozen Stack
  checkpoint, and the existing two-second / one-second-stride C contract.
- Do not rerun A tracking, tune thresholds on the 12 development objects, use
  labels/GT coordinates to define boundaries, or special-case any frame/object.
- A provenance source transition is established from the D0 TrackSequence frame
  set and accepted YOLO-recovery frame set. A Hybrid classifier window is a
  boundary-crossing window if its 60-sample resampling support would use points
  from both origins. Exclude the whole window. Do not interpolate across the
  resulting segment boundary; do not fill TrackSequence gaps or alter timestamps.
- Compare the frozen D0 result, existing full Hybrid result, and segmented Hybrid.
  Also report D0/full-Hybrid/segmented-Hybrid common-window comparisons where
  applicable. Preserve every candidate start, accept/reject reason, each window
  margin and starts, accepted/rejected counts, per-object outcomes, full-cohort
  coverage and correct/wrong/abstain counts, plus decided-only performance.
- Explain whether phantom66 changes due to fewer windows/abstention or whether
  remaining window scores become more consistent. Margins are not probabilities.
- Regression-check stored GT/A six-object pilot inputs and locally stored bird
  screening inputs where the existing runtime contract accepts them. Keep 14 Hz
  inputs invalid; do not fabricate timestamps or resample them into validity.
- This is a development-cohort ablation only and cannot promote a production
  model or tracker.

## Experiment 2: paired-source readiness audit

- Read the received package inventory and available local source, XML GT, and A
  TrackSequence material. Produce a source-group paired manifest with explicit
  provenance, class, source access, GT status, train/validation/development reuse,
  frame/time/coordinate correspondence, and hash evidence.
- Treat package arrays as model-training trajectories with their documented
  provenance; do not call them manual GT without direct evidence.
- Count source-disjoint train/validation/test options by class and paired GT/A
  availability. Do not synthesize missing pairs or silently move validation or
  development data into training.
- Deliver a concrete next-request proposal; do not train a new C model, MultiRocket,
  or image model in this experiment.

## Evaluation, budget, and outputs

- Primary outputs: `artifacts/trajectory_experiments/exp-continuity-01/` and
  `docs/exp_continuity_01_results.md`.
- Success means a reproducible accounting of how the boundary rule changes
  decisions and coverage, plus an auditable paired-source readiness decision.
  No improvement is a valid outcome. An error becoming an abstention is coverage
  tradeoff, not an accuracy gain.
- Reuse saved artifacts; no full A rerun, paid API, or large dataset download.
  Stop at 90 minutes and report any incomplete item with its blocker.
- Add isolated code and focused tests, record command/code/input/model hashes,
  test outputs, limitations, and before/after change lists. Selectively commit
  only this experiment's protocol, implementation, tests, and concise result
  document; never commit model/video binaries or pre-existing files.
- On completion or blocker, return one message to design session
  `01a1346e-1513-7480-a36f-d5a6b9bc8409` via `codex queue`, including experiment
  ID, status, core metrics, result path, commit and review request. Do not start
  another experiment after that handoff.
