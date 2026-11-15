# EXP-STACK-PORTABLE-19

**Implementation and functional verification complete; not published.**
Branch: `codex/minirocket-stack-training`; worktree:
`../skydetect-ai-stack-training`. Base is remote `model/minirocket` at
`d31a8fe9606e6adc71e10f798aff837b4a708f36`; protocol commit `0679684`.
The scoped result commit is the commit containing this document.

- Added `research/minirocket_stack_training.py` (opt-in
  audit/train/evaluate/predict) and `research/minirocket_stack_data.py`
  (immutable TrackSequence input and lineage checks).
- Added synthetic example, two-input parity helper, focused tests and the
  collaborator instructions in `docs/minirocket_stack_training.md`.
  The only ignore rule added is for local evidence under
  `artifacts/stack-portable-19/`. No production, A, UI, dependency-pin or
  checkpoint changes. No external project source imports.
- Original package audit: 156 real-only train windows and 59 validation
  windows. Read-only; no fit. One synthetic 168-kernel smoke: 16 train and
  4 validation windows, fit/save/load/predict successful. No smoke retry.
- Existing frozen checkpoint parity: bird `176/o1/f30–166` and drone
  `126/start1/1`. Both models exactly match the saved label, final and
  per-window margin, window starts and rejections (max delta 0; tolerance
  `atol=rtol=1e-8`). The two inputs test compatibility, not accuracy.
- Focused tests: 43 passed, 1 skipped. The inherited package-loader test's
  default folder was absent; the explicit original-package audit passed.
  First test attempt in the A environment lacked aeon and a fixture hit the
  duplicate guard before the split guard. The final run used the pinned C
  runtime first, with app/test dependencies read-only at lower precedence.
- macOS 26.4 arm64, Python 3.14.3; existing pinned aeon 1.6.0, numpy 2.3.5,
  pandas 3.0.2, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8,
  joblib 1.6.0, numba 0.63.1; pydantic 2.12.5. Smoke/parity used clean
  subprocesses with no PYTHONPATH, project module origin assertions and no
  writes to shared environments. Other OSes and fresh package installation
  were not validated.

The five source hashes in the protocol were rechecked unchanged. Existing
baseline SHA256
`9024731a649b5e0a84ead3fcc45c9dc0a692ce305cea6e5ec8b7d720867a1e5a`;
frozen Stack SHA256
`5ba8fc4ac843f30c8a2a590add3f5854f6cf23093d1b8ed1f3b0731b95ec8aab`.
Neither was modified. Evidence is local in `artifacts/stack-portable-19/`:
`original-audit.log`, `synthetic-smoke.log`, `parity/lock.json`,
`parity/results.json`, `tests.log`.

**Real-data fits: 0. Pushes: 0.** Mixed-label capture groups are still
rejected by the inherited group metric; splitting the group by label is not
an acceptable workaround. Before publication Astra should review the diff
and confirm the remote SHA and fast-forward ancestry. No claim of improved
performance or authority to start a real training run.
