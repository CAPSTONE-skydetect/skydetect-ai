# Optional MiniRocket Stack training

This branch adds an opt-in research CLI. The collaborator's MiniRocket,
StandardScaler, Ridge, sequence preprocessing, group weights/metrics, and
`research/requirements-minirocket.txt` are inherited. The addition is the
image-width-scale member and logistic Stack head, plus a TrackSequence
collection loader. The existing service and baseline checkpoint are unchanged.

## Setup

After the branch is published and its commit confirmed:

```sh
git clone --branch model/minirocket https://github.com/CAPSTONE-skydetect/skydetect-ai.git
cd skydetect-ai
python3.14 -m venv .venv-stack
. .venv-stack/bin/activate
python -m pip install -r research/requirements-minirocket.txt pydantic==2.12.5
```

Use the project root as the working directory. No second checkout or
`PYTHONPATH` is needed. Verified with CPython 3.14.3 on macOS 26.4 arm64;
wheel availability and runtime on other OSes were not tested.

## Data and commands

The original `real_reference_comparison_v1` package must be transferred
separately and extracted. Its inherited validator checks hashes and lineage;
Stack uses only the real-only training arrays and common validation.

```sh
python -m research.minirocket_stack_training audit --real-package /path/to/real_reference_comparison_v1
# Run the next command only after data and training are approved.
python -m research.minirocket_stack_training train --real-package /path/to/real_reference_comparison_v1 --output research/output/stack-run-01
python -m research.minirocket_stack_training evaluate --real-package /path/to/real_reference_comparison_v1 --model research/output/stack-run-01/stack.joblib
python -m research.minirocket_stack_training predict --model research/output/stack-run-01/stack.joblib --track /path/to/track_sequence.json
```

Alternatively replace `--real-package …` with `--collection /path/to/manifest.json`.
The schema is `stack-tracksequence-collection-1` with a `records` array.
Each row needs `source_sha256`, `source_group`, `session_group`, `parent_id`,
`object_id`, `segment_id`, `label` (`bird`/`drone`), `split`
(`train`/`validation`/`test`), `qa_disposition`, `track_path` (relative to
manifest, or absolute), and `track_sha256`. See the synthetic manifest generator:

```sh
python -m examples.stack_synthetic research/output/manifest-example
python -m research.minirocket_stack_training audit --collection research/output/manifest-example/manifest.json
```

Only `accepted` QA rows produce windows. Hashes must bind actual bytes.
The builder computes each window's normalization scale from the unchanged
sequence contract. Optional `normalization_scale` is a list in accepted-window
order and is verified rather than used as an override. Rejected clocks,
stabilization, short intervals and gaps are reported; no time repair or fake
arrays are used. Source/video/group/session/parent lineage cannot cross splits.
Use conservative source groups; unknown session does not imply independence.
The inherited group metric rejects mixed bird/drone groups: **do not split one
capture by label** to make it pass. Such a collection needs a separately
reviewed group evaluation contract. Test input is reserved unless explicitly
evaluated with `evaluate --split test` on a collection.

The output path must **not exist**. It receives a checkpoint, status and
alpha/OOF/validation tables, not a replacement for the shipped baseline.
Record accepted/rejected audit rows and group assignments before running.
Only load trusted joblib files; deserialization can execute code.

## Recipe and limits

- Original q/d channels and the scale-restored member both use 4×60 inputs.
  The second member multiplies centered q by per-window scale ×400 and
  recomputes displacement. This is apparent image motion, not physical speed.
- Each member alpha is selected with inherited train-only, group-stratified
  three-fold CV: 0.1, 1, 10, 100, smallest on ties. Seed 20260928. Fold feature
  transforms and member fits never see held fold data. Full real-package
  member fitting preserves supplied weights; collection uses inherited
  class/group-balanced weights. No augmentation.
- Fixed member alphas generate OOF margins; track means fit StandardScaler and
  LogisticRegression C=1 using class/group weights. OOF is meta-training, not
  independent evaluation. Validation is evaluation-only. Inference averages
  window margins with the existing sign at zero; scores are not probabilities.
- `examples.stack_synthetic --smoke` is synthetic-only and uses 168 kernels
  in that subprocess. The real training CLI keeps 10,000. Synthetic smoke
  metrics are not performance evidence.
