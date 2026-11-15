# Actual A to B preprocessing diagnostic

Run from the repository root:

```powershell
.\venv\Scripts\python.exe -X utf8 -B -m research.diagnose_real_preprocessing
```

The default output is `research/output/a_preprocessing_audit/index.html` and
`report.md`. Input defaults to the `bird` and `drone` folders under
`research/data/tracks`. `--input`, `--artifacts`, `--output`, `--model` and
`--skip-model` can override these locations or disable existing-model replay.

This is a local exploratory diagnostic. It does not train a model, tune a
simulator, choose a test split, or certify tracking correctness. The labels are
taken from folder names. Behavior labels remain unreviewed.

## Exact preprocessing

`extract_features(..., diagnostics=segments)` optionally records intermediate
arrays from the same calculation used by runtime B. Feature values, defaults,
version and contract are unchanged. Only successful extractions publish traces.
The report uses canonical width 1920 with the original aspect ratio preserved.
It separates long gaps, uses the original timestamps, and records the exact
Savitzky-Golay derivative used for speed, rather than estimating speed from a
new plotting implementation.

For each input the plots show:

1. Full-frame raw/exported/smoothed paths.
2. The same paths at a closer scale, with equal pixel geometry.
3. Position-difference speed versus the actual B derivative speed.
4. The residual removed by smoothing, containing both motion and tracking error.
5. Before/after position PSD on one sufficiently long gap-free run.
6. B's heading support gate and missing-frame intervals.

## Matching and grouping

An A run matches only when its source ID, dimensions and full exported history
match. CSV timestamps and clipped normalized coordinates are also checked.
Missing or ambiguous raw data is marked as unavailable, not reconstructed.

Candidate groups conservatively union equal source IDs, histories, source-video
byte hashes, and filename families such as `name(1)` / `name(2)`. Filename hints
are not verified provenance. Re-encoded clips and different recordings of one
flight can remain undetected. Review the manifest before using these groups for
training or evaluation. No train/test assignments are made here.

## Interpretation

Path retention compares lengths across identical retained segments. Spectral
retention compares **power**, not amplitude, at 3-10 Hz. It uses a gap-free run,
excludes SG edge samples and removes a linear trend. The retained interval must
be at least one second and source Nyquist at least 10 Hz. This band is a diagnostic
choice, not an assertion that it contains only wingbeat motion. Short recordings
have limited frequency resolution.

Raw-to-CMC RMS measures correction magnitude, not correction error. The input
contains no ground-truth positions. CMC correctness and biological interpretation
of residuals require video review or additional annotation.

Existing RF + RuleFilter default predictions are replayed through runtime B with
a research/runtime feature parity check. This is a file-weighted development
replay, not an independent test or grouped cross-validation. UI-specific threshold
overrides are not applied. Model and source hashes and feature provenance are
saved in `run.json`. Alternate smoothing feature tables are diagnostic only;
they are not fed into the unchanged trained model as a performance experiment.

## Verification

```powershell
.\venv\Scripts\python.exe -X utf8 -B -m pytest research/tests/test_features.py research/tests/test_preprocessing_diagnostic.py tests/test_feature_core.py tests/test_rule_filter.py -q -p no:cacheprovider
```

Tests cover unchanged feature results and inputs, long-gap separation, rejection
traces, analytic sinusoid attenuation, unsupported spectral intervals, exact raw
matching, provisional grouping and artifact generation without raw observations.
