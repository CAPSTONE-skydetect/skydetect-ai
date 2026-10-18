# Local registration and complementary fusion experiments

Date: 2026-10-18. These are development results on four videos/six objects.

Implemented optional `contrast_local_centroid` (N6) and `klt_patch_registration`
(N7-N9). Defaults remain unchanged because no variant passes all regression gates.
Registration removes slow background intensity variation and searches near the
KLT proposal with a subpixel correlation peak. Weak, ambiguous or boundary peaks
fall back to KLT. It does not use GT or smooth the trajectory.

| Variant | Measurement change | Median object position P95, 1280px | Mean observation rate | Median absolute relative acceleration P95 feature error |
| --- | --- | ---: | ---: | ---: |
| D0 | Existing tracker | 3.851 | 99.68% | 59.62% |
| N6 | Local contrast centroid | 4.774 | 99.83% | 71.56% |
| N7 | Patch registration, contrast OFF | 5.965 | 99.34% | 11.82% |
| N8 | Registration plus original contrast | 4.313 | 98.88% | 80.53% |
| N9 | Registration plus local centroid | 4.766 | 99.83% | 73.31% |
| N10 | Causal fusion of saved D0 position and N7 motion | 4.376 | 99.68% | 15.75% |

The table uses each variant's own observations. Paired comparisons in
`evaluation/paired_comparison.json` recompute all nine actual B features on
exactly shared frames. For N7 the paired baseline acceleration error is 63.09%,
versus 11.82%; N10 preserves all 1,529 D0 observed frame IDs/timestamps and its
paired baseline is unchanged at 59.62%.

N10 implements a causal complementary filter: motion position plus a slowly
updated D0-minus-motion offset, fixed time constant 0.25 seconds. It uses no GT
to generate coordinates. Missing, inconsistent or out-of-bounds motion falls
back to D0. D0's background camera transforms are applied to both the fused
coordinates and GT. Existing gaps, timestamps, sizes and confidence values are
preserved; quality is recalculated. This is an offline two-pass research path,
not integrated into the default API/UI. Confidence is inherited from D0 and has
not been calibrated for fusion uncertainty.

N10 reduces position P95 for drone_1_drone, drone_5 and drone_6; it worsens
bird_2 and both birds. Its acceleration P95 feature error improves for five of
six objects. Other features are mixed: median relative turn_rate_p95 error
increases from 10.12% to 16.58%, and heading_change_ratio from 11.04% to 27.99%.
A lower acceleration statistic error alone is not sufficient for adoption.
There is no independent holdout validation of these new variants.

Frozen deployed models on N10: MiniRocket 4/6 correct (unchanged from D0),
RF 0/6 (D0 1/6). MiniRocket production/research parity holds on all 18 inputs
(A, GT matched, GT window for six objects). This is inference, not retraining;
training-source overlap and model-training differences remain unresolved.

Reproduce from the repository root with the existing tracking environment:

```sh
python -m research.trajectory_experiments run --manifest artifacts/trajectory_experiments/pilot_manifest.json --output artifacts/trajectory_experiments --variants N6 N7 N8 N9
python -m research.trajectory_experiments.fusion --root artifacts/trajectory_experiments/pilot-20261017
python -m research.trajectory_experiments.compare --root artifacts/trajectory_experiments/pilot-20261017/evaluation --variants N6 N7 N8 N9 N10
```

Fusion requires the completed D0 and N7 artifacts and verified D0 camera
transforms. It rewrites only its own N10 derived outputs. Detailed scores and
handoff JSONs are under `evaluation/N10/`; deployment inference is under
`deployed_N10/`. No claim of a complete tracking breakthrough is supported yet.
