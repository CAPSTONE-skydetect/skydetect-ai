# Frozen D0 / MiniRocket comparison, 2026-10-18

## Fixed comparison

A remains the saved D0 output. The baseline is the existing runtime model;
the candidate is the saved learned dual-scale stack. Neither model is fitted
or tuned by the comparison runner.

Model SHA-256:

- Baseline: `9024731a649b5e0a84ead3fcc45c9dc0a692ce305cea6e5ec8b7d720867a1e5a`
- Candidate: `5ba8fc4ac843f30c8a2a590add3f5854f6cf23093d1b8ed1f3b0731b95ec8aab`

`research/minirocket_frozen_comparison.py` checks those hashes, input hashes,
schema, paired frame/timestamp support, processed resolution and stabilization
metadata before inference. Results retain window scores, rejection reasons,
input/model identities and the runner hash. Margins are not probabilities;
baseline and candidate margins use different scales.

## Executed auxiliary comparison

| Input | Baseline | Candidate | Ground truth |
| --- | --- | --- | --- |
| drone_3 matched GT | bird, -0.72961 | bird, -3.25165 | drone |
| drone_3 D0 | bird, -0.73703 | bird, -1.96323 | drone |

Each input yields only one accepted two-second window. Both models are wrong
on both inputs. Replacing A coordinates with GT does not fix this case. This
does not establish the cause or prove that all A errors are harmless.

This source was previously inspected. Its object is about 57-86 original
pixels, and GT tail visibility/repeated boxes need human review. Do not add
this result to the six-object pilot as an independent final test or claim a
generalization improvement. The GT and original scores were not edited.

## Reproduce

From the experiment repository root:

```sh
/Users/yuchan/Documents/형주플젝/skydetect-ai-minirocket-runtime/.venv/bin/python research/minirocket_frozen_comparison.py --manifest artifacts/trajectory_experiments/frozen-drone3-20261018.json --output artifacts/trajectory_experiments/frozen-drone3-results-repeat
```

The output directory must be new. Artifacts from the executed run are in
`artifacts/trajectory_experiments/frozen-drone3-results-20261018/`.

## Next data

The local labeling directory currently has five XML exports: the four pilot
videos and drone_3. No unused paired GT/A source is available there. Other
videos exist but require object identity, filming-source independence, GT and
an initial manual ROI to be confirmed. Folder names cannot supply labels.

For the next source, add a sample to a separate manifest using the same model
hashes, explicit bird/drone truth, source_group_id, and hashed GT/D0 JSON paths.
The GT must use the same observed frames, timestamps and coordinate system as
D0. Keep full GT and missing-frame coverage in the upstream A evaluation;
this classifier comparison uses matched support only. The same two-second
windowing and existing model decision policies apply without threshold tuning.

Evaluate both bird and drone sources, including 10-30px targets. Report object
and source-group outcomes, abstentions and window counts. Do not count
overlapping windows as independent test objects. Review errors before deciding
whether to train on the new data; once used for development, reserve other
filming sources for final evaluation.
