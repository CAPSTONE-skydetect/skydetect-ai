# EXP-STACK-PORTABLE-19 protocol

Remote model/minirocket verified by ls-remote at
d31a8fe9606e6adc71e10f798aff837b4a708f36. New local
codex/minirocket-stack-training worktree. Read-only Stack source d51bbd8:

| Research file | SHA256 |
| --- | --- |
| minirocket_stack.py | 1177a146234c8c5a685fdfabb2c0d61d0da412b14a0e5e0bf01e8695aff0c8ba |
| minirocket_framescale.py | b088fcbeae2c69c1d83f2f52265f0ec305797b882e095f0ed29dee5fd00ac3dc |
| minirocket_retrain.py | 439d57547ddd7088acc6d7f3d9955e0636855a38009a047fa4e4e8b52f166638 |
| minirocket_inference.py | 2b22ee89d673e5e2823b5b10a4e1028e1093ff8f919ae289960c58c610d2f51e |
| minirocket_predict.py | d4e97b395e431efed4594dd67806480b4e10d5178e33bf0c2ee834410e541f67 |

Reuse collaborator sequence contracts, model/weights/group metrics and pinned
dependencies. Separate opt-in CLI only. Select each member alpha by the
inherited train-only 3-fold group CV and group macro F1 (smallest alpha wins
ties), then freeze alphas for member OOF and full fitting. OOF meta fitting
is not independent evaluation (member alpha selection used the same train
pool). Keep gain 400, C=1, sign at zero and track/group mean aggregation.
The collaborator metric has no mixed-label group decision; fail explicitly
rather than split one source or silently change that metric.

Validate original packages without fitting; exercise the new collection
manifest on synthetic fixtures only. One synthetic full smoke, one repair
rerun if needed. Frozen checkpoint parity on only two preselected records
(first accepted bird and first accepted drone in experiment 18 with both
predictions), abs/rel tolerance 1e-8. Local clean subprocess with no
cross-worktree source imports. No actual dataset fitting, serving changes,
push, merge or PR. Four-hour limit; return evidence to Astra for review.
