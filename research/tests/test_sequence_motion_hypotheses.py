import pandas as pd

from research.evaluate_sequence_motion_hypotheses import select_candidate


def test_candidate_needs_both_distance_and_usable_adjusted_objective_improvement():
    rows = []
    for fold in range(3):
        rows.extend([dict(fold=fold, candidate="baseline", distance=1.0, objective=1.4),
                     dict(fold=fold, candidate="camera_200m", distance=1.05, objective=1.1)])
    assert select_candidate(pd.DataFrame(rows)) is None
    for row in rows:
        if row["candidate"] == "camera_200m":
            row["distance"] = .8
    assert select_candidate(pd.DataFrame(rows)) == "camera_200m"
