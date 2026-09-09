"""Class-conditional diagnostics, not an automatic certificate of realism."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from .evaluation import validate_table
from .features import FEATURE_COLUMNS
from .io import write_json


def _domain_score(sim, real, seed):
    # Original video groups and synthetic families never cross a fold boundary.
    if min(sim.family_id.nunique(), real.family_id.nunique()) < 6:
        return dict(status="insufficient_groups_for_domain_cv")
    x = pd.concat([sim, real], ignore_index=True)
    y = np.r_[np.zeros(len(sim), dtype=int), np.ones(len(real), dtype=int)]
    groups = np.r_["sim:" + sim.family_id.astype(str), "real:" + real.family_id.astype(str)]
    scores = []
    for train, test in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(x, y, groups):
        if min(len(set(y[train])), len(set(y[test]))) < 2:
            continue
        model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=3,
                                       class_weight="balanced", random_state=seed, n_jobs=1)
        model.fit(x.iloc[train][FEATURE_COLUMNS], y[train])
        scores.append(float(balanced_accuracy_score(y[test], model.predict(x.iloc[test][FEATURE_COLUMNS]))))
    return dict(status="diagnostic_only", fold_balanced_accuracy=scores,
                interpretation="Near 0.5 is NOT proof of equivalence; low power can hide domain shift.")


def compare_domains(synthetic, real, seed=42):
    sim, obs = validate_table(synthetic), validate_table(real)
    if not len(obs):
        return dict(status="no_accepted_real_tracks")
    if sim.empty or sim.feature_config_id.iloc[0] != obs.feature_config_id.iloc[0]:
        raise ValueError("Need synthetic data with matching feature configuration")
    sim = sim[sim.split == "train"]
    # Calibration may guide simulator development; final test must stay untouched.
    review_counts = obs.review_status.value_counts().to_dict() if "review_status" in obs else {}
    if "review_status" not in obs:
        raise ValueError("Real calibration requires manual review_status")
    obs = obs[(obs.split == "calibration") & (obs.review_status == "approved")]
    if obs.empty:
        return dict(status="no_real_calibration_tracks_test_not_used")
    report = dict(status="descriptive_only", scope="real_calibration_vs_synthetic_train",
                  input_manual_review_counts=review_counts,
                  validity="No universal Wasserstein/p-value cutoff certifies a simulator.", classes={})
    for label in ("bird", "drone"):
        a, b = sim[sim.label == label], obs[obs.label == label]
        if a.empty or b.empty:
            report["classes"][label] = dict(status="missing_class")
            continue
        # Equal weight per original video/family prevents window count dominance.
        aw = 1 / a.groupby("family_id").family_id.transform("size").to_numpy()
        bw = 1 / b.groupby("family_id").family_id.transform("size").to_numpy()
        metrics = {}
        for feature in FEATURE_COLUMNS:
            av, bv = a[feature].to_numpy(), b[feature].to_numpy()
            order = np.argsort(av)
            q05, q25, q75, q95 = np.interp([.05, .25, .75, .95], np.cumsum(aw[order])/aw.sum(), av[order])
            distance = float(wasserstein_distance(av, bv, aw, bw))
            metrics[feature] = dict(wasserstein=distance, synthetic_iqr=float(q75-q25),
                                    wasserstein_per_iqr=distance/(q75-q25) if q75-q25 > 1e-10 else None,
                                    real_coverage_in_sim_q05_q95=float(np.average((bv >= q05) & (bv <= q95), weights=bw)),
                                    synthetic_q05_q95=[float(q05), float(q95)],
                                    real_range=[float(bv.min()), float(bv.max())])
        baseline = None
        groups = b.family_id.unique()
        if len(groups) >= 6:
            rng = np.random.default_rng(seed)
            distances = {f: [] for f in FEATURE_COLUMNS}
            for _ in range(100):
                order = rng.permutation(groups)
                left = b[b.family_id.isin(order[:len(order)//2])]
                right = b[~b.family_id.isin(order[:len(order)//2])]
                lw = 1/left.groupby("family_id").family_id.transform("size").to_numpy()
                rw = 1/right.groupby("family_id").family_id.transform("size").to_numpy()
                for f in FEATURE_COLUMNS:
                    distances[f].append(wasserstein_distance(left[f], right[f], lw, rw))
            baseline = {f: np.quantile(v, [.05, .5, .95]).tolist() for f, v in distances.items()}
        report["classes"][label] = dict(real_groups=len(groups), synthetic_groups=a.family_id.nunique(),
                                         real_rows=len(b), synthetic_rows=len(a), metrics=metrics,
                                         real_real_split_distance_q05_q50_q95=baseline,
                                         domain_classifier=_domain_score(a, b, seed),
                                         manual_review_counts=b.review_status.value_counts().to_dict() if "review_status" in b else {})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("synthetic_csv", type=Path)
    parser.add_argument("real_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, compare_domains(pd.read_csv(args.synthetic_csv), pd.read_csv(args.real_csv)))


if __name__ == "__main__":
    main()
