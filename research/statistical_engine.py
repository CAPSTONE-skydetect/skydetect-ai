"""Descriptive stratified diagnostics without Gaussian validity assumptions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import validate_table
from .features import FEATURE_COLUMNS
from .io import write_json


def describe_dataset(table):
    valid = validate_table(table)
    strata = {}
    for dimension in ("label", "subtype", "scenario", "behavior_mode", "observation_profile", "split",
                      "requested_depth_mode", "training_length_group"):
        if dimension not in table:
            continue
        groups = {}
        for name, raw in table.groupby(dimension, observed=True):
            group = valid[valid[dimension] == name]
            groups[str(name)] = dict(attempted=len(raw), accepted=len(group),
                                     acceptance_ratio=len(group)/len(raw),
                                     rejected_reasons=raw.loc[raw.feature_status != "accepted", "rejection_reason"].fillna("unknown").value_counts().to_dict(),
                                     features={f: dict(mean=float(group[f].mean()),
                                                       quantiles=group[f].quantile([.05, .25, .5, .75, .95]).tolist())
                                               for f in FEATURE_COLUMNS} if len(group) else {})
        strata[dimension] = groups
    pairing = {}
    if "observation_profile" in valid:
        paired = valid.copy()
        paired["latent_id"] = paired.sample_id.str.rsplit(":", n=1).str[0]
        for feature in FEATURE_COLUMNS:
            pivot = paired.pivot(index="latent_id", columns="observation_profile", values=feature)
            if {"ideal", "noisy"}.issubset(pivot.columns):
                delta = (pivot.noisy-pivot.ideal).dropna()
                pairing[feature] = dict(pairs=len(delta),
                                        noisy_minus_ideal_quantiles=delta.quantile([.05, .5, .95]).tolist() if len(delta) else [])
    return dict(scope="synthetic_internal_diagnostics", total=len(table), accepted=len(valid),
                real_world_validity="not_measured", strata=strata, observation_sensitivity=pairing,
                note="No p-value or feature separation is interpreted as proof of physical realism.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, describe_dataset(pd.read_csv(args.csv)))


if __name__ == "__main__":
    main()
