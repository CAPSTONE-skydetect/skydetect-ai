"""C파트 final_observation_v1 데이터셋으로 MiniRocket 실험 A/B/D 를 수행한다.

데이터: data/c_part/final_observation_v1/dataset/<package>/minirocket.npz
레시피: ai_server.services.sequence_model 의 MiniRocket(10k)+StandardScaler+Ridge,
클래스·그룹 균형 가중치를 그대로 쓴다.

| 실험 | Train            | Validation            | Test            |
| A    | train_real       | validation_real       | test_real       |
| B    | train_augmented  | validation_real       | test_real       |
| D    | (B 모델 재사용)   | validation_augmented  | test_augmented  |

5-fold CV: fold k 에서 train_real[fold == k] 가 공통 검증 세트.
A 는 train_real[fold != k], B/D 는 fold_k_train_augmented 로 fit.
MiniRocket·scaler·Ridge 는 각 fold 의 fit 자료에만 적합한다.
alpha 는 CV 의 그룹 macro-F1 평균으로, 판정 임계값은 validation_real 의 track
balanced accuracy 로 고른다. test 는 마지막에 한 번만 평가한다.

    python -m research.c_part_minirocket_experiment --output research/output/c_part_minirocket_v1
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier

from ai_server.services import sequence_model as seq

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "c_part" / "final_observation_v1" / "dataset"
FOLDS = 5


def load(package: str) -> dict:
    folder = DATASET / package
    with np.load(folder / "minirocket.npz", allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in ("X", "y", "group_id", "sample_id", "fold")}
    seq.validate_windows(data["X"])
    if len(set(data["sample_id"])) != len(data["y"]) or set(data["y"]) != set(seq.LABELS):
        raise ValueError(f"Bad package: {package}")
    meta = pd.read_csv(folder / "minirocket_windows.csv", encoding="utf-8-sig").set_index("window_id")
    rows = meta.loc[data["sample_id"]]
    if not (np.array_equal(rows.label.to_numpy(), data["y"])
            and np.array_equal(rows.source_group_id.to_numpy(), data["group_id"])):
        raise ValueError(f"Window metadata mismatch: {package}")
    data["variant"] = rows.variant.to_numpy().astype(str)
    data["parent"] = rows.parent_track_id.to_numpy().astype(str)
    return data


def subset(data: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in data.items()}


def tables(data: dict, decision: np.ndarray) -> dict[str, pd.DataFrame]:
    windows, tracks, groups = seq.aggregate_scores(data, decision)
    windows["variant"] = data["variant"]
    return dict(window=windows, track=tracks, group=groups)


def thresholded(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return table.assign(decision=table.decision - threshold)


def summary(data: dict, decision: np.ndarray, threshold: float = 0.0) -> dict:
    out = {level: seq.metrics(thresholded(t, threshold)) for level, t in tables(data, decision).items()}
    return out


def fit_model(x, y, groups, alphas):
    rocket, scaler = seq.fit_representation(x)
    z = seq.transform(rocket, scaler, x)
    weight = seq.balanced_group_weights(y, groups)
    ridges = {a: RidgeClassifier(alpha=a).fit(z, y, sample_weight=weight) for a in alphas}
    for ridge in ridges.values():
        if ridge.classes_.tolist() != seq.LABELS:
            raise ValueError("Unexpected class order")
    return rocket, scaler, ridges


def decide(rocket, scaler, ridge, x):
    return ridge.decision_function(seq.transform(rocket, scaler, x))


def cross_validate(real: dict, log) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """A / B 각각 5-fold CV. 반환: 시나리오별 OOF 결정값(alpha별), fold 지표표."""
    oof = {s: {a: np.full(len(real["y"]), np.nan) for a in seq.ALPHAS} for s in ("A", "B")}
    rows = []
    for k in range(FOLDS):
        held = real["fold"] == k
        hold = subset(real, held)
        aug = load(f"fold_{k}_train_augmented")
        if set(aug["group_id"]) & set(hold["group_id"]):
            raise ValueError(f"fold {k}: augmented fit overlaps holdout groups")
        fits = {"A": subset(real, ~held), "B": aug}
        for scenario, fit in fits.items():
            if set(fit["group_id"]) & set(hold["group_id"]):
                raise ValueError(f"fold {k} {scenario}: group leak")
            t0 = time.time()
            rocket, scaler, ridges = fit_model(fit["X"], fit["y"], fit["group_id"], seq.ALPHAS)
            z = seq.transform(rocket, scaler, hold["X"])
            for alpha, ridge in ridges.items():
                d = ridge.decision_function(z)
                oof[scenario][alpha][held] = d
                m = summary(hold, d)
                rows.append(dict(fold=k, scenario=scenario, alpha=alpha, fit_windows=len(fit["y"]),
                                 holdout_windows=int(held.sum()),
                                 **{f"{lvl}_{key}": m[lvl][key] for lvl in ("window", "track", "group")
                                    for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")}))
            log(f"  fold {k} {scenario}: fit {len(fit['y'])} windows, {time.time() - t0:.0f}s")
    folds = pd.DataFrame(rows)
    chosen = (folds.groupby(["scenario", "alpha"]).group_macro_f1.mean().reset_index()
              .sort_values(["scenario", "group_macro_f1", "alpha"], ascending=[True, False, True])
              .groupby("scenario").head(1).set_index("scenario").alpha.to_dict())
    return oof, folds, chosen


def best_threshold(track_table: pd.DataFrame) -> float:
    """validation track balanced accuracy 최대 임계값. 동률이면 0 에 가장 가까운 값."""
    d = np.sort(track_table.decision.unique())
    candidates = np.unique(np.concatenate([[0.0], (d[1:] + d[:-1]) / 2]))
    truth = track_table.label.to_numpy() == "drone"
    best, best_score = 0.0, -1.0
    for c in sorted(candidates, key=abs):
        pred = track_table.decision.to_numpy() >= c
        score = (pred[truth].mean() + (~pred[~truth]).mean()) / 2
        if score > best_score + 1e-12:
            best, best_score = float(c), score
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    real = load("train_real")
    log("[1/3] 5-fold CV (A: real, B: fold_k_train_augmented)")
    oof, folds, chosen = cross_validate(real, log)
    folds.to_csv(out / "cv_folds.csv", index=False)
    cv = {}
    for s in ("A", "B"):
        a = chosen[s]
        part = folds[(folds.scenario == s) & (folds.alpha == a)]
        cv[s] = dict(alpha=a, pooled_oof=summary(real, oof[s][a]),
                     fold_mean={c: float(part[c].mean()) for c in part.columns if c.endswith(("macro_f1", "balanced_accuracy", "roc_auc", "accuracy"))},
                     fold_std={c: float(part[c].std(ddof=1)) for c in part.columns if c.endswith(("macro_f1", "balanced_accuracy", "roc_auc", "accuracy"))})
        pd.DataFrame(dict(sample_id=real["sample_id"], group_id=real["group_id"], label=real["y"],
                          fold=real["fold"], decision=oof[s][a])).to_csv(out / f"cv_oof_{s}.csv", index=False)
    log(f"  chosen alpha: {chosen}")

    log("[2/3] final fit on full train")
    train = {"A": real, "B": load("train_augmented")}
    models = {}
    for s, data in train.items():
        rocket, scaler, ridges = fit_model(data["X"], data["y"], data["group_id"], [chosen[s]])
        models[s] = (rocket, scaler, ridges[chosen[s]])
        bundle = dict(rocket=rocket, scaler=scaler, classifier=ridges[chosen[s]], classes=seq.LABELS,
                      alpha=chosen[s], arm=f"c_part_final_observation_v1_{s}",
                      contract_version=seq.CONTRACT_VERSION, contract_id=seq.CONTRACT_ID,
                      channels=["qx", "qy", "dx", "dy"], score_type=seq.SCORE_TYPE, score_is_probability=False,
                      train_windows=len(data["y"]), train_groups=len(set(data["group_id"])),
                      versions=seq.library_versions())
        joblib.dump(bundle, out / f"model_{s}.joblib")

    log("[3/3] validation / test evaluation")
    evals = {}
    sets = {"validation_real": load("validation_real"), "test_real": load("test_real"),
            "validation_augmented": load("validation_augmented"), "test_augmented": load("test_augmented")}
    decisions = {}
    for s in ("A", "B"):
        for name, data in sets.items():
            d = decide(*models[s], data["X"])
            decisions[(s, name)] = d
            pd.DataFrame(dict(sample_id=data["sample_id"], group_id=data["group_id"], label=data["y"],
                              variant=data["variant"], decision=d)).to_csv(out / f"pred_{s}_{name}.csv", index=False)
    thresholds = {s: best_threshold(tables(sets["validation_real"], decisions[(s, "validation_real")])["track"])
                  for s in ("A", "B")}
    # D 는 B 모델을 쓰므로 B 의 실촬영 validation 임계값을 그대로 쓴다.
    for s in ("A", "B"):
        for name, data in sets.items():
            d = decisions[(s, name)]
            label = "D" if (s == "B" and name.endswith("augmented")) else s
            key = f"{label}:{s}_model@{name}"
            evals[key] = dict(threshold0=summary(data, d), tuned=summary(data, d, thresholds[s]),
                              group_bootstrap=seq.group_bootstrap_ci(tables(data, d)["group"]))
            if name.endswith("augmented"):
                w = tables(data, d)["track"]
                variant = pd.Series(data["variant"], index=seq.track_ids(data["sample_id"])).groupby(level=0).first()
                w["variant"] = variant.loc[w.track_id].to_numpy()
                evals[key]["by_variant_track"] = {v: seq.metrics(g) for v, g in w.groupby("variant")}

    result = dict(dataset=str(DATASET.relative_to(ROOT)).replace("\\", "/"),
                  dataset_manifest_sha256=seq.file_hash(DATASET / "manifest.json"),
                  recipe=dict(n_kernels=seq.N_KERNELS, seed=seq.RANDOM_STATE, alphas=list(seq.ALPHAS),
                              weights="balanced class/group", alpha_selection="5-fold CV mean group macro-F1, tie→smallest",
                              threshold_selection="validation_real track balanced accuracy, tie→closest to 0"),
                  chosen_alpha=chosen, thresholds=thresholds, cv=cv, evaluation=evals,
                  versions=seq.library_versions(), python=platform.python_version(), os=platform.platform(),
                  runtime_s=round(time.time() - started, 1))
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"done in {result['runtime_s']}s → {out}")


if __name__ == "__main__":
    main()
