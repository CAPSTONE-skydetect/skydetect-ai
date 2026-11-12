"""E1: 증강 학습(B) 모델에 이동 크기를 보존하는 모델을 더한 Stack 을 5-case 로 평가한다.

멤버 2개
  baseline   : 창마다 정규화한 입력 (E0 의 B 와 같은 모델, alpha 는 E0 결과를 그대로 사용)
  framescale : 같은 창을 normalization_scale 로 되돌려 화면 기준 이동 크기를 보존한 입력
결합기: 두 멤버의 track 평균 점수 위에 LogisticRegression. train 안의 그룹 단위 3-fold
        교차 예측(OOF)으로만 학습하며 val/test 는 결합기 학습에 쓰지 않는다.
framescale 멤버의 alpha 는 E0 와 같은 방식으로 case 의 val 그룹 balanced accuracy 로 고른다.

    python -m research.c_part_cv5_stack --data research/output/c_part_cv5_v1 \
      --baseline research/output/c_part_cv5_E0 --output research/output/c_part_cv5_E1 --no-test
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ai_server.services import sequence_model as seq
from . import c_part_cv5_experiment as cv5
from . import c_part_minirocket_experiment as exp
from .minirocket_stack_training import scale_windows
from .plot_confusion import plot

ALPHAS = (1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000)
INNER_FOLDS = 3


def load(case_dir: Path, package: str) -> dict:
    data = cv5.load(case_dir, package)
    meta = pd.read_csv(case_dir / package / "minirocket_windows.csv", encoding="utf-8-sig").set_index("window_id")
    data["scale"] = meta.loc[data["sample_id"]].normalization_scale.to_numpy()
    data["X_frame"] = scale_windows(data["X"], data["scale"])
    return data


def decisions(rocket, scaler, ridge, x):
    return ridge.decision_function(seq.transform(rocket, scaler, x))


def track_mean(data: dict, scores: np.ndarray) -> pd.DataFrame:
    table = pd.DataFrame(dict(track=seq.track_ids(data["sample_id"]), group=data["group_id"], label=data["y"], **scores))
    return table.groupby(["track", "group", "label"], as_index=False).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="E0 결과 폴더 (B 의 alpha 와 점수 대조용)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--keep-variants", type=int, nargs="+",
                        help="train_augmented 에서 남길 변형 번호 (0=원본, 1~9=증강). 생략하면 전부")
    parser.add_argument("--reselect-baseline-alpha", action="store_true",
                        help="baseline 멤버 alpha 도 framescale 처럼 val 그룹 balanced accuracy 로 다시 고른다")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "pred").mkdir()
    evals = ["validation_real", "validation_augmented"] + ([] if args.no_test else ["test_real", "test_augmented"])
    base_rows = pd.read_csv(args.baseline / "case_results.csv", keep_default_na=False)
    base_alpha = base_rows[base_rows.scenario == "B"].drop_duplicates("case").set_index("case").alpha.to_dict()
    started, log_lines, rows = time.time(), [], []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    for k in args.cases:
        t0 = time.time()
        case_dir = args.data / f"case_{k}"
        train = load(case_dir, "train_augmented")
        if args.keep_variants is not None:
            index = np.array([int(sid.rsplit(":", 2)[1]) for sid in train["sample_id"]])
            keep = np.isin(index, args.keep_variants)
            train = {key: value[keep] for key, value in train.items()}
            log(f"case {k}: kept variants {sorted(args.keep_variants)} -> {keep.sum()} of {len(keep)} train windows")
        sets = {name: load(case_dir, name) for name in evals}
        for name, d in sets.items():
            if set(d["group_id"]) & set(train["group_id"]):
                raise ValueError(f"case {k}: group leak {name}")
        members = {"baseline": "X", "framescale": "X_frame"}
        fitted, alpha = {}, {"baseline": float(base_alpha[k])}
        val_dec = {}
        for name, key in members.items():
            search = name == "framescale" or args.reselect_baseline_alpha
            rocket, scaler, ridges = exp.fit_model(train[key], train["y"], train["group_id"],
                                                   ALPHAS if search else [alpha["baseline"]])
            if search:
                scores = []
                for a, r in ridges.items():
                    m = exp.summary(sets["validation_real"], decisions(rocket, scaler, r, sets["validation_real"][key]))
                    scores.append((m["group"]["balanced_accuracy"], -a, a))
                alpha[name] = float(max(scores)[2])
            fitted[name] = (rocket, scaler, ridges[alpha[name]])
        log(f"case {k}: member alphas {alpha} ({time.time() - t0:.0f}s)")

        # 결합기 학습용 OOF: train 안에서 그룹 단위로 나눠 각 멤버를 다시 학습
        splitter = StratifiedGroupKFold(INNER_FOLDS, shuffle=True, random_state=seq.RANDOM_STATE)
        oof = {name: np.full(len(train["y"]), np.nan) for name in members}
        for fit_idx, held_idx in splitter.split(train["X"], train["y"], train["group_id"]):
            for name, key in members.items():
                rocket, scaler, ridges = exp.fit_model(train[key][fit_idx], train["y"][fit_idx],
                                                       train["group_id"][fit_idx], [alpha[name]])
                oof[name][held_idx] = decisions(rocket, scaler, ridges[alpha[name]], train[key][held_idx])
        meta_tracks = track_mean(train, oof)
        z = meta_tracks[list(members)].to_numpy()
        meta_scaler = StandardScaler().fit(z)
        weight = seq.balanced_group_weights(meta_tracks.label.to_numpy(), meta_tracks.group.to_numpy())
        head = LogisticRegression(C=1.0, random_state=seq.RANDOM_STATE).fit(
            meta_scaler.transform(z), meta_tracks.label, sample_weight=weight)
        if head.classes_.tolist() != seq.LABELS:
            raise ValueError("Unexpected class order")
        log(f"case {k}: head coef (baseline, framescale) = {head.coef_[0].round(3).tolist()} ({time.time() - t0:.0f}s)")

        def stack_windows(d):
            """창 단위 점수: 멤버 점수를 표준화해 결합기의 선형식에 그대로 넣는다 (track 평균과 같은 선형 결합)."""
            cols = np.column_stack([decisions(*fitted[n], d[key]) for n, key in members.items()])
            return head.decision_function(meta_scaler.transform(cols)), cols

        dec = {}
        for name, d in sets.items():
            dec[name], cols = stack_windows(d)
            pd.DataFrame(dict(sample_id=d["sample_id"], group_id=d["group_id"], label=d["y"], variant=d["variant"],
                              decision=dec[name], baseline=cols[:, 0], framescale=cols[:, 1])).to_csv(
                args.output / "pred" / f"case{k}_S_{name}.csv", index=False)
        threshold = exp.best_threshold(exp.tables(sets["validation_real"], dec["validation_real"])["track"])
        for name, d in sets.items():
            for tag, thr in (("t0", 0.0), ("tuned", threshold)):
                m = exp.summary(d, dec[name], thr)
                for level in cv5.LEVELS:
                    rows.append(dict(case=k, scenario="S", alpha=alpha["framescale"], threshold=threshold, eval=name,
                                     threshold_mode=tag, level=level, **cv5.flat(m[level])))
        log(f"case {k}: done ({time.time() - t0:.0f}s)")

    result = pd.DataFrame(rows)
    result.assign(cm=result.cm.map(json.dumps)).to_csv(args.output / "case_results.csv", index=False)
    summary = {}
    for (name, tag, level), part in result.groupby(["eval", "threshold_mode", "level"]):
        key = f"S|{name}|{tag}|{level}"
        summary[key] = {c: dict(mean=float(part[c].mean()), std=float(part[c].std(ddof=1)))
                        for c in ("balanced_accuracy", "recall_bird", "recall_drone", "roc_auc", "macro_f1")}
        summary[key]["confusion_matrix_sum"] = np.sum(np.array(part.cm.tolist()), axis=0).tolist()
    plot(summary["S|validation_real|t0|track"]["confusion_matrix_sum"],
         f"{args.output.name} Stack (track, val, {len(args.cases)} cases pooled)",
         args.output / "confusion_S_val_track.png")
    meta = dict(data=str(args.data), baseline=str(args.baseline), alphas=list(ALPHAS), no_test=args.no_test,
                keep_variants=args.keep_variants, reselect_baseline_alpha=args.reselect_baseline_alpha,
                inner_folds=INNER_FOLDS, versions=seq.library_versions(), python=platform.python_version(),
                runtime_s=round(time.time() - started, 1), summary=summary)
    (args.output / "results.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.output / "log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"done in {meta['runtime_s']}s")


if __name__ == "__main__":
    main()
