"""C파트 MiniRocket 실험 A/B/D 를 5-case 교차검증으로 수행한다.

데이터: research.build_cv5_dataset 이 만든 case_k/{train,validation,test}_{real,augmented}
case 마다 train(3조각)으로 fit 하고, alpha·임계값은 그 case 의 validation 으로만 고른다.
개선 단계 비교는 5개 case validation 결과로 하고(--no-test), test 는 최종 후보에 한 번만 쓴다.

지표: 혼동행렬(5개 case 합산), balanced accuracy, 새/드론 recall, AUC.
alpha 선택: validation 그룹 balanced accuracy 최대(동률이면 작은 값).

    python -m research.c_part_cv5_experiment --data research/output/c_part_cv5_v1 \
      --output research/output/c_part_cv5_E0 --no-test
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ai_server.services import sequence_model as seq
from . import c_part_minirocket_experiment as exp

LEVELS = ("window", "track", "group")


def load(case_dir: Path, package: str) -> dict:
    exp.DATASET = case_dir
    return exp.load(package)


def flat(m: dict) -> dict:
    """seq.metrics 결과에서 기록할 값만 뽑는다."""
    return dict(balanced_accuracy=m["balanced_accuracy"], recall_bird=m["recall"]["bird"],
                recall_drone=m["recall"]["drone"], roc_auc=m["roc_auc"], macro_f1=m["macro_f1"],
                n=m["n"], cm=m["confusion_matrix"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000])
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--no-test", action="store_true",
                        help="test_real/test_augmented 를 읽지도 평가하지도 않는다 (개선 단계용)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "pred").mkdir()
    alphas = tuple(args.alphas)
    evals = ["validation_real", "validation_augmented"]
    if not args.no_test:
        evals += ["test_real", "test_augmented"]
    started = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    rows, alpha_rows = [], []
    for k in args.cases:
        case_dir = args.data / f"case_{k}"
        sets = {name: load(case_dir, name) for name in ["train_real", "train_augmented", *evals]}
        groups = {n: set(d["group_id"]) for n, d in sets.items()}
        for a in ("train_real", "train_augmented"):
            for b in evals:
                if groups[a] & groups[b]:
                    raise ValueError(f"case {k}: group leak {a} vs {b}")
        if not args.no_test and groups["validation_real"] & groups["test_real"]:
            raise ValueError(f"case {k}: validation/test group leak")
        for scenario, train_name in (("A", "train_real"), ("B", "train_augmented")):
            t0 = time.time()
            train = sets[train_name]
            rocket, scaler, ridges = exp.fit_model(train["X"], train["y"], train["group_id"], alphas)
            z = {name: seq.transform(rocket, scaler, sets[name]["X"]) for name in evals}
            dec = {name: {a: r.decision_function(z[name]) for a, r in ridges.items()} for name in evals}
            scores = []
            for a in alphas:
                m = exp.summary(sets["validation_real"], dec["validation_real"][a])
                scores.append(dict(case=k, scenario=scenario, alpha=a,
                                   group_balanced_accuracy=m["group"]["balanced_accuracy"],
                                   track_balanced_accuracy=m["track"]["balanced_accuracy"]))
            table = pd.DataFrame(scores)
            alpha_rows.append(table)
            best = float(table.sort_values(["group_balanced_accuracy", "alpha"],
                                           ascending=[False, True]).iloc[0].alpha)
            threshold = exp.best_threshold(exp.tables(sets["validation_real"], dec["validation_real"][best])["track"])
            for name in evals:
                d = sets[name]
                pd.DataFrame(dict(sample_id=d["sample_id"], group_id=d["group_id"], label=d["y"],
                                  variant=d["variant"], decision=dec[name][best])).to_csv(
                    args.output / "pred" / f"case{k}_{scenario}_{name}.csv", index=False)
                for tag, thr in (("t0", 0.0), ("tuned", threshold)):
                    m = exp.summary(d, dec[name][best], thr)
                    for level in LEVELS:
                        rows.append(dict(case=k, scenario=scenario, alpha=best, threshold=threshold,
                                         eval=name, threshold_mode=tag, level=level, **flat(m[level])))
            log(f"case {k} {scenario}: alpha {best:g}, threshold {threshold:.3f}, "
                f"val group BA {table.group_balanced_accuracy.max():.3f} ({time.time() - t0:.0f}s)")

    result = pd.DataFrame(rows)
    result.assign(cm=result.cm.map(json.dumps)).to_csv(args.output / "case_results.csv", index=False)
    pd.concat(alpha_rows).to_csv(args.output / "alpha_by_case.csv", index=False)
    summary = {}
    for (scenario, name, tag, level), part in result.groupby(["scenario", "eval", "threshold_mode", "level"]):
        key = f"{scenario}|{name}|{tag}|{level}"
        summary[key] = {c: dict(mean=float(part[c].mean()), std=float(part[c].std(ddof=1)))
                        for c in ("balanced_accuracy", "recall_bird", "recall_drone", "roc_auc", "macro_f1")}
        summary[key]["confusion_matrix_sum"] = np.sum(np.array(part.cm.tolist()), axis=0).tolist()
        summary[key]["cases"] = part.case.tolist()
    meta = dict(data=str(args.data), alphas=list(alphas), no_test=args.no_test,
                alpha_selection="validation_real group balanced accuracy, tie→smallest",
                threshold_selection="validation_real track balanced accuracy, tie→closest to 0",
                confusion_matrix_order="rows=truth [bird, drone], cols=pred [bird, drone]",
                chosen=result.drop_duplicates(["case", "scenario"])[["case", "scenario", "alpha", "threshold"]]
                .to_dict("records"),
                versions=seq.library_versions(), python=platform.python_version(),
                runtime_s=round(time.time() - started, 1), summary=summary)
    (args.output / "results.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.output / "log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"done in {meta['runtime_s']}s")


if __name__ == "__main__":
    main()
