"""두 5-case 실험 결과(val)를 같은 형식으로 비교해 이슈 기록용 표를 출력한다.

    python -m research.c_part_compare --ref research/output/c_part_cv5_E1 --new research/output/c_part_cv5_E6

판단 규칙(이슈 #52): track balanced accuracy 평균 +0.01 이상, 5개 case 중 4개 이상 상승, 드론 recall 유지.
test 결과는 읽지 않는다 (validation_* 평가만 사용).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = [("balanced_accuracy", "balanced accuracy"), ("recall_bird", "새 recall"),
           ("recall_drone", "드론 recall"), ("roc_auc", "AUC")]


def load(run: Path) -> pd.DataFrame:
    d = pd.read_csv(run / "case_results.csv", keep_default_na=False)
    if d["eval"].str.startswith("test").any():
        raise ValueError("test 결과가 들어 있는 실행은 비교하지 않는다")
    if "scenario" in d and (d.scenario == "B").any():
        d = d[d.scenario == "B"]
    return d


def sel(d, ev="validation_real", tag="t0", level="track"):
    return d[(d["eval"] == ev) & (d.threshold_mode == tag) & (d.level == level)].set_index("case").sort_index()


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def source_family(video: str) -> str:
    v = str(video)
    if v.startswith(("oido", "오이도")):
        return "oido"
    if v.startswith("LOCAL"):
        return "LOCAL"
    if re.fullmatch(r"[0-9a-f]{32}", v):
        return "hash"
    if v.startswith(("드론", "drone")):
        return "드론_/drone_"
    return "기타"


def track_errors(run: Path, sources: pd.DataFrame) -> pd.DataFrame:
    out = []
    scen = "S" if any((run / "pred").glob("case*_S_validation_real.csv")) else "B"
    for f in sorted((run / "pred").glob(f"case*_{scen}_validation_real.csv")):
        p = pd.read_csv(f)
        p["track"] = p.sample_id.str.replace(r":\d+:w\d+$", "", regex=True)
        out.append(p.groupby(["track", "label", "group_id"], as_index=False).decision.mean().rename(columns={"group_id": "group"}))
    t = pd.concat(out)
    t["err"] = np.where(t.decision >= 0, "drone", "bird") != t.label
    t["source"] = [source_family(sources.loc[x].source_video_id) for x in t.track]
    return t.set_index("track")


def paired_bootstrap(ta: pd.DataFrame, tb: pd.DataFrame, draws: int = 2000, seed: int = 20260928) -> tuple:
    """같은 val track 들의 track balanced accuracy 차이(새 - 기준)를 촬영 그룹 단위로 재표집한 95% 구간.

    5개 case 의 val 을 합치면 각 원본 track 이 한 번씩 들어가므로 그룹을 그대로 재표집할 수 있다.
    """
    t = ta[["label", "group"]].assign(a=~ta.err, b=~tb.err.loc[ta.index])
    groups = t.group.unique()
    index = {g: np.flatnonzero(t.group.to_numpy() == g) for g in groups}
    rng = np.random.default_rng(seed)
    label = t.label.to_numpy()

    def ba(correct, lab):
        return np.mean([correct[lab == c].mean() for c in ("bird", "drone")])
    point = ba(t.b.to_numpy(), label) - ba(t.a.to_numpy(), label)
    out = []
    for _ in range(draws):
        idx = np.concatenate([index[g] for g in rng.choice(groups, len(groups))])
        out.append(ba(t.b.to_numpy()[idx], label[idx]) - ba(t.a.to_numpy()[idx], label[idx]))
    lo, hi = np.quantile(out, [.025, .975])
    return point, lo, hi, float(np.mean(np.array(out) > 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("research/output/c_part_cv5_v1"))
    parser.add_argument("--ref-name", default="기준")
    parser.add_argument("--new-name", default="변경")
    args = parser.parse_args()
    a, b = load(args.ref), load(args.new)
    R, N = args.ref_name, args.new_name
    print(f"### 핵심 결과: val 실촬영 (임계값 0, 5개 case 평균 ± 표준편차)\n")
    print(f"| 단위 | 지표 | {R} | {N} | 변화 |\n| --- | --- | --- | --- | --- |")
    for level in ("track", "group"):
        x, y = sel(a, level=level), sel(b, level=level)
        for key, label in METRICS if level == "track" else METRICS[:1]:
            print(f"| {'track' if level == 'track' else '그룹'} | {label} | {fmt(x[key].mean(), x[key].std())} | "
                  f"{fmt(y[key].mean(), y[key].std())} | {y[key].mean() - x[key].mean():+.3f} |")
    x, y = sel(a), sel(b)
    diff = (y.balanced_accuracy - x.balanced_accuracy)
    gx, gy = sel(a, level="group"), sel(b, level="group")
    gdiff = gy.balanced_accuracy - gx.balanced_accuracy
    print("\n#### case별 balanced accuracy 변화\n")
    print("| 단위 | " + " | ".join(f"case {k}" for k in diff.index) + " |\n| --- |" + " --- |" * len(diff))
    print("| track | " + " | ".join(f"{v:+.3f}" for v in diff) + " |")
    print("| 그룹 | " + " | ".join(f"{v:+.3f}" for v in gdiff) + " |")
    up = int((diff > 0).sum())
    drone_ok = y.recall_drone.mean() >= x.recall_drone.mean() - 0.01
    verdict = diff.mean() >= 0.01 and up >= 4 and drone_ok
    print(f"\n판단 기준: 평균 {diff.mean():+.3f} (기준 +0.010), 상승 case {up}/5 (기준 4), "
          f"드론 recall {y.recall_drone.mean() - x.recall_drone.mean():+.3f} → {'통과' if verdict else '미통과'}")
    print("\n#### 혼동행렬 (track, 5개 case 합산, 임계값 0)\n")
    print("| 모델 | 새→새 | 새→드론 | 드론→새 | 드론→드론 |\n| --- | --- | --- | --- | --- |")
    for name, d in ((R, x), (N, y)):
        cm = np.sum([json.loads(c) for c in d.cm], axis=0)
        print(f"| {name} | {cm[0][0]} | {cm[0][1]} | {cm[1][0]} | {cm[1][1]} |")
    print("\n#### 그 밖의 val 지표 (track balanced accuracy)\n")
    print(f"| 평가 | {R} | {N} |\n| --- | --- | --- |")
    rows = [("튜닝 임계값 적용 (val로 고름)", "validation_real", "tuned"),
            ("임계값을 train 내부 OOF로 고름", "validation_real", "oof"),
            ("점검 D: val 증강 세트", "validation_augmented", "t0")]
    for ev in sorted(set(b["eval"]) - {"validation_real", "validation_augmented"}):
        rows.append((f"점검: {ev}", ev, "t0"))
    for label, ev, tag in rows:
        xa, yb = sel(a, ev, tag), sel(b, ev, tag)
        if not len(xa) and not len(yb):
            continue
        left = fmt(xa.balanced_accuracy.mean(), xa.balanced_accuracy.std()) if len(xa) else "-"
        right = fmt(yb.balanced_accuracy.mean(), yb.balanced_accuracy.std()) if len(yb) else "-"
        print(f"| {label} | {left} | {right} |")
    if "alpha" in y:
        print("\n#### 선택된 alpha\n")
        print(f"| case | {R} | {N} |\n| --- | --- | --- |")
        for k in y.index:
            print(f"| {k} | {x.loc[k, 'alpha']} | {y.loc[k, 'alpha']} |")
    sources = pd.read_csv(args.data / "sources.csv", encoding="utf-8-sig").set_index("parent_track_id")
    ta, tb = track_errors(args.ref, sources), track_errors(args.new, sources)
    point, lo, hi, prob = paired_bootstrap(ta, tb)
    print("\n#### 그룹 부트스트랩 (5개 case val 합산 track, 촬영 그룹 2,000회 재표집)\n")
    print(f"track balanced accuracy 차이 {point:+.3f}, 95% 구간 [{lo:+.3f}, {hi:+.3f}], 차이가 0보다 클 비율 {prob:.1%}")
    t = ta[["label", "source"]].assign(ref=ta.err, new=tb.err.loc[ta.index])
    g = t.groupby(["label", "source"]).agg(n=("ref", "size"), ref=("ref", "sum"), new=("new", "sum")).reset_index()
    print("\n#### 영상 출처별 오판 수 (val, track)\n")
    print(f"| 클래스 | 출처 | track 수 | {R} 오판 | {N} 오판 |\n| --- | --- | --- | --- | --- |")
    for r in g.itertuples():
        print(f"| {'새' if r.label == 'bird' else '드론'} | {r.source} | {r.n} | {r.ref} | {r.new} |")


if __name__ == "__main__":
    main()
