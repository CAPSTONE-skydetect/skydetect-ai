"""MiniRocket Stack 을 5-case 교차검증으로 평가한다 (E1 채택 구성 + E3~E7 옵션).

멤버
  baseline   : 창마다 정규화한 입력 (E0 의 B 와 같은 모델)
  framescale : 같은 창을 normalization_scale 로 되돌려 화면 기준 이동 크기를 보존한 입력 (E1)
  box        : 물체 상자(w, h) 크기·모양 채널 (--box, E4/E5)
결합기: 멤버들의 track 평균 점수 위에 LogisticRegression. train 안의 그룹 단위 3-fold
        교차 예측(OOF)으로만 학습하며 val/test 는 결합기 학습에 쓰지 않는다.
alpha : 멤버마다 val 실촬영 그룹 balanced accuracy 로 고른다 (동률이면 작은 값).
        --alpha-mode per-case 는 case 마다, common 은 5개 case 평균이 가장 높은 하나를 모든 case 에 쓴다 (E6).
        per-case 에서 baseline 은 --reselect-baseline-alpha 가 없으면 E0 에서 고른 값을 그대로 쓴다.
--seeds 를 여러 개 주면 멤버마다 MiniRocket seed 별로 따로 학습해 점수를 평균한다 (E7).
        alpha 선택은 첫 seed 로만 한다.
--box-train-jitter 는 학습 상자 값에만 흔들림을 넣는다 (E5). val 은 깨끗한 값으로 평가하고,
        --box-jitter 로 val 상자에 흔들림을 넣은 점검 세트를 따로 잰다 (선택에는 쓰지 않음).
그 밖의 점검 옵션 (한 번에 하나씩, 값은 미리 정함)
  --n-kernels (E8), --member-classifier logistic (E10), --val-stride (E11, val 추론 때만 창 간격 변경),
  --eval-data (E12, 증강 세기만 바꾼 학습 데이터를 쓸 때 val 세트는 기존 데이터에서 읽음),
  --kinematics (E13, 속도 크기·진행 방향 변화 2채널 추가), --transform multirocket (E14).
  --hflip (E16, 학습에 좌우 반전 사본 추가 + 평가 때 원본·반전 점수 평균), --train-pieces (E17, 학습 곡선),
  --bodyscale (E18, 이동을 물체 크기의 배수로 나타낸 입력의 멤버 추가. 상자는 창 구간 크기 중앙값만 쓴다).
임계값은 세 가지를 기록한다: t0 (0 고정, 대표값), tuned (val 로 고름, 낙관적), oof (train 내부 OOF 로 고름).
test 는 읽지 않는다 (--no-test). 최종 후보가 정해진 뒤 --final-test 로 한 번만 평가한다.

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
from .box_channels import package_box_sizes, package_box_windows
from .minirocket_stack_training import scale_windows
from .plot_confusion import plot

ALPHAS = (1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000)
INNER_FOLDS = 3


def load(case_dir: Path, package: str, sources: pd.DataFrame | None = None, box_jitter: float = 0.0,
         seed: int = 0) -> dict:
    data = cv5.load(case_dir, package)
    if BOX:
        data["X_box"] = package_box_windows(case_dir / package, sources, data["sample_id"],
                                            jitter_px=box_jitter, seed=seed)
    meta = pd.read_csv(case_dir / package / "minirocket_windows.csv", encoding="utf-8-sig").set_index("window_id")
    data["scale"] = meta.loc[data["sample_id"]].normalization_scale.to_numpy()
    if BODYSCALE:
        data["size"] = package_box_sizes(case_dir / package, sources, data["sample_id"])
    return finish(data)


N_KERNELS = seq.N_KERNELS
CLASSIFIER = "ridge"
KINEMATICS = False
TRANSFORM = "minirocket"
HFLIP = False
BOX = False
BODYSCALE = False


def hflip(x: np.ndarray) -> np.ndarray:
    """창을 좌우로 뒤집는다 (E16). 가로 위치·가로 변화량(qx, dx)의 부호만 바꾸고 세로 채널은 그대로 둔다.

    화면을 거울에 비춘 것과 같아서 새/드론 라벨은 바뀌지 않는다. 세로 반전은 중력 방향이 바뀌므로 쓰지 않는다.
    속도·방향 채널(--kinematics)이 붙어 있으면 방향 변화량의 부호도 함께 바꾼다.
    """
    out = x.copy()
    out[:, [0, 2]] *= -1
    if x.shape[1] == 6:
        out[:, 5] *= -1
    return out


def add_kinematics(x: np.ndarray) -> np.ndarray:
    """(N, 4, T) 창 뒤에 속도 크기 |d| 와 진행 방향 변화량(라디안, -pi~pi)을 붙여 (N, 6, T) 로 만든다 (E13).

    둘 다 같은 창의 dx, dy 에서 계산하므로 새 정보가 아니라 MiniRocket(선형 합성곱)이 직접 만들기 어려운
    비선형 조합을 넣어 주는 것이다. 정지(속도 0)에 가까운 칸의 방향 변화는 0 으로 둔다.
    """
    d = x[:, 2:4].astype(np.float64)
    speed = np.hypot(d[:, 0], d[:, 1])
    heading = np.arctan2(d[:, 1], d[:, 0])
    turn = np.diff(heading, axis=1, prepend=heading[:, :1])
    turn = (turn + np.pi) % (2 * np.pi) - np.pi
    moving = (speed > 1e-6) & (np.roll(speed, 1, axis=1) > 1e-6)
    moving[:, 0] = False
    turn = np.where(moving, turn, 0.0)
    return np.concatenate([x, speed[:, None].astype(np.float32), turn[:, None].astype(np.float32)], axis=1)


def check(x: np.ndarray) -> np.ndarray:
    """기본 4채널 창은 서비스 계약 검사(seq.validate_windows)를 그대로 쓰고, 채널을 붙인 창은 앞 4채널만 검사한다."""
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[1] > seq.N_CHANNELS:
        seq.validate_windows(np.ascontiguousarray(x[:, :seq.N_CHANNELS]))
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("Invalid extra channels")
        return x
    return seq.validate_windows(x)


def transform(rocket, scaler, x: np.ndarray) -> np.ndarray:
    return scaler.transform(rocket.transform(check(x))).astype(np.float64)


def body_windows(x: np.ndarray, scale: np.ndarray, size: np.ndarray) -> np.ndarray:
    """이동을 '물체 크기의 몇 배'로 나타낸 창 (E18).

    scale(창의 normalization_scale)과 size(창 구간 상자 크기 중앙값)는 둘 다 화면 가로 길이 대비 값이라
    정규화한 위치 q 에 scale / size 를 곱하면 물체 크기 단위의 위치가 된다. 거리가 달라도 값이 같다
    (멀어지면 화면 속 이동과 상자가 같은 비율로 작아진다). 상수 배율은 MiniRocket 특징에 영향이 없다.
    """
    out = scale_windows(np.ascontiguousarray(x[:, :seq.N_CHANNELS]), scale / size)
    return add_kinematics(out) if KINEMATICS else out


def finish(data: dict) -> dict:
    data["X_frame"] = scale_windows(data["X"], data["scale"])
    if BODYSCALE:
        data["X_body"] = body_windows(data["X"], data["scale"], data["size"])
    if KINEMATICS:
        data["X"], data["X_frame"] = add_kinematics(data["X"]), add_kinematics(data["X_frame"])
    return data


def fit_member(x, y, groups, alphas, seed):
    """seq.fit_representation 과 같은 구성(MiniRocket 10,000 → StandardScaler → Ridge)에 seed 만 바꾼다."""
    from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket
    if TRANSFORM == "multirocket":  # E14: 4가지 pooling + 1차 차분 시계열. n_kernels 1,250 이면 특징 약 9,400개
        rocket = MultiRocket(n_kernels=N_KERNELS, random_state=seed, n_jobs=1)
    else:
        rocket = MiniRocket(n_kernels=N_KERNELS, random_state=seed, n_jobs=1)
    scaler = StandardScaler(with_mean=False).fit(rocket.fit_transform(check(x)))
    z = transform(rocket, scaler, x)
    weight = seq.balanced_group_weights(y, groups)
    if CLASSIFIER == "ridge":
        ridges = {a: RidgeClassifier(alpha=a).fit(z, y, sample_weight=weight) for a in alphas}
    else:  # L2 로지스틱. 규제 강도는 Ridge 와 같은 후보를 C = 1 / alpha 로 쓴다.
        ridges = {a: LogisticRegression(C=1.0 / a, max_iter=3000).fit(z, y, sample_weight=weight) for a in alphas}
    if any(r.classes_.tolist() != seq.LABELS for r in ridges.values()):
        raise ValueError("Unexpected class order")
    return rocket, scaler, ridges


def rewindow(data: dict, data_dir: Path, stride: float) -> dict:
    """val 실촬영 원본 track 을 같은 규칙(2초 창, 30Hz)에 창 간격만 바꿔 다시 자른다."""
    from dataclasses import replace
    from .trajectory_sequence import SequenceConfig, window_track
    sources = pd.read_csv(data_dir / "sources.csv", encoding="utf-8-sig").set_index("parent_track_id")
    config = replace(SequenceConfig(), stride_seconds=stride)
    parents = pd.unique(pd.Series(data["sample_id"]).str.replace(r":\d+:w\d+$", "", regex=True))
    out = {k: [] for k in ("X", "y", "group_id", "sample_id", "variant", "scale")}
    for parent in parents:
        row = sources.loc[parent]
        track = json.loads(Path(row.source_path).read_text(encoding="utf-8"))
        try:
            windows, _ = window_track(track, config)
        except ValueError:
            windows, _ = window_track(track, config, clock_step_tolerance_seconds=0.003)
        for w in windows:
            out["X"].append(w["X"]); out["y"].append(row.label); out["group_id"].append(row.source_group_id)
            out["sample_id"].append(f"{parent}:0:w{w['window_index']}"); out["variant"].append("original")
            out["scale"].append(w["normalization_scale"])
    data = {k: np.asarray(v) for k, v in out.items()}
    data["X"] = data["X"].astype(np.float32)
    return finish(data)


def decide(models, x):
    """seed 별 (rocket, scaler, ridge) 점수의 평균."""
    return np.mean([r.decision_function(transform(rk, sc, x)) for rk, sc, r in models], axis=0)


def decide_eval(models, x):
    """평가 세트용 점수. --hflip 이면 원본 창과 좌우 반전 창의 점수를 평균한다."""
    score = decide(models, x)
    return (score + decide(models, hflip(x))) / 2 if HFLIP else score


def track_mean(data: dict, scores: dict) -> pd.DataFrame:
    table = pd.DataFrame(dict(track=seq.track_ids(data["sample_id"]), group=data["group_id"], label=data["y"], **scores))
    return table.groupby(["track", "group", "label"], as_index=False).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="E0 결과 폴더 (per-case 모드 baseline alpha)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--no-test", action="store_true", help="개선 단계용. val 만 평가한다")
    parser.add_argument("--final-test", action="store_true",
                        help="최종 후보를 정한 뒤 한 번만 쓴다. alpha·head 는 그대로 val·train 으로만 고르고, "
                             "test_real/test_augmented 는 점수만 낸다 (임계값 0 이 대표값)")
    parser.add_argument("--eval-data", type=Path,
                        help="val 세트를 읽을 데이터 폴더 (기본 --data). 증강 세기만 바꾼 학습 데이터(E12)를 쓸 때 "
                             "val 실촬영·val 증강 점검 세트를 기존 데이터에서 그대로 읽기 위해 쓴다")
    parser.add_argument("--keep-variants", type=int, nargs="+",
                        help="train_augmented 에서 남길 변형 번호 (0=원본, 1~9=증강). 생략하면 전부")
    parser.add_argument("--reselect-baseline-alpha", action="store_true",
                        help="per-case 모드에서 baseline alpha 도 val 로 다시 고른다")
    parser.add_argument("--alpha-mode", choices=["per-case", "common"], default="per-case")
    parser.add_argument("--seeds", type=int, nargs="+", default=[seq.RANDOM_STATE])
    parser.add_argument("--n-kernels", type=int, default=seq.N_KERNELS, help="MiniRocket 특징 수 (E8)")
    parser.add_argument("--member-classifier", choices=["ridge", "logistic"], default="ridge",
                        help="멤버 분류기 (E10)")
    parser.add_argument("--val-stride", type=float,
                        help="val 실촬영 창을 원본 track 에서 이 간격(초)으로 다시 만든다. 학습은 그대로 (E11)")
    parser.add_argument("--transform", choices=["minirocket", "multirocket"], default="minirocket",
                        help="특징 변환 (E14). multirocket 은 --n-kernels 1250 으로 특징 수를 MiniRocket 10,000 과 맞춘다")
    parser.add_argument("--kinematics", action="store_true",
                        help="두 멤버 입력에 속도 크기·진행 방향 변화량 2채널을 붙인다 (E13)")
    parser.add_argument("--hflip", action="store_true",
                        help="좌우 반전 대칭 (E16): 학습에 좌우 반전 사본을 더하고, 평가 때 원본·반전 점수를 평균한다")
    parser.add_argument("--train-pieces", type=int, choices=[1, 2, 3],
                        help="학습 곡선 점검 (E17): case 의 train 3조각 중 앞에서부터 이 개수만 학습에 쓴다")
    parser.add_argument("--bodyscale", action="store_true",
                        help="이동을 물체 크기의 배수로 나타낸 입력의 멤버를 추가한다 (E18). 상자는 창 구간 크기 중앙값만 쓴다")
    parser.add_argument("--size-jitter", type=float, nargs="*", default=[1.0, 2.0],
                        help="점검용: val 실촬영 상자에 흔들림(px)을 넣고 크기 중앙값을 다시 구한 세트. 선택에는 쓰지 않음")
    parser.add_argument("--box", action="store_true", help="물체 상자(w, h) 채널 멤버를 추가한다")
    parser.add_argument("--box-train-jitter", type=float, default=0.0,
                        help="학습 상자 크기에만 넣을 흔들림(px, 표준편차)")
    parser.add_argument("--box-jitter", type=float, nargs="*", default=[1.0, 2.0],
                        help="점검용: val 실촬영 상자 크기에 넣을 흔들림(px). 선택에는 쓰지 않음")
    args = parser.parse_args()
    global N_KERNELS, CLASSIFIER, KINEMATICS, TRANSFORM, HFLIP, BOX, BODYSCALE
    N_KERNELS, CLASSIFIER, KINEMATICS, TRANSFORM, HFLIP = (args.n_kernels, args.member_classifier, args.kinematics,
                                                           args.transform, args.hflip)
    BOX, BODYSCALE = args.box, args.bodyscale
    if args.bodyscale and args.val_stride:
        raise SystemExit("--bodyscale 과 --val-stride 는 함께 쓰지 않는다")
    if args.no_test == args.final_test:
        raise SystemExit("--no-test(개선 단계) 와 --final-test(최종 1회) 중 하나만 붙일 것")
    if args.hflip and args.box:
        raise SystemExit("--hflip 과 --box 는 함께 쓰지 않는다")
    pieces = pd.read_csv(args.data / "sources.csv", encoding="utf-8-sig").set_index("parent_track_id").piece
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "pred").mkdir()
    sources = (pd.read_csv(args.data / "sources.csv", encoding="utf-8-sig").set_index("parent_track_id")
               if args.box or args.bodyscale else None)
    evals = ["validation_real", "validation_augmented"] + (["test_real", "test_augmented"] if args.final_test else [])
    base_rows = pd.read_csv(args.baseline / "case_results.csv", keep_default_na=False)
    base_alpha = base_rows[base_rows.scenario == "B"].drop_duplicates("case").set_index("case").alpha.to_dict()
    members = {"baseline": "X", "framescale": "X_frame", **({"bodyscale": "X_body"} if args.bodyscale else {}),
               **({"box": "X_box"} if args.box else {})}
    seed0 = args.seeds[0]
    started, log_lines, rows, alpha_rows = time.time(), [], [], []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    def searched(name):
        return args.alpha_mode == "common" or name != "baseline" or args.reselect_baseline_alpha

    # 1단계: case 마다 데이터를 읽고, 첫 seed 로 멤버 전체 모델을 alpha 후보에 대해 학습
    state = {}
    for k in args.cases:
        t0 = time.time()
        case_dir = args.data / f"case_{k}"
        train = load(case_dir, "train_augmented", sources, args.box_train_jitter, seed=1000 + k)
        if args.keep_variants is not None:
            index = np.array([int(sid.rsplit(":", 2)[1]) for sid in train["sample_id"]])
            keep = np.isin(index, args.keep_variants)
            train = {key: value[keep] for key, value in train.items()}
            log(f"case {k}: kept variants {sorted(args.keep_variants)} -> {keep.sum()} of {len(keep)} train windows")
        if args.train_pieces is not None:
            use = [(k + 1 + i) % 5 for i in range(args.train_pieces)]
            parent = pd.Series(train["sample_id"]).str.replace(r":\d+:w\d+$", "", regex=True)
            keep = pieces.loc[parent].isin(use).to_numpy()
            train = {key: value[keep] for key, value in train.items()}
            log(f"case {k}: train pieces {use} -> {keep.sum()} of {len(keep)} train windows, "
                f"{len(set(train['group_id']))} groups")
        if args.hflip:  # 좌우 반전 사본. sample_id 를 그대로 두어 OOF track 점수가 원본·반전 평균이 되게 한다
            n = len(train["y"])
            train = {key: np.concatenate([value, hflip(value) if key in ("X", "X_frame", "X_body") else value])
                     for key, value in train.items()}
            log(f"case {k}: hflip copies added -> {len(train['y'])} train windows (from {n})")
        eval_dir = (args.eval_data or args.data) / f"case_{k}"
        sets = {name: load(eval_dir, name, sources) for name in evals}
        if args.val_stride:
            sets["validation_real"] = rewindow(sets["validation_real"], args.data, args.val_stride)
            log(f"case {k}: validation_real re-windowed at {args.val_stride}s -> {len(sets['validation_real']['y'])} windows")
        if args.bodyscale:
            for j in args.size_jitter:
                jittered = dict(sets["validation_real"])
                size = package_box_sizes(eval_dir / "validation_real", sources, jittered["sample_id"],
                                         jitter_px=j, seed=k)
                jittered["X_body"] = body_windows(jittered["X"], jittered["scale"], size)
                sets[f"validation_real_sizejitter{j:g}"] = jittered
        if args.box:
            for j in args.box_jitter:
                jittered = dict(sets["validation_real"])
                jittered["X_box"] = package_box_windows(case_dir / "validation_real", sources,
                                                        jittered["sample_id"], jitter_px=j, seed=k)
                sets[f"validation_real_boxjitter{j:g}"] = jittered
        for name, d in sets.items():
            if set(d["group_id"]) & set(train["group_id"]):
                raise ValueError(f"case {k}: group leak {name}")
        if args.final_test and set(sets["validation_real"]["group_id"]) & set(sets["test_real"]["group_id"]):
            raise ValueError(f"case {k}: validation/test group leak")
        full = {}
        for name, key in members.items():
            alphas = ALPHAS if searched(name) else [float(base_alpha[k])]
            rocket, scaler, ridges = fit_member(train[key], train["y"], train["group_id"], alphas, seed0)
            full[name] = (rocket, scaler, ridges)
            z = transform(rocket, scaler, sets["validation_real"][key])
            z_flip = transform(rocket, scaler, hflip(sets["validation_real"][key])) if HFLIP else None
            for a, r in ridges.items():
                score = r.decision_function(z)
                if z_flip is not None:
                    score = (score + r.decision_function(z_flip)) / 2
                m = exp.summary(sets["validation_real"], score)
                alpha_rows.append(dict(case=k, member=name, alpha=a,
                                       group_balanced_accuracy=m["group"]["balanced_accuracy"],
                                       track_balanced_accuracy=m["track"]["balanced_accuracy"]))
        state[k] = dict(train=train, sets=sets, full=full)
        log(f"case {k}: full-member fits done ({time.time() - t0:.0f}s)")

    # 2단계: alpha 선택 (val 그룹 balanced accuracy, 동률이면 작은 값)
    table = pd.DataFrame(alpha_rows)
    table.to_csv(args.output / "alpha_by_case.csv", index=False)
    chosen = {}
    for name in members:
        part = table[table.member == name]
        if args.alpha_mode == "common":
            mean = part.groupby("alpha").group_balanced_accuracy.mean().reset_index()
            best = float(mean.sort_values(["group_balanced_accuracy", "alpha"], ascending=[False, True]).iloc[0].alpha)
            chosen.update({(k, name): best for k in args.cases})
        else:
            for k, g in part.groupby("case"):
                chosen[(k, name)] = float(g.sort_values(["group_balanced_accuracy", "alpha"],
                                                        ascending=[False, True]).iloc[0].alpha)
    log(f"chosen alpha: { {f'{k}:{n}': a for (k, n), a in chosen.items()} }")

    # 3단계: seed 추가 학습, OOF 결합기, 평가
    for k in args.cases:
        t0 = time.time()
        train, sets, full = state[k]["train"], state[k]["sets"], state[k]["full"]
        alpha = {name: chosen[(k, name)] for name in members}
        fitted = {}
        for name, key in members.items():
            rocket, scaler, ridges = full[name]
            models = [(rocket, scaler, ridges[alpha[name]])]
            for seed in args.seeds[1:]:
                rk, sc, rg = fit_member(train[key], train["y"], train["group_id"], [alpha[name]], seed)
                models.append((rk, sc, rg[alpha[name]]))
            fitted[name] = models
        splitter = StratifiedGroupKFold(INNER_FOLDS, shuffle=True, random_state=seq.RANDOM_STATE)
        oof = {name: np.full(len(train["y"]), np.nan) for name in members}
        for fit_idx, held_idx in splitter.split(train["X"], train["y"], train["group_id"]):
            for name, key in members.items():
                models = []
                for seed in args.seeds:
                    rk, sc, rg = fit_member(train[key][fit_idx], train["y"][fit_idx], train["group_id"][fit_idx],
                                            [alpha[name]], seed)
                    models.append((rk, sc, rg[alpha[name]]))
                oof[name][held_idx] = decide(models, train[key][held_idx])
        meta_tracks = track_mean(train, oof)
        z = meta_tracks[list(members)].to_numpy()
        meta_scaler = StandardScaler().fit(z)
        weight = seq.balanced_group_weights(meta_tracks.label.to_numpy(), meta_tracks.group.to_numpy())
        head = LogisticRegression(C=1.0, random_state=seq.RANDOM_STATE).fit(
            meta_scaler.transform(z), meta_tracks.label, sample_weight=weight)
        if head.classes_.tolist() != seq.LABELS:
            raise ValueError("Unexpected class order")
        log(f"case {k}: alphas {alpha}, head coef {dict(zip(members, head.coef_[0].round(3).tolist()))}")

        dec = {}
        for name, d in sets.items():
            cols = np.column_stack([decide_eval(fitted[n], d[key]) for n, key in members.items()])
            dec[name] = head.decision_function(meta_scaler.transform(cols))
            pd.DataFrame(dict(sample_id=d["sample_id"], group_id=d["group_id"], label=d["y"], variant=d["variant"],
                              decision=dec[name], **{n: cols[:, i] for i, n in enumerate(members)})).to_csv(
                args.output / "pred" / f"case{k}_S_{name}.csv", index=False)
        threshold = exp.best_threshold(exp.tables(sets["validation_real"], dec["validation_real"])["track"])
        # 점검용: 임계값을 val 이 아닌 train 내부 OOF track 점수에서 고른 값 (val 을 전혀 쓰지 않음)
        oof_dec = head.decision_function(meta_scaler.transform(z))
        oof_table = meta_tracks[["track", "group", "label"]].assign(decision=oof_dec).rename(
            columns={"track": "track_id", "group": "group_id"})
        threshold_oof = exp.best_threshold(oof_table)
        for name, d in sets.items():
            for tag, thr in (("t0", 0.0), ("tuned", threshold), ("oof", threshold_oof)):
                m = exp.summary(d, dec[name], thr)
                for level in cv5.LEVELS:
                    rows.append(dict(case=k, scenario="S", alpha=json.dumps(alpha), threshold=thr, eval=name,
                                     threshold_mode=tag, level=level, **cv5.flat(m[level])))
        log(f"case {k}: threshold val {threshold:.3f}, oof {threshold_oof:.3f}; done ({time.time() - t0:.0f}s)")

    result = pd.DataFrame(rows)
    result.assign(cm=result.cm.map(json.dumps)).to_csv(args.output / "case_results.csv", index=False)
    summary = {}
    for (name, tag, level), part in result.groupby(["eval", "threshold_mode", "level"]):
        key = f"S|{name}|{tag}|{level}"
        summary[key] = {c: dict(mean=float(part[c].mean()), std=float(part[c].std(ddof=1)))
                        for c in ("balanced_accuracy", "recall_bird", "recall_drone", "roc_auc", "macro_f1")}
        summary[key]["confusion_matrix_sum"] = np.sum(np.array(part.cm.tolist()), axis=0).tolist()
    plot(summary["S|validation_real|t0|track"]["confusion_matrix_sum"],
         f"{args.output.name} (track, val, {len(args.cases)} cases pooled)",
         args.output / "confusion_S_val_track.png")
    meta = dict(data=str(args.data), eval_data=str(args.eval_data or args.data), baseline=str(args.baseline), alphas=list(ALPHAS), no_test=args.no_test, final_test=args.final_test,
                members=list(members), alpha_mode=args.alpha_mode, seeds=args.seeds, n_kernels=args.n_kernels,
                member_classifier=args.member_classifier, val_stride=args.val_stride, kinematics=args.kinematics, transform=args.transform,
                hflip=args.hflip, train_pieces=args.train_pieces,
                bodyscale=args.bodyscale, size_jitter=args.size_jitter if args.bodyscale else None,
                keep_variants=args.keep_variants, reselect_baseline_alpha=args.reselect_baseline_alpha,
                box=args.box, box_train_jitter=args.box_train_jitter, box_jitter=args.box_jitter if args.box else None,
                inner_folds=INNER_FOLDS, versions=seq.library_versions(), python=platform.python_version(),
                runtime_s=round(time.time() - started, 1), summary=summary)
    (args.output / "results.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.output / "log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"done in {meta['runtime_s']}s")


if __name__ == "__main__":
    main()
