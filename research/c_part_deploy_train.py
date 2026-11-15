"""배포용 MultiRocket Stack 모델 하나를 5조각 전체 데이터로 학습해 저장한다.

5-case 교차검증(c_part_cv5_stack --final-test)은 "이 학습 방법"의 성능을 잰 것이고,
그때 만든 case별 모델 5개는 저장하지 않았다. 배포에는 같은 방법으로 데이터 전체를 학습한
모델 하나를 쓴다. 보고할 성능은 교차검증 값이며, 이 모델로 데이터셋 점수를 다시 재지 않는다.

평가 때와 같게 두는 것
  멤버 baseline(창 정규화 입력) + framescale(normalization_scale 로 되돌린 입력)
  MultiRocket n_kernels 1,250 → StandardScaler(with_mean=False) → Ridge (그룹 균형 가중치)
  결합기: 멤버 track 평균 점수 위 LogisticRegression(C=1), train 내부 그룹 3-fold OOF 로만 학습
  판정 임계값 0
평가와 다른 것 (미리 정함)
  학습 데이터: case 0~4 의 train_augmented 합집합 = 5조각 실촬영 전체 + 그 증강 사본
  alpha: 5-case val 평균 그룹 balanced accuracy 가 가장 높은 값으로 고정
         (c_part_cv5_FINAL_multirocket/alpha_by_case.csv: baseline 10000, framescale 30000)

    python -m research.c_part_deploy_train --data research/output/c_part_cv5_v1 \
      --output research/output/c_part_deploy_multirocket
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ai_server.services import sequence_model as seq
from . import c_part_cv5_stack as stack
from .minirocket_stack_training import GAIN

BUNDLE_VERSION = "multirocket-dual-scale-stack-1.0.0"
MEMBERS = {"baseline": "X", "framescale": "X_frame"}
ALPHA = {"baseline": 10000.0, "framescale": 30000.0}
N_KERNELS = 1250


def load_union(data_dir: Path, cases) -> dict:
    """case 마다 train_augmented 를 읽어 sample_id 기준으로 합친다.

    원본 창(variant 0)은 case 가 달라도 값이 같아야 한다. 증강 사본은 case 마다 따로 뽑혀 값이 다르므로
    그 조각을 처음 학습에 쓴 case 의 사본만 쓴다. 원본 하나당 증강 개수가 평가 때와 같게 된다.
    """
    parts = [stack.load(data_dir / f"case_{k}", "train_augmented") for k in cases]
    keys = ("X", "X_frame", "y", "group_id", "sample_id", "scale")
    seen: dict[str, int] = {}
    out = {k: [] for k in keys}
    for part in parts:
        for i, sid in enumerate(part["sample_id"]):
            if sid in seen:
                j = seen[sid]
                if out["y"][j] != part["y"][i] or out["group_id"][j] != part["group_id"][i]:
                    raise ValueError(f"Same sample_id with different label/group across cases: {sid}")
                if int(sid.rsplit(":", 2)[1]) == 0 and not np.array_equal(out["X"][j], part["X"][i]):
                    raise ValueError(f"Original window differs across cases: {sid}")
                continue
            seen[sid] = len(out["sample_id"])
            for k in keys:
                out[k].append(part[k][i])
    data = {k: np.asarray(v) for k, v in out.items()}
    data["X"], data["X_frame"] = data["X"].astype(np.float32), data["X_frame"].astype(np.float32)
    return data


def check_coverage(data: dict, data_dir: Path) -> None:
    """합집합이 2초 창이 있는 원본 track 전부(어느 case 의 실촬영 패키지에든 창이 있는 것)를 덮는지 확인한다.

    창을 하나도 못 만드는 짧은 track 은 평가 때도 MiniRocket 데이터에 없었으므로 제외된다.
    """
    windowed = set()
    for path in data_dir.glob("case_*/*_real/minirocket_windows.csv"):
        windowed |= set(pd.read_csv(path, encoding="utf-8-sig").parent_track_id)
    parents = set(pd.Series(data["sample_id"]).str.replace(r":\d+:w\d+$", "", regex=True))
    missing = windowed - parents
    if missing:
        raise ValueError(f"{len(missing)} source tracks missing from training union, e.g. {sorted(missing)[:3]}")
    if set(data["y"]) != set(seq.LABELS):
        raise ValueError("Both labels required")
    tracks = pd.DataFrame(dict(group=data["group_id"], label=data["y"]))
    if (tracks.groupby("group").label.nunique() > 1).any():
        raise ValueError("Mixed-label source group")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--output-path", type=Path,
                        help="학습한 번들을 이 경로에도 복사한다 (예: models/multirocket_stack_classifier.joblib)")
    args = parser.parse_args()
    stack.TRANSFORM, stack.N_KERNELS, stack.CLASSIFIER = "multirocket", N_KERNELS, "ridge"
    args.output.mkdir(parents=True, exist_ok=False)
    started, log_lines = time.time(), []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    data = load_union(args.data, args.cases)
    check_coverage(data, args.data)
    log(f"training union: {len(data['y'])} windows, {len(set(seq.track_ids(data['sample_id'])))} tracks, "
        f"{len(set(data['group_id']))} groups, labels {dict(pd.Series(data['y']).value_counts())}")

    seed = seq.RANDOM_STATE
    members = {}
    for name, key in MEMBERS.items():
        t0 = time.time()
        rocket, scaler, ridges = stack.fit_member(data[key], data["y"], data["group_id"], [ALPHA[name]], seed)
        members[name] = dict(rocket=rocket, scaler=scaler, classifier=ridges[ALPHA[name]], alpha=ALPHA[name],
                             input=key)
        log(f"member {name}: full fit done ({time.time() - t0:.0f}s)")

    # 결합기: 평가 때와 같은 train 내부 그룹 3-fold OOF
    splitter = StratifiedGroupKFold(stack.INNER_FOLDS, shuffle=True, random_state=seq.RANDOM_STATE)
    oof = {name: np.full(len(data["y"]), np.nan) for name in MEMBERS}
    for fold, (fit_idx, held_idx) in enumerate(splitter.split(data["X"], data["y"], data["group_id"])):
        if set(data["group_id"][fit_idx]) & set(data["group_id"][held_idx]):
            raise ValueError("Group leak inside OOF")
        t0 = time.time()
        for name, key in MEMBERS.items():
            rk, sc, rg = stack.fit_member(data[key][fit_idx], data["y"][fit_idx], data["group_id"][fit_idx],
                                          [ALPHA[name]], seed)
            oof[name][held_idx] = stack.decide([(rk, sc, rg[ALPHA[name]])], data[key][held_idx])
        log(f"OOF fold {fold} done ({time.time() - t0:.0f}s)")
    if any(not np.isfinite(v).all() for v in oof.values()):
        raise ValueError("Incomplete OOF scores")
    meta_tracks = stack.track_mean(data, oof)
    z = meta_tracks[list(MEMBERS)].to_numpy()
    meta_scaler = StandardScaler().fit(z)
    weight = seq.balanced_group_weights(meta_tracks.label.to_numpy(), meta_tracks.group.to_numpy())
    head = LogisticRegression(C=1.0, random_state=seq.RANDOM_STATE).fit(
        meta_scaler.transform(z), meta_tracks.label, sample_weight=weight)
    if head.classes_.tolist() != seq.LABELS:
        raise ValueError("Unexpected class order")
    log(f"head coef {dict(zip(MEMBERS, head.coef_[0].round(3).tolist()))}, intercept {head.intercept_[0]:.3f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    manifest = args.data / "manifest.json"
    bundle = dict(
        bundle_version=BUNDLE_VERSION,
        transform="multirocket", n_kernels=N_KERNELS, seed=seed,
        members=members, member_order=list(MEMBERS), meta_scaler=meta_scaler, head=head,
        scale_gain=GAIN, classes=list(seq.LABELS), channels=list(seq.CHANNELS),
        contract_version=seq.CONTRACT_VERSION, contract_id=seq.CONTRACT_ID,
        score_type="stack_logit_window_mean", score_is_probability=False, threshold=0.0,
        alpha_policy="fixed: best mean val group balanced accuracy over 5 CV cases (c_part_cv5_FINAL_multirocket)",
        train_windows=len(data["y"]), train_tracks=int(len(meta_tracks)), train_groups=len(set(data["group_id"])),
        data=str(args.data), data_manifest_sha256=sha256(manifest) if manifest.exists() else None,
        reported_performance="5-case CV test_real track balanced accuracy 0.922 ± 0.031 (threshold 0)",
        held_out_data=None, git_commit=git.stdout.strip() or None,
        versions=seq.library_versions(), python=platform.python_version(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    path = args.output / "multirocket_stack_classifier.joblib"
    joblib.dump(bundle, path)
    meta = {k: v for k, v in bundle.items() if k not in ("members", "meta_scaler", "head")}
    meta.update(member_alpha=ALPHA, head_coef=head.coef_[0].tolist(), head_intercept=float(head.intercept_[0]),
                checkpoint_sha256=sha256(path), runtime_s=round(time.time() - started, 1))
    (args.output / "model_card.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    meta_tracks.to_csv(args.output / "oof_tracks.csv", index=False)
    (args.output / "log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB) in {meta['runtime_s']}s")
    if args.output_path:
        import shutil
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, args.output_path)
        log(f"copied to {args.output_path}")


if __name__ == "__main__":
    main()
