"""Part C: MiniRocket + Ridge 분류기 학습 스크립트.

기본값은 B-3 에서 개발 기준선으로 정한 실제-only 구성이다.

    python -m ai_server.services.train
    python -m ai_server.services.train --arm real_plus_augmentation --output-path models/x.joblib

alpha 는 학습 구성의 train 그룹 3-fold CV 로만 고른다. validation 은 학습·선택에
쓰지 않는다. joblib 은 실행 가능한 직렬화이므로 신뢰하는 출처의 모델만 로드한다.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib

from ai_server.services.sequence_model import (
    ALPHA_SOURCE,
    ARMS,
    DEFAULT_MODEL_PATH,
    DEFAULT_PACKAGE,
    choose_alpha,
    file_hash,
    fit_arm,
    load_package,
)

# 저장 용량을 줄인다. 압축 없이 저장하면 저장소 히스토리가 무거워진다.
_COMPRESS_LEVEL = 3


def train_and_save(
    output_path: str | Path = DEFAULT_MODEL_PATH,
    package: str | Path = DEFAULT_PACKAGE,
    arm: str = "real_only",
    alpha: float | None = None,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}. Choose one of {ARMS}")
    package = Path(package)
    _, train, _ = load_package(package)

    cv_table = None
    if alpha is None:
        alpha, cv_table = choose_alpha(train[ALPHA_SOURCE[arm]])
        print(cv_table.groupby("alpha").group_macro_f1.agg(["mean", "std"]).to_string())

    model = fit_arm(train, arm, alpha)
    model.update(
        created_at=datetime.now(timezone.utc).isoformat(),
        package_manifest_sha256=file_hash(package / "dataset_manifest.json"),
        alpha_selection=(
            f"grouped {len(cv_table.fold.unique())}-fold train CV on {ALPHA_SOURCE[arm]}"
            if cv_table is not None else "fixed by caller"
        ),
        alpha_cv=cv_table.to_dict("records") if cv_table is not None else None,
        validation_used_for_fit=False,
    )

    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    joblib.dump(model, output_path, compress=_COMPRESS_LEVEL)
    print(
        f"MiniRocket 학습 완료 — arm={arm}, alpha={alpha:g}, "
        f"학습 창 {model['train_windows']}개 / 그룹 {model['train_groups']}개"
    )
    print(f"모델 저장: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniRocket + Ridge 분류기 학습")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE, help="B 데이터 패키지 폴더")
    parser.add_argument("--arm", choices=ARMS, default="real_only", help="학습 구성")
    parser.add_argument("--alpha", type=float, default=None, help="지정하면 CV 선택을 건너뜀")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_MODEL_PATH, help="모델 저장 경로")
    args = parser.parse_args()
    train_and_save(args.output_path, args.package, args.arm, args.alpha)
