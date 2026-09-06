"""Part C: RF 분류기 학습 스크립트."""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 학습에 사용하는 피처 목록. 이 저장소의 단일 진실 공급원(single source of truth)으로,
# evaluate.py와 classifier.py가 이 목록을 참조한다.
FEATURE_NAMES = [
    "v_mean",
    "v_std",
    "a_mean",
    "turn_rate_mean",
    "turn_rate_p95",
    "heading_change_ratio",
    "straightness",
    "stationary_ratio",
    "bbox_area_mean",
    "bbox_area_cv",
    "bbox_scale_rate_std",
]

# 데이터 버전을 추적하기 위해 모델 번들에 함께 저장하는 컬럼.
_PROVENANCE_COLUMNS = ["simulator_version", "feature_version", "feature_config_id"]

_DEFAULT_MODEL_PATH = "models/rf_classifier.pkl"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_PATH = str(_PROJECT_ROOT / "data" / "train_features.csv")

# 하이퍼파라미터는 train 내부 GroupKFold(family_id) 교차검증으로 선정했다.
# CV f1_macro가 후보 간 0.006 이내(표준편차 ±0.011)로 통계적 차이가 없어,
# 동등한 성능 중 모델 파일이 가장 작은 조합을 택했다. 자세한 근거는
# docs/rf_evaluation.md 참고.
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 5
RANDOM_STATE = 42

# 압축 없이 저장하면 파일이 수십 MB에 달해 저장소 히스토리를 무겁게 만든다.
_COMPRESS_LEVEL = 3


def build_classifier(n_estimators: int = N_ESTIMATORS) -> RandomForestClassifier:
    """학습·평가가 동일한 설정을 쓰도록 분류기 생성을 한 곳에 모은다."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def load_dataset(
    file_path: str = _DEFAULT_DATA_PATH,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """학습 CSV를 읽어 (X, y, 원본 DataFrame)을 반환한다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"학습 데이터셋이 {file_path} 경로에 없습니다. "
            f"B파트 생성 파일을 먼저 배치해주세요."
        )

    names = feature_names or FEATURE_NAMES
    df = pd.read_csv(file_path)

    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV에 다음 피처 컬럼이 없습니다: {missing}\n"
            f"사용 가능한 컬럼: {list(df.columns)}"
        )

    X = df[names].to_numpy()
    y = df["label"].to_numpy()
    return X, y, df


def extract_provenance(df: pd.DataFrame) -> dict[str, str]:
    """데이터셋 버전 정보를 뽑아낸다. 지표의 출처를 재현 가능하게 남기기 위함이다."""
    provenance: dict[str, str] = {}
    for column in _PROVENANCE_COLUMNS:
        if column in df.columns and df[column].nunique() == 1:
            provenance[column] = str(df[column].iloc[0])
    return provenance


def train_and_save(
    output_path: str = _DEFAULT_MODEL_PATH,
    data_path: str = _DEFAULT_DATA_PATH,
    n_estimators: int = N_ESTIMATORS,
) -> None:
    """RF 분류기를 학습하고 pkl 파일로 저장한다."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    X, y, df = load_dataset(data_path)
    provenance = extract_provenance(df)

    clf = build_classifier(n_estimators)
    clf.fit(X, y)

    joblib.dump(
        {
            "model": clf,
            "feature_names": FEATURE_NAMES,
            "provenance": provenance,
            "n_estimators": n_estimators,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "random_state": RANDOM_STATE,
        },
        output_path,
        compress=_COMPRESS_LEVEL,
    )

    print(f"RF 학습 완료 — 학습 샘플 수: {len(y)}, 피처 수: {len(FEATURE_NAMES)}")
    if provenance:
        print(f"데이터 버전: {provenance}")
    print(f"모델 저장: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RF 분류기 학습")
    parser.add_argument("--data-path", default=_DEFAULT_DATA_PATH, help="학습 CSV 경로")
    parser.add_argument("--output-path", default=_DEFAULT_MODEL_PATH, help="모델 저장 경로")
    parser.add_argument(
        "--n-estimators", type=int, default=N_ESTIMATORS, help="트리 개수"
    )
    args = parser.parse_args()

    train_and_save(
        output_path=args.output_path,
        data_path=args.data_path,
        n_estimators=args.n_estimators,
    )
