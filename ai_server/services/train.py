"""Part C: RF 분류기 학습 스크립트."""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

_FEATURE_NAMES = [
    "v_mean",
    "v_std",
    "a_mean",
    "heading_change_ratio",
    "maneuverability_sigma",
]

_DEFAULT_MODEL_PATH = "models/rf_classifier.pkl"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_PATH = str(_PROJECT_ROOT / "data" / "simulation_features.csv")


def _generate_simulation_data(
    file_path: str = _DEFAULT_DATA_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """bird/drone 시뮬레이션 결과에서 추출된 실제 특징 데이터셋 로드.

    B파트 Phase 2 추출 완료 데이터 적용.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"실제 시뮬레이션 데이터셋이 {file_path} 경로에 없습니다. "
            f"B파트 생성 파일을 먼저 배치해주세요."
        )

    df = pd.read_csv(file_path)
    X = df[_FEATURE_NAMES].to_numpy()
    y = df["label"].to_numpy()
    return X, y


def train_and_save(
    output_path: str = _DEFAULT_MODEL_PATH,
    data_path: str = _DEFAULT_DATA_PATH,
) -> None:
    """RF 분류기를 학습하고 pkl 파일로 저장한다."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    X, y = _generate_simulation_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train) # 학습

    accuracy = clf.score(X_test, y_test)
    print(f"RF 학습 완료 — test accuracy: {accuracy:.4f}")
    print(f"모델 저장: {output_path}")

    joblib.dump({"model": clf, "feature_names": _FEATURE_NAMES}, output_path)


if __name__ == "__main__":
    train_and_save()
