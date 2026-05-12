"""Part C: RF 분류기 학습 스크립트."""

import os

import joblib
import numpy as np
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


def _generate_simulation_data(
    n_samples: int = 500,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """bird/drone 시뮬레이션 학습 데이터 생성.

    B파트 TrajectoryGenerator(PR #8) merge 전 임시 구현.
    실제 데이터셋으로 교체 시 이 함수만 수정하면 된다.
    """
    rng = np.random.default_rng(random_state)

    # bird: 빠르고 불규칙한 움직임
    bird = np.column_stack([
        rng.uniform(10.0, 25.0, n_samples),   # v_mean
        rng.uniform(1.5, 5.0, n_samples),     # v_std
        rng.uniform(0.5, 2.0, n_samples),     # a_mean
        rng.uniform(0.3, 0.7, n_samples),     # heading_change_ratio
        rng.uniform(5.0, 20.0, n_samples),    # maneuverability_sigma
    ])

    # drone: 일정하고 안정적인 움직임
    drone = np.column_stack([
        rng.uniform(5.0, 15.0, n_samples),    # v_mean
        rng.uniform(0.1, 1.0, n_samples),     # v_std
        rng.uniform(0.0, 0.5, n_samples),     # a_mean
        rng.uniform(0.0, 0.2, n_samples),     # heading_change_ratio
        rng.uniform(0.0, 5.0, n_samples),     # maneuverability_sigma
    ])

    X = np.vstack([bird, drone])
    y = np.array(["bird"] * n_samples + ["drone"] * n_samples)
    return X, y


def train_and_save(output_path: str = _DEFAULT_MODEL_PATH) -> None:
    """RF 분류기를 학습하고 pkl 파일로 저장한다."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    X, y = _generate_simulation_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"RF 학습 완료 — test accuracy: {accuracy:.4f}")
    print(f"모델 저장: {output_path}")

    joblib.dump({"model": clf, "feature_names": _FEATURE_NAMES}, output_path)


if __name__ == "__main__":
    train_and_save()
