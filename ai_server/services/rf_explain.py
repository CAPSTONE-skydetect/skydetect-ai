"""Part C: RF 판정 한 건의 근거를 피처별 기여도로 분해한다.

feature_importances_ 는 모델 전체에서 어떤 피처를 많이 썼는지일 뿐, 이 궤적이 왜
드론(또는 새)이 됐는지는 말해주지 않는다. 그래서 이 샘플이 각 트리에서 지나간 경로를
따라가며, 분기마다 드론 확률이 얼마나 바뀌었는지를 그 분기 피처의 몫으로 더한다.

    드론 확률 = 기준값(루트 노드 평균) + Σ 피처별 기여

RandomForest 의 predict_proba 는 트리 확률의 평균이므로 이 분해는 근사가 아니라
정확히 맞아떨어진다 (tests/test_rf_explain.py 에서 확인).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

POSITIVE_CLASS = "drone"


@dataclass(frozen=True)
class RFExplanation:
    base_drone_proba: float
    drone_proba: float
    contributions: dict[str, float]


def explain_rf(
    clf: RandomForestClassifier,
    feature_names: list[str],
    x_row: np.ndarray,
) -> RFExplanation:
    """x_row (1, n_features) 하나의 드론 확률을 기준값 + 피처별 기여로 나눈다."""
    classes = list(clf.classes_)
    positive = classes.index(POSITIVE_CLASS)
    contributions = np.zeros(len(feature_names))
    base = 0.0

    for estimator in clf.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        # 노드 값을 클래스 비율로 맞춘다 (sklearn 버전에 따라 개수 또는 비율로 저장된다).
        proba = values / values.sum(axis=1, keepdims=True)
        # decision_path 의 노드 번호는 깊이 우선으로 매겨져 루트 → 리프 순서와 같다.
        path = estimator.decision_path(x_row).indices
        base += proba[path[0], positive]
        for parent, child in zip(path[:-1], path[1:]):
            contributions[tree.feature[parent]] += proba[child, positive] - proba[parent, positive]

    count = len(clf.estimators_)
    base /= count
    contributions /= count
    return RFExplanation(
        base_drone_proba=float(base),
        drone_proba=float(base + contributions.sum()),
        contributions={
            name: float(value) for name, value in zip(feature_names, contributions)
        },
    )
