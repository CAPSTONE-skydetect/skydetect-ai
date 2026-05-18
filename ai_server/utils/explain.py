"""Part C: feature importance 기반 top_features 계산."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def get_top_features(
    clf: RandomForestClassifier,
    feature_names: list[str],
    n: int = 3,
) -> dict[str, float]:
    """RF 모델의 feature importance 기준 상위 n개를 반환한다.

    반환값은 상위 n개의 importance 비율로 재정규화된 dict이다.
    """
    importances: np.ndarray = clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:n]

    top = {feature_names[i]: float(importances[i]) for i in top_indices}

    total = sum(top.values())
    if total > 0:
        top = {k: round(v / total, 4) for k, v in top.items()}

    return top
