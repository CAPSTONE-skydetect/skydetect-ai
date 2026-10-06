import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ai_server.services.rf_explain import explain_rf

FEATURES = ["a", "b", "c"]


@pytest.fixture(scope="module")
def forest():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = np.where(X[:, 0] + 0.5 * X[:, 1] > 0, "drone", "bird")
    return RandomForestClassifier(n_estimators=25, max_depth=5, random_state=0).fit(X, y), X


def test_base_plus_contributions_equals_predict_proba(forest):
    clf, X = forest
    positive = list(clf.classes_).index("drone")
    for row in X[:20]:
        x = row.reshape(1, -1)
        explanation = explain_rf(clf, FEATURES, x)
        expected = clf.predict_proba(x)[0, positive]
        assert explanation.drone_proba == pytest.approx(expected, abs=1e-9)
        assert explanation.base_drone_proba + sum(explanation.contributions.values()) == pytest.approx(
            expected, abs=1e-9
        )


def test_irrelevant_feature_contributes_least(forest):
    clf, _ = forest
    # 라벨은 a, b 로만 정해지므로 c 의 기여는 a 보다 작아야 한다.
    explanation = explain_rf(clf, FEATURES, np.array([[2.0, 0.0, 0.0]]))
    assert explanation.contributions["a"] > 0
    assert abs(explanation.contributions["c"]) < abs(explanation.contributions["a"])


def test_real_model_bundle_decomposes_exactly():
    import joblib

    bundle = joblib.load("models/rf_classifier.pkl")
    clf, names = bundle["model"], bundle["feature_names"]
    x = np.array([[480.0, 0.08, 116.0, 450.0, 0.07, 0.19, 0.70, 1.0, 0.0]])
    explanation = explain_rf(clf, names, x)
    positive = list(clf.classes_).index("drone")
    assert explanation.drone_proba == pytest.approx(clf.predict_proba(x)[0, positive], abs=1e-9)
    assert set(explanation.contributions) == set(names)
