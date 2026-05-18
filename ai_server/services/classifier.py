"""Part C: RF 분류기 추론 모듈."""

from pathlib import Path

import joblib
import numpy as np

from ai_server.schemas import FeatureVector, PredictionLabel

_DEFAULT_MODEL_PATH = "models/rf_classifier.pkl"
_CONFIDENCE_THRESHOLD = 0.6


class RFClassifier:
    """학습된 RandomForest 모델을 로드하고 FeatureVector를 분류한다."""

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {model_path}\n"
                "먼저 python -m ai_server.services.train 을 실행하세요."
            )
        bundle = joblib.load(path)
        self._clf = bundle["model"]
        self._feature_names: list[str] = bundle["feature_names"]

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def clf(self):
        return self._clf

    def predict(self, fv: FeatureVector) -> tuple[PredictionLabel, float]:
        """FeatureVector를 받아 (label, confidence)를 반환한다.

        confidence가 threshold 미만이면 "uncertain"을 반환한다.
        """
        X = self._to_array(fv)
        proba = self._clf.predict_proba(X)[0]
        classes: list[str] = list(self._clf.classes_)

        best_idx = int(np.argmax(proba))
        confidence = float(proba[best_idx])
        label = classes[best_idx]

        if confidence < _CONFIDENCE_THRESHOLD:
            return "uncertain", confidence

        return label, confidence  # type: ignore[return-value]

    def _to_array(self, fv: FeatureVector) -> np.ndarray:
        f = fv.features
        assert f is not None
        values = [
            f.v_mean,
            f.v_std,
            f.a_mean,
            f.heading_change_ratio,
            f.maneuverability_sigma,
        ]
        return np.array([values])
