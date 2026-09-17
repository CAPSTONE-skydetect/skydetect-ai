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
        """모델 번들의 feature_names 순서대로 값을 뽑아 입력 행렬을 만든다.

        피처 목록을 여기에 나열하지 않는 이유가 있다. 과거에는 5종을 하드코딩해
        두었는데, 학습 쪽 피처가 바뀌어도 이 함수가 따라가지 않아 추론 시점에야
        "X has 5 features, but RandomForestClassifier is expecting 11" 로 터졌다.
        번들에 저장된 목록을 단일 진실 공급원으로 삼으면 피처가 바뀌어도
        이 함수는 수정할 필요가 없다.
        """
        f = fv.features
        assert f is not None

        missing = [name for name in self._feature_names if not hasattr(f, name)]
        if missing:
            raise ValueError(
                f"모델이 요구하는 피처가 TrackFeatures에 없습니다: {missing}\n"
                f"모델 학습 피처: {self._feature_names}\n"
                "models/rf_classifier.pkl 과 ai_server/schemas.py 의 버전이 어긋났습니다."
            )

        values = [getattr(f, name) for name in self._feature_names]
        if any(value is None for value in values):
            none_fields = [
                name for name, value in zip(self._feature_names, values) if value is None
            ]
            raise ValueError(f"피처 값이 None입니다: {none_fields}")

        return np.array([values], dtype=float)
