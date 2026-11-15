"""Part C: MultiRocket Stack 분류기 추론 모듈.

멤버 두 개(baseline: 창 정규화 입력, framescale: normalization_scale 로 되돌린 입력)의
MultiRocket → StandardScaler → Ridge 점수를 결합기(StandardScaler → LogisticRegression)에 넣어
창마다 점수를 내고, 창 점수를 평균해 판정한다. 결합기가 선형이라 창 점수 평균은
연구 평가(c_part_cv5_stack)의 track 평균 점수 위 결합기와 같은 값이다.

번들은 research/c_part_deploy_train.py 가 만든다. joblib 은 실행 가능한 직렬화이므로
신뢰하는 출처의 모델만 로드한다.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from ai_server.classifiers.minirocket.classifier import SequencePrediction
from ai_server.classifiers.minirocket.contracts import TrackSequence
from ai_server.services import sequence_model as seq
from research.trajectory_sequence import SequenceConfig, window_track

BUNDLE_VERSION = "multirocket-dual-scale-stack-1.0.0"
DEFAULT_MODEL_PATH = seq.PROJECT_ROOT / "models" / "multirocket_stack_classifier.joblib"


def check_bundle(model: dict) -> None:
    if model.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(f"Unexpected bundle version: {model.get('bundle_version')}")
    if model.get("contract_version") != seq.CONTRACT_VERSION or model.get("contract_id") != seq.CONTRACT_ID:
        raise ValueError("Model/preprocessing contract mismatch")
    if list(model.get("classes", [])) != seq.LABELS or model["head"].classes_.tolist() != seq.LABELS:
        raise ValueError("Unexpected class order; score sign would be wrong")
    if model.get("member_order") != ["baseline", "framescale"]:
        raise ValueError("Unexpected stack members")
    for member in model["members"].values():
        if member["classifier"].classes_.tolist() != seq.LABELS:
            raise ValueError("Unexpected member class order")


def scale_windows(x: np.ndarray, scales: np.ndarray, gain: float) -> np.ndarray:
    """정규화 창을 화면 기준 크기로 되돌린다 (research.minirocket_stack_training.scale_windows 와 같은 식)."""
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (len(x),) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Missing or invalid per-window scales")
    out = x.copy()
    out[:, :2] = (x[:, :2].astype(np.float64) * scales[:, None, None] * gain).astype(np.float32)
    out[:, 2:] = np.diff(out[:, :2], axis=2, prepend=out[:, :2, :1])
    return out


def window_scores(model: dict, x: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """창마다 결합기 점수(logit). 양수면 drone."""
    # window_track 창을 쌓으면 C-연속이 아닌 배열이 나온다. numba 는 메모리 배치마다 따로 컴파일하므로
    # 웜업(연속 배열)과 같은 경로를 타도록 맞춘다. 안 맞추면 첫 실제 요청이 18초쯤 걸린다.
    x = np.ascontiguousarray(seq.validate_windows(x))
    if not len(x):
        return np.empty(0)
    inputs = {"X": x, "X_frame": scale_windows(x, scales, model["scale_gain"])}
    cols = []
    for name in model["member_order"]:
        member = model["members"][name]
        part = inputs[member["input"]]
        z = member["scaler"].transform(member["rocket"].transform(part)).astype(np.float64)
        cols.append(member["classifier"].decision_function(z))
    return model["head"].decision_function(model["meta_scaler"].transform(np.column_stack(cols)))


class MultiRocketStackClassifier:
    """학습된 MultiRocket Stack 번들을 로드하고 TrackSequence 를 분류한다."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {path}\n"
                "먼저 python -m research.c_part_deploy_train 으로 번들을 만드세요."
            )
        self._model = joblib.load(path)
        check_bundle(self._model)
        self._config = SequenceConfig()
        # numba JIT 준비. 첫 요청이 그 비용을 떠안지 않게 미리 돌린다.
        window_scores(self._model, np.zeros((1, 4, self._config.samples), dtype=np.float32), np.array([0.0025]))

    @property
    def model(self) -> dict:
        return self._model

    @property
    def version(self) -> str:
        m = self._model
        alphas = ",".join(f"{n}={m['members'][n]['alpha']:g}" for n in m["member_order"])
        return f"{m['contract_version']}/{m['contract_id']}/multirocket-stack/{alphas}"

    def predict(
        self,
        track: TrackSequence,
        *,
        margin_threshold: float = 0.0,
        min_windows: int = 1,
    ) -> SequencePrediction:
        try:
            windows, rejected = window_track(track.model_dump(mode="json"), self._config)
        except (ValueError, KeyError, TypeError) as exc:
            return SequencePrediction("uncertain", None, "invalid_input", str(exc))

        rejections = dict(Counter(r["reason"] for r in rejected))
        if len(windows) < min_windows:
            return SequencePrediction(
                "uncertain", None, "insufficient_observation",
                f"유효 2초 창 {len(windows)}개 < 최소 {min_windows}개",
                window_rejections=rejections,
            )

        scores = window_scores(
            self._model,
            np.stack([w["X"] for w in windows]),
            np.array([w["normalization_scale"] for w in windows]),
        )
        mean = float(scores.mean())
        result = SequencePrediction(
            "drone" if mean >= 0 else "bird", mean,
            window_scores=[float(s) for s in scores], window_rejections=rejections,
            window_starts_s=[float(w["relative_start_s"]) for w in windows],
        )
        if abs(mean) < margin_threshold:
            result.label = "uncertain"
            result.abstain_reason = "low_separation"
            result.abstain_detail = f"|score| {abs(mean):.3f} < {margin_threshold:g}"
        return result
