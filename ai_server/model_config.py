"""Process-start model selection; never mutate this inside a serving process."""
import os

MODEL_NAME = os.environ.get("SKYDETECT_MODEL", "rf").strip().lower()
if MODEL_NAME not in {"rf", "minirocket", "multirocket"}:
    raise ValueError("SKYDETECT_MODEL must be 'rf', 'minirocket' or 'multirocket'")

# TrackSequence 를 그대로 받는 시퀀스 모델 (RF 는 요약 특징 벡터를 받는다).
SEQUENCE_MODEL = MODEL_NAME != "rf"
