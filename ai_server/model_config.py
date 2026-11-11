"""Process-start model selection; never mutate this inside a serving process."""
import os

MODEL_NAME = os.environ.get("SKYDETECT_MODEL", "rf").strip().lower()
if MODEL_NAME not in {"rf", "minirocket"}:
    raise ValueError("SKYDETECT_MODEL must be 'rf' or 'minirocket'")
