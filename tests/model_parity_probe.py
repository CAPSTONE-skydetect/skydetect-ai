"""Read-only compatibility probe, runnable on frozen or unified source trees."""
import hashlib
import json
import math
import os
from pathlib import Path

from ai_server.schemas import ClassifyRequest, TrackSequence


def make_track(count=150, *, drop=(), stabilized=True, fps=30, conf=.9):
    return {
        "track_id": 7, "source_video_id": "unification-fixture",
        "processed_width": 1280, "processed_height": 720,
        "history": [dict(frame_index=i, timestamp_ms=round(i * 1000 / fps),
                         cx=.3 + .025 * i / fps, cy=.5 + .02 * math.sin(4 * i / fps),
                         w=.02, h=.02, conf=conf)
                    for i in range(count) if i not in drop],
        "stabilization": {"applied": stabilized,
                          "method": "opencv_feature_cmc" if stabilized else "none"},
    }


def probe():
    model = os.environ.get("SKYDETECT_MODEL", "rf")
    from ai_server.main import create_app
    from fastapi.testclient import TestClient
    cases = {
        "smooth": make_track(), "short": make_track(30),
        "tiny": make_track(4), "unstabilized": make_track(stabilized=False),
        "long_gap": make_track(180, drop=range(60, 75)),
        "low_conf": make_track(conf=.1), "fps60": make_track(300, fps=60),
    }
    frozen_inputs = Path(__file__).parent / "fixtures/model_parity/inputs.json"
    if frozen_inputs.exists():
        cases = json.loads(frozen_inputs.read_text())
    result = {"model": model, "cases": {}}
    with TestClient(create_app()) as client:
        for name, data in cases.items():
            track = TrackSequence.model_validate(data)
            features = None
            if model == "rf":
                from ai_server.services.feature_core import build_feature_vector
                extraction = build_feature_vector(track)
                features = extraction.feature_vector.model_dump(mode="json")
                payload = ClassifyRequest(feature_vector=extraction.feature_vector)
            else:
                payload = ClassifyRequest(track_sequence=track)
            response = client.post("/classify", json=payload.model_dump(mode="json"))
            body = response.json()
            if response.status_code != 200:
                raise AssertionError((name, response.status_code, body))
            body.pop("processing_time_ms", None)
            result["cases"][name] = {"response": body, "features": features}
        schema = client.get("/openapi.json").json()
        result["classify_request_schema"] = schema["components"]["schemas"]["ClassifyRequest"]
        result["prediction_schema"] = schema["components"]["schemas"]["PredictionResult"]
        result["paths"] = sorted(schema["paths"])
    result["fixtures_sha256"] = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    return result


if __name__ == "__main__":
    print(json.dumps(probe(), sort_keys=True, allow_nan=False))
