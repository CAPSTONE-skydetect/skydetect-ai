"""Model-dependent portion of the otherwise shared manual-tracking endpoint."""
from ai_server.model_config import MODEL_NAME

if MODEL_NAME == "rf":
    from ai_server.classifiers.rf.prediction import classify_feature_vector
    from ai_server.services.feature_core import build_feature_vector

    def classify_track(track):
        extraction = build_feature_vector(track)
        prediction = classify_feature_vector(extraction.feature_vector)
        features = {
            "status": extraction.feature_vector.feature_status,
            "version": extraction.feature_version,
            "config_id": extraction.feature_config_id,
            "values": (
                extraction.feature_vector.features.model_dump(mode="json")
                if extraction.feature_vector.features else None
            ),
            "reasons": extraction.reasons,
            "quality": extraction.raw_quality,
        }
        return prediction, features
else:
    from importlib import import_module

    from ai_server.classifiers.minirocket.contracts import ClassifyRequest

    classify_track_sequence = import_module(f"ai_server.classifiers.{MODEL_NAME}.prediction").classify_track_sequence

    def classify_track(track):
        return classify_track_sequence(ClassifyRequest(track_sequence=track)), None
