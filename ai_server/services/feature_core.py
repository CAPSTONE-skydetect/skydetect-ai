"""Core feature extraction stubs for part B."""

from ai_server.schemas import FeatureVector, TrackFeatures, TrackSequence


def extract_core_features(track: TrackSequence) -> FeatureVector:
    """Build the minimum required feature payload for downstream classification."""

    features = TrackFeatures(
        v_mean=0.0,
        v_std=0.0,
        a_mean=0.0,
        heading_change_ratio=0.0,
        maneuverability_sigma=0.0,
    )
    return FeatureVector(
        track_id=track.track_id,
        features=features,
        quality=track.quality,
        feature_status="ok",
    )
