"""Extended feature extraction stubs for part B research features."""

from ai_server.schemas import TrackSequence


def extract_signal_features(track: TrackSequence) -> dict[str, float | None]:
    """Return optional research features until real extraction is implemented."""

    _ = track
    return {
        "curvature_mean": None,
        "curvature_cv": None,
        "turning_angle_mean": None,
        "bbox_area_mean": None,
        "bbox_area_std": None,
        "glcm_corr": None,
    }
