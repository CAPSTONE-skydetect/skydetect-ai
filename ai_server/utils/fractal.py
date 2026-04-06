"""Fractal feature helpers for part B research."""

from ai_server.schemas import TrackSequence


def compute_fractal_features(track: TrackSequence) -> dict[str, float | None]:
    """Return placeholder fractal-derived features."""

    _ = track
    return {"curvature_mean": None, "curvature_cv": None}
