"""Shared feature-space policy identifiers for research datasets and runtime B."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from . import FEATURE_VERSION

COORDINATE_POLICY = "fhd_width_1920_v1"
CANONICAL_WIDTH_PX = 1920.0
TRAINING_ASPECT_RATIO = 16.0 / 9.0
ASPECT_RATIO_TOLERANCE = 0.02


def canonical_dimensions(
    processed_width: int,
    processed_height: int,
) -> tuple[float, float, float]:
    if processed_width <= 0 or processed_height <= 0:
        raise ValueError("processed dimensions must be greater than zero")
    scale = CANONICAL_WIDTH_PX / float(processed_width)
    return CANONICAL_WIDTH_PX, float(processed_height) * scale, scale


def matches_training_aspect_ratio(
    processed_width: int,
    processed_height: int,
) -> bool:
    aspect_ratio = float(processed_width) / float(processed_height)
    return (
        abs(aspect_ratio - TRAINING_ASPECT_RATIO) / TRAINING_ASPECT_RATIO
        <= ASPECT_RATIO_TOLERANCE
    )


def timebase_policy(target_fps: float) -> str:
    """Return a policy id that cannot conceal a non-default resampling rate."""
    fps = float(target_fps)
    fps_label = str(int(fps)) if fps.is_integer() else format(fps, "g")
    return f"timestamp_ms_priority_{fps_label}hz_resample_v1"


def feature_contract_descriptor(
    feature_config_id: str,
    *,
    target_fps: float,
) -> dict[str, Any]:
    """Describe formula and adapter policies without sample-specific dimensions."""
    return {
        "feature_version": FEATURE_VERSION,
        "feature_config_id": feature_config_id,
        "coordinate_policy": COORDINATE_POLICY,
        "canonical_width": CANONICAL_WIDTH_PX,
        "training_aspect_ratio": TRAINING_ASPECT_RATIO,
        "aspect_ratio_policy": "warning_only_within_2pct_v1",
        "timebase_policy": timebase_policy(target_fps),
        "target_feature_fps": float(target_fps),
        "vfr_policy": "timestamp_supported_source_pts_not_verified_v1",
    }


def feature_contract_id(feature_config_id: str, *, target_fps: float) -> str:
    payload = feature_contract_descriptor(
        feature_config_id,
        target_fps=target_fps,
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def feature_observation_metadata(
    processed_width: int,
    processed_height: int,
    *,
    feature_config_id: str,
    target_fps: float,
) -> dict[str, Any]:
    canonical_width, canonical_height, scale = canonical_dimensions(
        processed_width,
        processed_height,
    )
    return {
        "coordinate_policy": COORDINATE_POLICY,
        "timebase_policy": timebase_policy(target_fps),
        "feature_contract_id": feature_contract_id(
            feature_config_id,
            target_fps=target_fps,
        ),
        "target_feature_fps": float(target_fps),
        "coordinate_scale": scale,
        "source_processed_width": processed_width,
        "source_processed_height": processed_height,
        "canonical_width": canonical_width,
        "canonical_height": canonical_height,
        "training_aspect_ratio": TRAINING_ASPECT_RATIO,
        "aspect_ratio_matches_training": matches_training_aspect_ratio(
            processed_width,
            processed_height,
        ),
    }
