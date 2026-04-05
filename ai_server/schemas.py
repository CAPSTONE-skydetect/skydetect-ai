from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model for shared contracts.

    All shared payloads reject undeclared fields so A/B/C can detect interface
    drift early during parallel development.
    """

    model_config = ConfigDict(extra="forbid")


StabilizationMethod = Literal["none", "ffmpeg_vidstab", "opencv_ecc"]
TrackStability = Literal["good", "fair", "poor"]
PredictionLabel = Literal["bird", "drone", "uncertain"]


class StabilizationInfo(StrictModel):
    """Metadata about whether global motion compensation was applied."""

    applied: bool = Field(
        default=False,
        description="Whether stabilization was applied before detection/tracking.",
    )
    method: StabilizationMethod = Field(
        default="none",
        description="Stabilization implementation used by A.",
    )


class TrackPoint(StrictModel):
    """One normalized point inside a tracked object history."""

    frame_index: int = Field(
        ...,
        ge=0,
        description="Zero-based frame index from the input video.",
    )
    timestamp_ms: int = Field(
        ...,
        ge=0,
        description="Timestamp of the frame in milliseconds.",
    )
    cx: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized bounding-box center X coordinate.",
    )
    cy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized bounding-box center Y coordinate.",
    )
    w: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized bounding-box width.",
    )
    h: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized bounding-box height.",
    )
    conf: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detector confidence associated with the point.",
    )


class TrackQuality(StrictModel):
    """A-side quality summary attached to a track."""

    num_points: int = Field(
        ...,
        ge=1,
        description="Number of points included in the track history.",
    )
    mean_conf: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean detector confidence across the history.",
    )
    missing_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Observed missing-frame ratio before any interpolation.",
    )
    track_stability: TrackStability = Field(
        ...,
        description="Simple A-side stability label for downstream filtering.",
    )


class TrackSequence(StrictModel):
    """Primary handoff object from A to B.

    A is responsible for creating this object. B should treat `history` as the
    raw tracked sequence and handle interpolation/feature extraction on top of
    it. C does not consume this object directly.
    """

    track_id: int = Field(
        ...,
        ge=0,
        description="Track identifier scoped to one analyzed video.",
    )
    history: list[TrackPoint] = Field(
        ...,
        min_length=1,
        description="Ordered sequence of normalized points for one tracked object.",
    )
    source_video_id: str | None = Field(
        default=None,
        description="Optional external identifier for the source video.",
    )
    stabilization: StabilizationInfo | None = Field(
        default=None,
        description="A-side stabilization metadata proposed for the shared contract.",
    )
    quality: TrackQuality | None = Field(
        default=None,
        description="A-side quality summary proposed for the shared contract.",
    )


class FeatureVector(StrictModel):
    """Feature payload produced by B and consumed by C."""

    track_id: int = Field(..., ge=0, description="Track identifier copied from TrackSequence.")
    features: dict[str, float] = Field(
        ...,
        description="Flat feature dictionary derived from track trajectory and bbox history.",
    )


class PredictionResult(StrictModel):
    """Final classifier output produced by C."""

    track_id: int = Field(..., ge=0, description="Track identifier copied from upstream stages.")
    label: PredictionLabel = Field(
        ...,
        description="Classifier label for the tracked object.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted label.",
    )
    top_features: dict[str, float] = Field(
        ...,
        description="Importance-like summary values exposed in the response payload.",
    )
    quality: dict[str, str | int | float] = Field(
        ...,
        description="Compact quality metadata included with the final prediction.",
    )


class AnalyzeRequest(StrictModel):
    source_video_id: str = Field(..., min_length=1)
    video_path: str = Field(..., min_length=1)
    stabilization_method: StabilizationMethod = "ffmpeg_vidstab"


class AnalyzeResponse(StrictModel):
    source_video_id: str
    tracks: list[TrackSequence]
    message: str
