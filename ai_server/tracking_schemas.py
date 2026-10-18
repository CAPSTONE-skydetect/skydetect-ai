from pydantic import Field, model_validator

from ai_server.schemas import StrictModel


class TrackingTuning(StrictModel):
    klt_accept_conf: float = Field(default=0.40, ge=0.10, le=0.95)
    recovery_conf: float = Field(default=0.58, ge=0.10, le=0.95)
    update_conf: float = Field(default=0.76, ge=0.10, le=0.99)
    search_radius_multiplier: float = Field(default=2.5, ge=1.0, le=8.0)
    exit_confirmation_frames: int = Field(default=3, ge=1, le=120)
    online_update_enabled: bool = False
    appearance_local_refine: bool = False
    appearance_subpixel: bool = False
    contrast_recentering_enabled: bool = True
    contrast_recentering_strength: float = Field(default=0.15, ge=0.0, le=0.5)
    contrast_recentering_max_step_px: float | None = Field(default=None, gt=0.0, le=20.0)
    contrast_correction_alpha: float = Field(default=1.0, gt=0.0, le=1.0)
    contrast_subpixel: bool = False
    contrast_local_centroid: bool = False
    klt_patch_registration: bool = False
    target_scale_klt: bool = False
    motion_lock_audit: bool = False
    motion_audit_min_score: float = Field(default=0.60, ge=0.38, le=0.95)
    reacquisition_confirmation_frames: int = Field(default=1, ge=1, le=10)
    temporal_evidence_frames: int = Field(default=1, ge=1, le=9)
    motion_temporal_alpha: float = Field(default=1.0, gt=0.0, le=1.0)
    cmc_analysis_width: int | None = Field(default=None, ge=320, le=7680)


class ManualTrackingRequest(StrictModel):
    source_video_id: str = Field(..., min_length=1)
    video_path: str = Field(..., min_length=1)
    init_frame_index: int = Field(default=0, ge=0)
    target_bbox: tuple[float, float, float, float]
    max_seconds: float | None = Field(default=None, gt=0)
    stabilize: bool = True
    track_id: int = Field(default=1, ge=0)
    resize_width: int | None = Field(default=1280, ge=320, le=7680)
    write_overlay: bool = True
    tuning: TrackingTuning = Field(default_factory=TrackingTuning)

    @model_validator(mode="after")
    def validate_target_bbox(self) -> "ManualTrackingRequest":
        _, _, width, height = self.target_bbox
        if width <= 0 or height <= 0:
            raise ValueError("target_bbox width and height must be greater than zero")
        return self
