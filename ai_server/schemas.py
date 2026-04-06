"""
schemas_v4.py
=============
Sky Detect 프로젝트의 FastAPI 데이터 계약 정의 파일 (v4 주석 강화본)

각 파트가 주고받는 데이터의 형태(타입, 범위, 필수 여부)를 선언합니다.
Java의 DTO + @Valid 인터페이스와 같은 역할입니다.

담당 파트:
  A (정유찬) - 영상 처리  : StabilizationInfo, TrackPoint, TrackQuality, TrackSequence
  B (문형주) - 특징 추출  : TrackFeatures, FeatureVector
  C (강동규) - 분류 & 응답: ClassifyRequest, RuleFilterResult, ResponseQuality,
                           PredictionResult, BatchPredictionResult

[주의]
- 이 파일은 shared contract 용도입니다.
- TrackSequence → FeatureVector → PredictionResult 인터페이스가
  세 파트의 충돌 방지 핵심입니다.
- 선언되지 않은 필드는 허용하지 않습니다. (extra="forbid")
- 현재 A bootstrap server 호환을 위해 AnalyzeRequest / AnalyzeResponse도 유지합니다.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# 공통 베이스 모델
# =============================================================================
class StrictModel(BaseModel):
    """
    shared schema의 공통 부모 모델입니다.

    extra="forbid"를 통해 선언되지 않은 필드가 payload에 들어오면
    조용히 무시하지 않고 바로 validation error를 발생시킵니다.

    목적:
    - 필드 오타 조기 발견
    - 파트별 드리프트 방지
    - 병렬 개발 시 계약 깨짐을 빠르게 잡기
    """

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# 공통 타입 별칭
# =============================================================================
StabilizationMethod = Literal["none", "ffmpeg_vidstab", "opencv_ecc"]
TrackStability = Literal["good", "fair", "poor"]
PredictionLabel = Literal["bird", "drone", "uncertain"]
FeatureStatus = Literal["ok", "partial", "failed"]
RejectReason = Literal[
    "short_track",
    "feature_error",
    "high_noise",
    "low_confidence",
]


# =============================================================================
# [파트 A] 영상 처리 — 담당: 정유찬
# OpenCV + YOLO/ByteTrack 등으로 비행체를 프레임별로 추적한 결과를 담는 스키마
# =============================================================================

class StabilizationInfo(StrictModel):
    """
    영상 손떨림 보정(stabilization) 또는 전역 움직임 보정(global motion compensation)
    적용 여부와 방식을 기록합니다.

    촬영 영상이 흔들렸을 경우 궤적 품질에 영향을 주므로 함께 전달합니다.

    method:
        - "none"           : 보정 없음
        - "ffmpeg_vidstab" : FFmpeg vidstab 필터 사용
        - "opencv_ecc"     : OpenCV ECC 기반 정합 사용
    """

    applied: bool = Field(default=False)
    method: StabilizationMethod = Field(default="none")


class TrackPoint(StrictModel):
    """
    단일 프레임에서 추적된 비행체의 위치 정보입니다.
    좌표(cx, cy, w, h)는 모두 [0.0, 1.0] 범위의 정규화된 값입니다.
    (픽셀 좌표가 아닌 영상 전체 크기 대비 비율)

    더미 데이터 예시:
        {
          "frame_index": 0,
          "timestamp_ms": 0,
          "cx": 0.40, "cy": 0.62,
          "w": 0.03, "h": 0.02,
          "conf": 0.91
        }
    """

    frame_index: int = Field(..., ge=0, description="영상 내 프레임 번호 (0부터 시작)")
    timestamp_ms: int = Field(..., ge=0, description="해당 프레임의 타임스탬프 (밀리초)")
    cx: float = Field(..., ge=0.0, le=1.0, description="바운딩박스 중심 x (정규화)")
    cy: float = Field(..., ge=0.0, le=1.0, description="바운딩박스 중심 y (정규화)")
    w: float = Field(..., ge=0.0, le=1.0, description="바운딩박스 너비 (정규화)")
    h: float = Field(..., ge=0.0, le=1.0, description="바운딩박스 높이 (정규화)")
    conf: float = Field(..., ge=0.0, le=1.0, description="해당 프레임의 탐지 신뢰도")


class TrackQuality(StrictModel):
    """
    전체 트랙(추적 시퀀스)의 품질 지표입니다.
    파트 C의 규칙 기반 필터가 이 값을 직접 참조합니다.

    필터 판단 기준 예시 (classifier.py 기본값 기준):
        num_points    < min_track_length   → short_track
        mean_conf     < min_mean_conf      → low_confidence
        missing_ratio > max_missing_ratio  → high_noise

    A 출력 예시:
        {
          "num_points": 42,
          "mean_conf": 0.89,
          "missing_ratio": 0.02,
          "track_stability": "good"
        }
    """

    num_points: int = Field(..., ge=1, description="유효하게 추적된 프레임 수")
    mean_conf: float = Field(..., ge=0.0, le=1.0, description="전체 프레임 평균 탐지 신뢰도")
    missing_ratio: float = Field(..., ge=0.0, le=1.0, description="추적 실패 프레임 비율")
    # "good" : 분석에 충분한 품질
    # "fair" : 분석 가능하나 신뢰도 주의
    # "poor" : 분석 부적합 → uncertain 처리 권장
    track_stability: TrackStability


class TrackSequence(StrictModel):
    """
    한 비행체에 대한 전체 추적 시퀀스입니다.
    파트 A → 파트 B로 전달되는 핵심 데이터입니다.

    출력 예시:
        {
          "track_id": 101,
          "history": [
            {
              "frame_index": 1, "timestamp_ms": 40,
              "cx": 0.452, "cy": 0.312, "w": 0.050, "h": 0.030, "conf": 0.91
            }
          ],
          "quality": {
            "num_points": 42,
            "mean_conf": 0.89,
            "missing_ratio": 0.02,
            "track_stability": "good"
          }
        }
    """

    track_id: int = Field(..., ge=0)
    history: list[TrackPoint] = Field(..., min_length=1, description="프레임별 위치 목록")
    source_video_id: str | None = None
    stabilization: StabilizationInfo | None = None
    quality: TrackQuality | None = None


# =============================================================================
# [파트 B] 특징 추출 — 담당: 문형주
# TrackSequence의 좌표 시계열에서 수치 특징을 계산한 결과를 담는 스키마
# =============================================================================

class TrackFeatures(StrictModel):
    """
    파트 B → 파트 C shared feature contract 입니다.

    [설계 원칙]
    - Core feature는 MVP 분류에 반드시 필요한 값으로 간주하고 필수로 둡니다.
    - Optional feature는 단계적 개발 / 연구용 확장 항목으로 두고,
      초기 개발 단계에서는 None이어도 허용합니다.
    - 다만 C의 RF 모델이 고정 컬럼을 기대한다면,
      classifier 내부에서 densify(기본값/학습 평균값 대체) 후 사용해야 합니다.

    [참고 논문 소스]
    - 1번: Curvature, Turn-related metrics, Velocity/Accel 계열
    - 2번: BBox Area Stats (날갯짓 / 크기 변화 분석)
    - 3번: Maneuverability Sigma(σ), Heading Change 관련 지표
    """

    # -------------------------------------------------------------------------
    # [필수: Core Features] 즉시 계산 및 MVP 분류의 기초
    # -------------------------------------------------------------------------
    v_mean: float = Field(..., description="평균 속도")
    v_std: float = Field(..., ge=0.0, description="속도 표준편차")
    a_mean: float = Field(..., description="평균 가속도")
    heading_change_ratio: float = Field(..., ge=0.0, le=1.0, description="방향 전환 비율")
    maneuverability_sigma: float = Field(..., ge=0.0, description="기동성 지표 σ")

    # -------------------------------------------------------------------------
    # [선택: Optional / Staged Rollout Features]
    # 현재 단계에서는 None 허용
    # -------------------------------------------------------------------------
    curvature_mean: float | None = Field(default=None, ge=0.0, description="평균 곡률")
    curvature_cv: float | None = Field(default=None, ge=0.0, description="곡률 변동계수")
    turning_angle_mean: float | None = Field(default=None, description="평균 전환각")
    bbox_area_mean: float | None = Field(default=None, ge=0.0, description="BBox 면적 평균")
    bbox_area_std: float | None = Field(default=None, ge=0.0, description="BBox 면적 표준편차")
    glcm_corr: float | None = Field(default=None, description="GLCM 텍스처 상관관계")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "v_mean": 18.5,
                "v_std": 0.22,
                "a_mean": 0.08,
                "heading_change_ratio": 0.14,
                "maneuverability_sigma": 5.82,
                "curvature_mean": 0.07,
            }
        },
    )


class FeatureVector(StrictModel):
    """
    파트 B(문형주) → 파트 C(강동규)로 전달되는 최종 특징 벡터 패키지

    feature_status 의미:
        - "ok"      : core feature 정상 계산 완료
        - "partial" : 일부 값이 보간/수정되었거나 optional metric이 비어 있음
        - "failed"  : feature 계산 자체 실패

    규칙:
        - failed  → features는 반드시 None
        - ok/partial → features는 반드시 존재
        - imputed_fields는 partial일 때만 허용
    """

    track_id: int = Field(..., ge=0)
    features: TrackFeatures | None = None
    quality: TrackQuality | None = None
    feature_status: FeatureStatus = "ok"
    imputed_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_feature_state(self) -> "FeatureVector":
        if self.feature_status == "failed":
            if self.features is not None:
                raise ValueError("features must be None when feature_status='failed'")
            if self.imputed_fields:
                raise ValueError("imputed_fields must be empty when feature_status='failed'")
            return self

        if self.features is None:
            raise ValueError("features are required unless feature_status='failed'")

        if self.feature_status != "partial" and self.imputed_fields:
            raise ValueError("imputed_fields are allowed only when feature_status='partial'")

        return self


# =============================================================================
# [파트 C] 분류 & 응답 — 담당: 강동규
# FeatureVector를 입력받아 규칙 기반 필터 → RF 분류 → 응답 JSON 구성
# =============================================================================

class ClassifyRequest(StrictModel):
    """
    Spring Boot → FastAPI로 분류를 요청할 때의 입력 형식입니다.
    FeatureVector 안에 TrackQuality가 포함되어 있으므로
    C는 여기서 품질 정보를 꺼내 규칙 필터를 적용합니다.

    규칙 필터 임계값은 선택적 override이며,
    미전달 시 classifier.py 내부 기본값을 사용할 수 있습니다.
    """

    feature_vector: FeatureVector
    min_track_length: int | None = Field(default=None, ge=1, description="최소 유효 트랙 길이")
    min_mean_conf: float | None = Field(default=None, ge=0.0, le=1.0, description="최소 평균 탐지 신뢰도")
    max_missing_ratio: float | None = Field(default=None, ge=0.0, le=1.0, description="최대 허용 누락 비율")


class RuleFilterResult(StrictModel):
    """
    규칙 기반 필터 결과입니다.
    RF 추론 전에 트랙 품질과 feature 상태를 검사하여
    분석 가능 여부를 결정합니다.

    판단 예시:
        num_points 부족        → short_track
        mean_conf 부족         → low_confidence
        missing_ratio 초과     → high_noise
        feature 계산 실패      → feature_error

    규칙:
        passed=True  → reject_reason은 반드시 None
        passed=False → reject_reason은 반드시 존재
    """

    passed: bool
    reject_reason: RejectReason | None = None

    @model_validator(mode="after")
    def validate_reject_reason(self) -> "RuleFilterResult":
        if self.passed and self.reject_reason is not None:
            raise ValueError("reject_reason must be None when passed=True")
        if not self.passed and self.reject_reason is None:
            raise ValueError("reject_reason is required when passed=False")
        return self


class ResponseQuality(StrictModel):
    """
    PredictionResult에 포함되는 품질 요약 블록입니다.
    응답에서도 A의 naming과 맞추기 위해 num_points를 그대로 사용합니다.

    응답 예시:
        {
          "num_points": 74,
          "track_stability": "good",
          "feature_status": "ok"
        }
    """

    num_points: int = Field(..., ge=0)
    track_stability: TrackStability
    feature_status: FeatureStatus


class PredictionResult(StrictModel):
    """
    RF 분류 결과 + top feature + 최종 응답입니다.
    FastAPI → Spring Boot로 반환되는 최종 JSON 구조입니다.

    응답 예시:
    {
      "track_id": 101,
      "label": "drone",
      "confidence": 0.86,
      "rule_filter": {"passed": true, "reject_reason": null},
      "top_features": {
        "maneuverability_sigma": 0.31,
        "heading_change_ratio": 0.27,
        "bbox_area_std": 0.18
      },
      "quality": {
        "num_points": 74,
        "track_stability": "good",
        "feature_status": "ok"
      },
      "processing_time_ms": 128
    }

    [주의]
    top_features 값은 RF Gini importance 기준 상대적 기여도입니다.
    "왜 이렇게 분류됐는지"의 완전한 인과 설명이라기보다,
    "모델이 상대적으로 많이 참고한 특징" 정도로 해석합니다.
    """

    track_id: int = Field(..., ge=0)
    label: PredictionLabel
    confidence: float = Field(..., ge=0.0, le=1.0, description="RF 예측 확률 (uncertain 시 0.0)")
    rule_filter: RuleFilterResult
    top_features: dict[str, float] = Field(default_factory=dict, description="RF 중요도 기반 상위 특징")
    quality: ResponseQuality
    processing_time_ms: int | None = Field(default=None, ge=0, description="FastAPI 내부 처리 시간 (ms)")


class BatchPredictionResult(StrictModel):
    """
    여러 트랙을 한 번에 분류할 때의 응답 형식입니다.
    tracks_features.json 처럼 배열 입력을 처리할 때 사용할 수 있습니다.
    """

    results: list[PredictionResult]
    total_count: int = Field(..., ge=0, description="전체 트랙 수")
    drone_count: int = Field(..., ge=0, description="드론으로 분류된 수")
    bird_count: int = Field(..., ge=0, description="새로 분류된 수")
    uncertain_count: int = Field(..., ge=0, description="uncertain 처리된 수")


# =============================================================================
# 현재 A bootstrap server 호환용 모델
# =============================================================================

class AnalyzeRequest(StrictModel):
    """
    현재 A bootstrap server와의 호환을 위해 유지하는 요청 모델입니다.
    """

    source_video_id: str = Field(..., min_length=1)
    video_path: str = Field(..., min_length=1)
    stabilization_method: StabilizationMethod = "ffmpeg_vidstab"


class AnalyzeResponse(StrictModel):
    """
    현재 A bootstrap server와의 호환을 위해 유지하는 응답 모델입니다.
    """

    source_video_id: str
    tracks: list[TrackSequence]
    message: str