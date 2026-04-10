"""
schemas_v4.py
=============
Sky Detect 프로젝트의 FastAPI 데이터 계약 정의 파일 (v4 주석 강화본)

역할:
- A(영상 처리) → B(특징 추출) → C(분류/응답) 사이의 shared contract를 정의한다.
- 파트별 데이터 구조를 명확히 고정해 병렬 개발 시 충돌을 줄인다.
- 현재 A bootstrap server와의 호환도 유지한다.

설계 원칙:
1. 선언되지 않은 필드는 허용하지 않는다. (extra="forbid")
2. 같은 의미의 값은 가능한 한 같은 이름을 유지한다. (예: num_points)
3. 상태 조합이 모순되면 조용히 통과시키지 않고 validation error를 낸다.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# 공통 베이스 모델
# =============================================================================
class StrictModel(BaseModel):
    """
    모든 shared schema의 공통 부모 모델.

    한 줄 설명:
        선언되지 않은 필드(extra)를 거부하는 엄격한 BaseModel.

    왜 이렇게 했는가:
        A/B/C가 병렬로 개발할 때 가장 흔한 문제는 필드 이름 오타, 임의 필드 추가,
        버전 드리프트다. extra="forbid"를 걸어두면 이런 실수를 초기에 바로 잡을 수 있다.
    """

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# 공통 타입 별칭
# =============================================================================
# 한 줄 설명:
#   반복해서 쓰는 Literal 타입을 별칭으로 빼서 가독성과 일관성을 높인다.
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
    영상 안정화/전역 움직임 보정 적용 여부를 담는 모델.

    한 줄 설명:
        입력 영상에 stabilization이 적용되었는지와 방법을 기록한다.

    왜 이렇게 했는가:
        손떨림/카메라 움직임이 trajectory 품질에 영향을 주므로,
        후속 단계(B/C)가 이 메타데이터를 함께 볼 수 있어야 한다.

    필드:
        applied (bool): 보정 적용 여부
        method (StabilizationMethod): 사용한 보정 방식
    """

    applied: bool = Field(default=False)
    method: StabilizationMethod = Field(default="none")


class TrackPoint(StrictModel):
    """
    단일 프레임에서 추적된 객체의 위치 정보를 담는 모델.

    한 줄 설명:
        한 프레임 안에서의 bbox 중심/크기/신뢰도를 표현한다.

    왜 이렇게 했는가:
        B 파트는 결국 이 시계열(history)로부터 속도, 가속도, heading 변화 등
        trajectory feature를 계산하므로, 프레임 단위 좌표 구조가 명확해야 한다.

    필드:
        frame_index (int): 프레임 번호, 0 이상
        timestamp_ms (int): 해당 프레임 시각(ms), 0 이상
        cx (float): bbox 중심 x, 0~1 정규화
        cy (float): bbox 중심 y, 0~1 정규화
        w (float): bbox 너비, 0~1 정규화
        h (float): bbox 높이, 0~1 정규화
        conf (float): 탐지 신뢰도, 0~1
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
    전체 track의 품질 요약 정보를 담는 모델.

    한 줄 설명:
        A가 계산한 track 품질 지표를 B/C에 전달한다.

    왜 이렇게 했는가:
        C의 규칙 기반 필터는 모델 추론 전에 track이 분석 가능한 수준인지
        먼저 판단해야 한다. 그래서 num_points / mean_conf / missing_ratio를
        구조적으로 받도록 했다.

    필드:
        num_points (int): 유효 추적 프레임 수
        mean_conf (float): 전체 프레임 평균 탐지 신뢰도
        missing_ratio (float): 추적 실패 프레임 비율
        track_stability (TrackStability): track 품질 등급
    """

    num_points: int = Field(..., ge=1, description="유효하게 추적된 프레임 수")
    mean_conf: float = Field(..., ge=0.0, le=1.0, description="전체 프레임 평균 탐지 신뢰도")
    missing_ratio: float = Field(..., ge=0.0, le=1.0, description="추적 실패 프레임 비율")
    track_stability: TrackStability


class TrackSequence(StrictModel):
    """
    하나의 비행체에 대한 전체 추적 시퀀스를 담는 모델.

    한 줄 설명:
        A → B로 넘기는 핵심 handoff 객체.

    왜 이렇게 했는가:
        B는 이 객체의 history와 quality를 기반으로 feature를 계산한다.
        따라서 track_id, history, quality를 하나의 명시적 패키지로 묶는 것이 안전하다.

    필드:
        track_id (int): track 식별자
        history (list[TrackPoint]): 프레임별 좌표/크기 목록
        source_video_id (str | None): 원본 영상 식별자
        stabilization (StabilizationInfo | None): 영상 보정 정보
        quality (TrackQuality | None): track 품질 요약
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
    B가 계산한 feature 묶음을 담는 모델.

    한 줄 설명:
        RF 분류 전 단계에서 사용하는 trajectory/bbox 기반 feature 벡터.

    왜 이렇게 했는가:
        - Core feature는 MVP 단계에서 항상 계산 가능해야 하므로 required로 둔다.
        - 연구/확장 단계 feature는 아직 미구현일 수 있으므로 Optional로 둔다.
        - 이렇게 하면 B 개발 현실을 반영하면서도 shared schema를 유지할 수 있다.

    주의:
        Optional feature가 None일 수 있으므로,
        C에서 RF 입력 전에 densify(기본값/학습 평균값 대체)를 해야 할 수 있다.

    필드:
        [Core / required]
            v_mean: 평균 속도
            v_std: 속도 표준편차
            a_mean: 평균 가속도
            heading_change_ratio: 방향 전환 비율
            maneuverability_sigma: 기동성 지표 σ

        [Optional / staged rollout]
            curvature_mean: 평균 곡률
            curvature_cv: 곡률 변동계수
            turning_angle_mean: 평균 전환각
            bbox_area_mean: bbox 면적 평균
            bbox_area_std: bbox 면적 표준편차
            glcm_corr: 텍스처 상관관계
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
    B → C로 전달되는 최종 feature 패키지.

    한 줄 설명:
        특징값 + 품질 정보 + feature 계산 상태를 함께 넘기는 handoff 모델.

    왜 이렇게 했는가:
        단순히 features만 넘기면,
        C는 "정상 계산인지 / 일부 보간했는지 / 완전 실패인지"를 알 수 없다.
        그래서 feature_status와 imputed_fields를 같이 넣어
        모델 추론 전 rule-based filtering이나 fallback 판단이 가능하도록 했다.

    필드:
        track_id (int): 대응되는 track 식별자
        features (TrackFeatures | None): 계산된 feature들
        quality (TrackQuality | None): A가 만든 품질 정보
        feature_status (FeatureStatus): 계산 상태 ("ok" / "partial" / "failed")
        imputed_fields (list[str]): 보간/수정된 필드 목록

    상태 규칙:
        - failed  → features는 반드시 None
        - ok      → features는 반드시 존재
        - partial → features는 존재해야 하며, 보간한 필드가 있으면 imputed_fields에 기록
    """

    track_id: int = Field(..., ge=0)
    features: TrackFeatures | None = None
    quality: TrackQuality | None = None
    feature_status: FeatureStatus = "ok"
    imputed_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_feature_state(self) -> "FeatureVector":
        """
        FeatureVector 내부 상태 조합의 일관성을 검증한다.

        한 줄 설명:
            feature_status / features / imputed_fields 간 모순을 막는다.

        왜 이렇게 했는가:
            "failed인데 features가 있음", "ok인데 features가 없음" 같은 payload는
            시스템 해석을 애매하게 만든다. 이런 모순은 shared schema 단계에서
            바로 막는 것이 가장 안전하다.

        Returns:
            FeatureVector: 검증을 통과한 자기 자신

        Raises:
            ValueError: 상태 조합이 계약 규칙에 맞지 않을 때
        """
        if self.feature_status == "failed":
            if self.features is not None:
                raise ValueError("features must be None when feature_status='failed'")
            if self.imputed_fields:
                raise ValueError("imputed_fields must be empty when feature_status='failed'")
            return self

        # ok / partial 인 경우에는 최소한 feature 묶음은 있어야 한다.
        if self.features is None:
            raise ValueError("features are required unless feature_status='failed'")

        # imputed_fields는 partial일 때만 의미가 있다.
        if self.feature_status != "partial" and self.imputed_fields:
            raise ValueError("imputed_fields are allowed only when feature_status='partial'")

        return self


# =============================================================================
# [파트 C] 분류 & 응답 — 담당: 강동규
# FeatureVector를 입력받아 규칙 기반 필터 → RF 분류 → 응답 JSON 구성
# =============================================================================

class ClassifyRequest(StrictModel):
    """
    분류 요청 payload를 담는 모델.

    한 줄 설명:
        Spring Boot → FastAPI classifier로 들어오는 입력 형식.

    왜 이렇게 했는가:
        품질 threshold는 상황에 따라 override하고 싶을 수 있어서,
        기본 feature_vector와 함께 선택적 threshold 파라미터를 받도록 했다.

    필드:
        feature_vector (FeatureVector): B가 만든 특징 패키지
        min_track_length (int | None): 최소 유효 트랙 길이 override
        min_mean_conf (float | None): 최소 평균 신뢰도 override
        max_missing_ratio (float | None): 최대 누락 비율 override
    """

    feature_vector: FeatureVector
    min_track_length: int | None = Field(default=None, ge=1, description="최소 유효 트랙 길이")
    min_mean_conf: float | None = Field(default=None, ge=0.0, le=1.0, description="최소 평균 탐지 신뢰도")
    max_missing_ratio: float | None = Field(default=None, ge=0.0, le=1.0, description="최대 허용 누락 비율")


class RuleFilterResult(StrictModel):
    """
    규칙 기반 필터 결과를 담는 모델.

    한 줄 설명:
        RF 추론 전에 track/feature 상태를 검사한 결과를 표현한다.

    왜 이렇게 했는가:
        품질이 너무 낮은 track는 굳이 RF에 넣기보다 uncertain으로 바로 처리하는 편이
        더 안전하다. 그래서 reject reason을 명시적으로 남기도록 했다.

    필드:
        passed (bool): 필터 통과 여부
        reject_reason (RejectReason | None): 실패 사유 코드
    """

    passed: bool
    reject_reason: RejectReason | None = None

    @model_validator(mode="after")
    def validate_reject_reason(self) -> "RuleFilterResult":
        """
        passed / reject_reason 조합의 논리 일관성을 검증한다.

        한 줄 설명:
            통과했으면 사유가 없어야 하고, 실패했으면 사유가 있어야 한다.

        왜 이렇게 했는가:
            passed=True인데 reject_reason이 채워져 있거나,
            passed=False인데 이유가 비어 있으면 downstream이 해석하기 애매해진다.

        Returns:
            RuleFilterResult: 검증을 통과한 자기 자신

        Raises:
            ValueError: passed와 reject_reason 조합이 모순될 때
        """
        if self.passed and self.reject_reason is not None:
            raise ValueError("reject_reason must be None when passed=True")
        if not self.passed and self.reject_reason is None:
            raise ValueError("reject_reason is required when passed=False")
        return self


class ResponseQuality(StrictModel):
    """
    최종 응답에 포함되는 품질 요약 블록.

    한 줄 설명:
        Spring Boot에 돌려줄 최소 품질 정보 묶음.

    왜 이렇게 했는가:
        응답에서도 A의 원래 naming인 num_points를 유지하면
        중간 remapping drift를 줄일 수 있다.

    필드:
        num_points (int): 유효 프레임 수
        track_stability (TrackStability): track 품질 등급
        feature_status (FeatureStatus): feature 계산 상태
    """

    num_points: int = Field(..., ge=0)
    track_stability: TrackStability
    feature_status: FeatureStatus


class PredictionResult(StrictModel):
    """
    최종 분류 결과를 담는 모델.

    한 줄 설명:
        C가 계산한 label / confidence / 품질 / 설명용 top feature를 함께 반환한다.

    왜 이렇게 했는가:
        단순 label만 반환하면 왜 uncertain이 되었는지, 어떤 품질 상태였는지,
        어떤 feature가 상대적으로 중요했는지 알기 어렵다.
        그래서 운영/디버깅/프론트 연동에 필요한 최소 정보를 함께 묶었다.

    필드:
        track_id (int): 대응 track 식별자
        label (PredictionLabel): bird / drone / uncertain
        confidence (float): 예측 확률
        rule_filter (RuleFilterResult): 규칙 기반 필터 결과
        top_features (dict[str, float]): 상대적 기여도가 큰 feature들
        quality (ResponseQuality): 품질 요약
        processing_time_ms (int | None): 처리 시간(ms)
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
    여러 PredictionResult를 한 번에 반환하기 위한 배치 응답 모델.

    한 줄 설명:
        다수 track 분류 결과를 묶어서 반환한다.

    왜 이렇게 했는가:
        프론트/백엔드가 전체 개수와 클래스별 집계를 한 번에 받으면
        후처리와 화면 표시가 단순해진다.

    필드:
        results (list[PredictionResult]): 개별 결과 목록
        total_count (int): 전체 개수
        drone_count (int): drone 수
        bird_count (int): bird 수
        uncertain_count (int): uncertain 수
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
    현재 A bootstrap server와 호환되는 분석 요청 모델.

    한 줄 설명:
        기존 A 서버가 받는 입력 형식을 유지하기 위한 호환용 DTO.

    필드:
        source_video_id (str): 원본 영상 식별자
        video_path (str): 서버 내 영상 경로
        stabilization_method (StabilizationMethod): 사용할 보정 방식
    """

    source_video_id: str = Field(..., min_length=1)
    video_path: str = Field(..., min_length=1)
    stabilization_method: StabilizationMethod = "ffmpeg_vidstab"


class AnalyzeResponse(StrictModel):
    """
    현재 A bootstrap server와 호환되는 분석 응답 모델.

    한 줄 설명:
        기존 A 서버가 반환하는 형식을 유지하기 위한 호환용 DTO.

    필드:
        source_video_id (str): 원본 영상 식별자
        tracks (list[TrackSequence]): 추적 결과 목록
        message (str): 처리 결과 메시지
    """

    source_video_id: str
    tracks: list[TrackSequence]
    message: str