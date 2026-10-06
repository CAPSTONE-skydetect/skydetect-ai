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
StabilizationMethod = Literal[
    "none",
    "ffmpeg_vidstab",
    "opencv_ecc",
    "opencv_feature_cmc",
]
TrackStability = Literal["good", "fair", "poor"]
PredictionLabel = Literal["bird", "drone", "uncertain"]
FeatureStatus = Literal["ok", "partial", "failed"]
AbstainReason = Literal[
    "invalid_input",            # 계약 위반: 비보정 입력, 시계 불일치, FPS 미달 등
    "insufficient_observation", # 2초 창을 만들 관측이 부족 (짧은 track, 긴 gap)
    "low_separation",           # |평균 margin| < margin_threshold
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
    conf: float = Field(..., ge=0.0, le=1.0, description="해당 프레임의 관측 신뢰도")


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
        source_video_id (str | None): 원본 영상 식별자
        processed_width (int): 좌표 정규화에 사용한 처리 프레임 너비
        processed_height (int): 좌표 정규화에 사용한 처리 프레임 높이
        history (list[TrackPoint]): 프레임별 좌표/크기 목록
        stabilization (StabilizationInfo | None): 영상 보정 정보
        quality (TrackQuality | None): track 품질 요약
    """

    track_id: int = Field(..., ge=0)
    source_video_id: str | None = None
    processed_width: int = Field(
        ...,
        ge=1,
        description="cx/w 정규화에 사용한 처리 프레임 너비",
    )
    processed_height: int = Field(
        ...,
        ge=1,
        description="cy/h 정규화에 사용한 처리 프레임 높이",
    )
    history: list[TrackPoint] = Field(..., min_length=1, description="프레임별 위치 목록")
    stabilization: StabilizationInfo | None = None
    quality: TrackQuality | None = None

    @model_validator(mode="after")
    def validate_history_order(self) -> "TrackSequence":
        for previous, current in zip(self.history, self.history[1:]):
            if current.frame_index <= previous.frame_index:
                raise ValueError(
                    "history must be strictly ordered by frame_index without duplicates"
                )
            if current.timestamp_ms < previous.timestamp_ms:
                raise ValueError("history timestamp_ms must not go backwards")
        return self


# =============================================================================
# [파트 B] 특징 추출 — 담당: 문형주
# TrackSequence의 좌표 시계열에서 수치 특징을 계산한 결과를 담는 스키마
# =============================================================================

class TrackFeatures(StrictModel):
    """
    B가 계산한 feature 묶음을 담는 모델.

    한 줄 설명:
        RF 분류 전 단계에서 사용하는, bbox에 의존하지 않는 궤적 기반 feature 벡터.

    왜 이렇게 했는가:
        - feature v4(research/FEATURES_V4.md) 9종을 그대로 반영한다.
        - bbox 기반 feature를 쓰지 않는다. A의 bbox가 객체 외곽 크기를 안정적으로
          나타낸다는 가정을 제거했기 때문이다. 모든 값은 post-CMC 중심점과
          실제 관측 시각만으로 계산한다.
        - 9종 모두 required로 둔다. research/features.py가 계산 실패 시 일부 값을
          None으로 채우지 않고 샘플 전체를 rejected 처리하므로, 부분 계산 상태를
          표현할 필요가 없다. 계산이 불완전하면 feature_status로 알린다.

    주의:
        speed/acceleration은 pixel/s, pixel/s^2 단위의 apparent image-plane motion이다.
        실제 물리 속도나 거리 보정 속도가 아니므로 대상까지의 거리에 영향을 받는다.

    필드:
        speed_median: 속력의 시간 가중 중앙값 (pixel/s)
        speed_cv: std(speed) / mean(speed). 배율 불변인 속도 변동성
        acceleration_median: 속력 변화율의 시간 가중 중앙값 (pixel/s^2)
        acceleration_p95: 가속도 95백분위수. 드문 급가속의 크기
        turn_rate_median: heading 변화율의 중앙값 (rad/s)
        turn_rate_p95: turn rate 95백분위수. 드문 급회전의 크기
        curvature_cv: 이동 거리당 굴곡(kappa)의 변동계수
        tortuosity: 전체 이동거리 / 시작-끝 직선거리. 1.0이 완전 직선
        heading_change_ratio: 유효 turn rate가 임계값을 넘는 시간 비율
    """

    # -------------------------------------------------------------------------
    # feature v4 9종. 순서는 research.features.FEATURE_COLUMNS와 일치시킨다.
    # -------------------------------------------------------------------------
    speed_median: float = Field(..., ge=0.0, description="속력 중앙값 (pixel/s)")
    speed_cv: float = Field(..., ge=0.0, description="속도 변동계수")
    acceleration_median: float = Field(..., ge=0.0, description="가속도 중앙값 (pixel/s^2)")
    acceleration_p95: float = Field(..., ge=0.0, description="가속도 95백분위수")
    turn_rate_median: float = Field(..., ge=0.0, description="turn rate 중앙값 (rad/s)")
    turn_rate_p95: float = Field(..., ge=0.0, description="turn rate 95백분위수")
    curvature_cv: float = Field(..., ge=0.0, description="곡률 변동계수")
    tortuosity: float = Field(..., ge=1.0, description="경로 우회 정도 (1.0 = 직선)")
    heading_change_ratio: float = Field(..., ge=0.0, le=1.0, description="방향 전환 비율")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "speed_median": 74.67,
                "speed_cv": 0.21,
                "acceleration_median": 27.89,
                "acceleration_p95": 76.84,
                "turn_rate_median": 0.25,
                "turn_rate_p95": 0.94,
                "curvature_cv": 0.93,
                "tortuosity": 1.01,
                "heading_change_ratio": 0.19,
            }
        },
    )


class FeatureProvenance(StrictModel):
    """B가 사용한 수식·좌표·시간축 계약과 입력별 변환 정보를 기록한다.

    `feature_config_id`는 research 특징 계산 설정만 식별한다. 좌표 canonical
    변환과 시간축 재표본화 정책은 별도 계약이므로 `feature_contract_id`가 이들을
    함께 묶는다. C는 이 값을 모델 bundle의 provenance와 비교할 수 있다.
    """

    feature_version: str = Field(..., min_length=1)
    feature_config_id: str = Field(..., min_length=1)
    feature_contract_id: str = Field(..., min_length=1)
    coordinate_policy: str = Field(..., min_length=1)
    timebase_policy: str = Field(..., min_length=1)
    target_feature_fps: float = Field(..., gt=0.0)
    source_processed_width: int = Field(..., ge=1)
    source_processed_height: int = Field(..., ge=1)
    coordinate_scale: float = Field(..., gt=0.0)
    canonical_width: float = Field(..., gt=0.0)
    canonical_height: float = Field(..., gt=0.0)
    aspect_ratio_matches_training: bool


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
        provenance (FeatureProvenance | None): B가 사용한 특징 계약과 좌표 변환

    상태 규칙:
        - failed  → features는 반드시 None
        - ok      → features는 반드시 존재
        - partial → features는 존재해야 하며, 보간한 필드가 있으면 imputed_fields에 기록
    """

    track_id: int = Field(..., ge=0)
    features: TrackFeatures | None = None
    quality: TrackQuality | None = None
    feature_status: FeatureStatus = "ok"
    imputed_fields: list[str] = Field(default_factory=list, examples=[[]])
    provenance: FeatureProvenance | None = None

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
# TrackSequence → 2초 창(trajectory-sequence-1.0.1) → MiniRocket + Ridge → 응답 JSON
# =============================================================================

class ClassifyRequest(StrictModel):
    """
    분류 요청 payload를 담는 모델.

    한 줄 설명:
        Spring Boot → FastAPI classifier로 들어오는 입력 형식.

    왜 이렇게 했는가:
        MiniRocket은 요약 특징이 아니라 시간 순서가 있는 중심 좌표를 받는다.
        그래서 B의 9개 요약 특징(FeatureVector) 대신 A의 TrackSequence를 그대로 받고,
        2초 창 변환은 C가 B의 trajectory_sequence 구현을 그대로 호출해 수행한다.
        학습용 전처리와 서비스용 전처리를 같은 함수로 묶어야 결과가 갈리지 않는다.

    필드:
        track_sequence (TrackSequence): A가 만든 CMC 보정 후 추적 결과
        margin_threshold (float): |평균 margin|이 이 값보다 작으면 uncertain (0이면 보류 없음)
        min_windows (int): 판정에 필요한 최소 유효 창 수
    """

    track_sequence: TrackSequence
    margin_threshold: float = Field(
        default=0.0, ge=0.0, description="|평균 margin| 보류 임계값. 확률 임계값이 아님"
    )
    min_windows: int = Field(default=1, ge=1, description="판정에 필요한 최소 유효 2초 창 수")


class PredictionResult(StrictModel):
    """
    최종 분류 결과를 담는 모델.

    한 줄 설명:
        C가 계산한 label / decision_score / 보류 사유 / 창별 점수를 함께 반환한다.

    왜 이렇게 했는가:
        RidgeClassifier의 decision_function은 확률이 아니다. 0.73을 "드론 확률 73%"로
        표시하면 안 되므로 confidence 대신 decision_score와 score_type을 반환한다.
        양수는 drone, 음수는 bird 쪽이다. 보류(uncertain)는 이유를 구분해서 남긴다.

    필드:
        track_id (int): 대응 track 식별자
        label (PredictionLabel): bird / drone / uncertain
        decision_score (float | None): 창별 Ridge margin의 평균 (보류 사유가 입력 문제면 None)
        score_type (str): 점수 종류. 항상 "ridge_margin_mean"
        score_is_probability (bool): 항상 False
        abstain_reason (AbstainReason | None): uncertain일 때 사유
        abstain_detail (str | None): 사람이 읽는 보조 설명
        windows_used (int): 판정에 쓴 2초 창 수
        window_scores (list[float]): 창별 margin
        window_rejections (dict[str, int]): 제외한 창의 사유별 개수
        window_starts_s (list[float]): 창별 시작 시각 (track 첫 관측 기준 초). 판정 근거 표시용
        model_version (str): 전처리 계약 버전 + 계약 ID + 학습 구성
        quality (TrackQuality | None): A가 보낸 track 품질 요약 그대로
        processing_time_ms (int | None): 처리 시간(ms)
    """

    track_id: int = Field(..., ge=0)
    label: PredictionLabel
    decision_score: float | None = Field(default=None, description="Ridge margin 평균. 확률 아님")
    score_type: Literal["ridge_margin_mean"] = "ridge_margin_mean"
    score_is_probability: Literal[False] = False
    abstain_reason: AbstainReason | None = None
    abstain_detail: str | None = None
    windows_used: int = Field(default=0, ge=0)
    window_scores: list[float] = Field(default_factory=list)
    window_rejections: dict[str, int] = Field(default_factory=dict)
    window_starts_s: list[float] = Field(default_factory=list)
    model_version: str
    quality: TrackQuality | None = None
    processing_time_ms: int | None = Field(default=None, ge=0, description="FastAPI 내부 처리 시간 (ms)")

    @model_validator(mode="after")
    def validate_abstain(self) -> "PredictionResult":
        if (self.label == "uncertain") != (self.abstain_reason is not None):
            raise ValueError("abstain_reason is required exactly when label='uncertain'")
        if self.windows_used != len(self.window_scores):
            raise ValueError("windows_used must match window_scores")
        if self.window_starts_s and len(self.window_starts_s) != len(self.window_scores):
            raise ValueError("window_starts_s must match window_scores")
        return self


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
# A API 호환용 모델
# =============================================================================

class AnalyzeRequest(StrictModel):
    """
    이전 A bootstrap server와 호환되는 분석 요청 모델.

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
    A server 분석 응답 모델.

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
