from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


StabilizationMethod = Literal[
    "none",
    "ffmpeg_vidstab",
    "opencv_ecc",
    "opencv_feature_cmc",
]


TrackStability = Literal["good", "fair", "poor"]


PredictionLabel = Literal["bird", "drone", "uncertain"]


FeatureStatus = Literal["ok", "partial", "failed"]


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
