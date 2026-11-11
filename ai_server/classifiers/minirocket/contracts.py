from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_server.contracts.common import *

AbstainReason = Literal[
    "invalid_input",            # 계약 위반: 비보정 입력, 시계 불일치, FPS 미달 등
    "insufficient_observation", # 2초 창을 만들 관측이 부족 (짧은 track, 긴 gap)
    "low_separation",           # |평균 margin| < margin_threshold
]


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
