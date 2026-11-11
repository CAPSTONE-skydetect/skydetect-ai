from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_server.contracts.common import *

RejectReason = Literal[
    "short_track",
    "feature_error",
    "high_noise",
    "low_confidence",
]


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

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "feature_vector": {
                    "track_id": 1,
                    "feature_status": "ok",
                    "features": {
                        "speed_median": 74.67,
                        "speed_cv": 0.21,
                        "acceleration_median": 27.89,
                        "acceleration_p95": 76.84,
                        "turn_rate_median": 0.25,
                        "turn_rate_p95": 0.94,
                        "curvature_cv": 0.93,
                        "tortuosity": 1.01,
                        "heading_change_ratio": 0.19,
                    },
                    "quality": {
                        "num_points": 10,
                        "mean_conf": 0.9,
                        "missing_ratio": 0.05,
                        "track_stability": "good",
                    },
                    "imputed_fields": [],
                },
                "min_track_length": None,
                "min_mean_conf": None,
                "max_missing_ratio": None,
            }
        },
    )

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

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {"passed": False, "reject_reason": "short_track"},
        },
    )

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
        mean_conf (float): 전체 프레임 평균 탐지 신뢰도
        track_stability (TrackStability): track 품질 등급
        feature_status (FeatureStatus): feature 계산 상태
    """

    num_points: int = Field(..., ge=0)
    mean_conf: float = Field(..., ge=0.0, le=1.0, description="전체 프레임 평균 탐지 신뢰도")
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
        top_features (dict[str, float]): 상대적 기여도가 큰 feature들 (모델 전체 기준)
        quality (ResponseQuality): 품질 요약
        processing_time_ms (int | None): 처리 시간(ms)
        base_drone_proba (float | None): 이 궤적의 판정 근거 - 트리 루트의 평균 드론 확률
        feature_contributions (dict[str, float]): 이 궤적의 판정 근거 - 피처별 드론 확률 기여.
            양수면 드론 쪽, 음수면 새 쪽. base_drone_proba + 합 = 드론 확률.
            규칙 필터에서 탈락해 모델을 부르지 않았으면 비어 있다.
    """

    track_id: int = Field(..., ge=0)
    label: PredictionLabel
    confidence: float = Field(..., ge=0.0, le=1.0, description="RF 예측 확률 (uncertain 시 0.0)")
    rule_filter: RuleFilterResult
    top_features: dict[str, float] = Field(default_factory=dict, description="RF 중요도 기반 상위 특징")
    quality: ResponseQuality
    processing_time_ms: int | None = Field(default=None, ge=0, description="FastAPI 내부 처리 시간 (ms)")
    base_drone_proba: float | None = Field(default=None, ge=0.0, le=1.0, description="기여 분해의 기준값")
    feature_contributions: dict[str, float] = Field(
        default_factory=dict, description="피처별 드론 확률 기여 (양수 = 드론 쪽)"
    )


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
