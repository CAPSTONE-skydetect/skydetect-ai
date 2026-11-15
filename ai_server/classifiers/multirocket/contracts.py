"""MultiRocket Stack 은 MiniRocket 과 같은 요청(TrackSequence)을 받고 같은 모양의 결과를 낸다.

점수만 Ridge margin 이 아니라 결합기 logit 의 창 평균이라 score_type 을 바꾼다. 확률이 아니다.
"""
from typing import Literal

from pydantic import Field

from ai_server.classifiers.minirocket.contracts import *  # noqa: F403
from ai_server.classifiers.minirocket.contracts import PredictionResult as _SequencePredictionResult


class PredictionResult(_SequencePredictionResult):
    decision_score: float | None = Field(default=None, description="결합기 logit 의 창 평균. 확률 아님")
    score_type: Literal["stack_logit_mean"] = "stack_logit_mean"
