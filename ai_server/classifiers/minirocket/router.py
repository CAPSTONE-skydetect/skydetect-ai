from fastapi import APIRouter

from ai_server.classifiers.minirocket.contracts import ClassifyRequest, PredictionResult
from ai_server.classifiers.minirocket.prediction import classify_track_sequence

router = APIRouter(tags=["classify"])


@router.post("/classify", response_model=PredictionResult)
def classify(payload: ClassifyRequest) -> PredictionResult:
    return classify_track_sequence(payload)
