from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_server.routers.analyze import router as analyze_router
from ai_server.routers.classify import router as classify_router
from ai_server.routers.ui import router as ui_router
from ai_server.services.prediction import get_classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MiniRocket 모델 로딩과 numba JIT 준비는 수십 초가 걸린다.
    # 첫 분석 요청이 그 비용을 떠안지 않도록 서버 기동 시점에 미리 끝낸다.
    get_classifier()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="SkyDetect-AI",
        version="0.1.0",
        description="Manual ROI tracking API for TrackSequence generation.",
        lifespan=lifespan,
    )
    app.include_router(analyze_router)
    app.include_router(classify_router)
    # 웹 프론트(skydetect-frontend)가 쓰는 업로드 / 추적+분류 / 파일 다운로드 API
    app.include_router(ui_router, prefix="/api")
    return app


app = create_app()
