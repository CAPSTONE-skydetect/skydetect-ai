from contextlib import asynccontextmanager

from fastapi import FastAPI

from importlib import import_module

from ai_server.model_config import MODEL_NAME, SEQUENCE_MODEL

from ai_server.routers.analyze import router as analyze_router
from ai_server.routers.classify import router as classify_router
from ai_server.routers.ui import router as ui_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    if SEQUENCE_MODEL:
        import_module(f"ai_server.classifiers.{MODEL_NAME}.prediction").get_classifier()
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
