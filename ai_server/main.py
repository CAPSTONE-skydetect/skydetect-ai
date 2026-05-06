from fastapi import FastAPI

from ai_server.routers.analyze import router as analyze_router
from ai_server.routers.classify import router as classify_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="SkyDetect-AI",
        version="0.1.0",
        description="Bootstrap API for track-sequence generation.",
    )
    app.include_router(analyze_router)
    app.include_router(classify_router)
    return app


app = create_app()
