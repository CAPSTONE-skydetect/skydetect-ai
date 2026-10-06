# SkyDetect AI 서버 (FastAPI)
# 로컬 venv 와 같은 파이썬 버전을 쓴다. models/rf_classifier.pkl 이 이 버전으로 저장됐다.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libglib2.0-0, libgomp1: opencv-python-headless 가 런타임에 요구하는 시스템 라이브러리
# ffmpeg: 브라우저가 재생 못 하는 업로드 영상을 H.264 로 변환 (services/video_transcode.py)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ai_server ./ai_server
COPY research ./research
COPY models ./models

# 업로드 영상과 트래킹 산출물. compose 에서 볼륨으로 덮어쓴다.
RUN mkdir -p storage/uploads artifacts/manual_tracks

EXPOSE 8000
CMD ["uvicorn", "ai_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
