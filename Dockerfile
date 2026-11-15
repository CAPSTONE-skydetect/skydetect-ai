# SkyDetect AI 서버 (FastAPI)
# 로컬 venv 와 같은 파이썬 버전을 쓴다. models/rf_classifier.pkl 이 이 버전으로 저장됐다.
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libglib2.0-0, libgomp1: opencv-python-headless 가 런타임에 요구하는 시스템 라이브러리
# ffmpeg: 브라우저가 재생 못 하는 업로드 영상을 H.264 로 변환 (services/video_transcode.py)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 업로드 영상과 트래킹 산출물. compose 에서 볼륨으로 덮어쓴다.
RUN mkdir -p storage/uploads artifacts/manual_tracks

EXPOSE 8000
CMD ["uvicorn", "ai_server.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Same source, separate dependency environments and immutable checkpoints.
# 시퀀스 모델 서버 공통층: 서빙에 필요한 런타임 의존성과 코드만 담는다.
# research 는 서버가 import 하는 전처리 모듈(B 계약)만 복사한다. 학습·평가 스크립트와
# 노트북 도구는 넣지 않는다 (테스트 이미지 docker/Dockerfile.test 가 따로 추가한다).
FROM runtime AS sequence-base
COPY requirements/serve-sequence.txt ./requirements.txt
RUN pip install -r requirements.txt && pip check
COPY ai_server ./ai_server
COPY research/__init__.py research/trajectory_sequence.py research/features.py research/feature_contract.py ./research/

FROM sequence-base AS minirocket
ENV SKYDETECT_MODEL=minirocket
COPY models/minirocket_classifier.joblib ./models/minirocket_classifier.joblib

# MultiRocket Stack: MiniRocket 과 같은 의존성(aeon), 체크포인트만 다르다.
FROM sequence-base AS multirocket
ENV SKYDETECT_MODEL=multirocket
COPY models/multirocket_stack_classifier.joblib ./models/multirocket_stack_classifier.joblib

# Last target preserves the historical default: docker build . runs RF.
FROM runtime AS rf
ENV SKYDETECT_MODEL=rf
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY ai_server ./ai_server
COPY research ./research
COPY models/rf_classifier.pkl ./models/rf_classifier.pkl
