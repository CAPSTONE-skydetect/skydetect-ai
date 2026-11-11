# Docker로 전체 스택 실행

프론트 · 백엔드 · AI · MySQL 을 한 번에 띄운다. 세 저장소의 서버를 각각 켜지 않아도 된다.

## 전제

세 저장소를 같은 상위 폴더에 clone 한다. AI는 이 체크아웃 하나에서
RF와 MiniRocket 이미지를 각각 빌드한다. 모델별 worktree는 필요 없다.

```text
<상위>/
├── skydetect-ai/             ← docker/docker-compose.yml 이 여기 있다
├── skydetect-backend/
└── skydetect-frontend/
```

AI 코드와 체크포인트는 같은 커밋에서 가져온다. Dockerfile의 `rf`와
`minirocket` target은 서로 다른 의존성과 모델 파일을 사용한다.
기존 `AI_RF_DIR` / `AI_MINIROCKET_DIR` 설정은 사용하지 않는다.
다른 사람이 작업 중인 모델 브랜치를 변경하거나 학습을 실행하지 않는다.
호환성, 학습 명령과 고정 기준은 [통합 기록](../docs/model_unification.md)을 참고한다.

## 실행

```bash
cd skydetect-ai/docker
cp .env.example .env        # 최초 1회. 비밀번호 채우기
docker compose up -d --build
```

- 프론트: http://localhost:5173 (로그인: `.env` 의 `OPERATOR_USERNAME` / `OPERATOR_PASSWORD`)
- 백엔드 Swagger: http://localhost:8080/swagger-ui.html
- AI(RF): http://localhost:8000/docs
- AI(MiniRocket): http://localhost:8001/docs  (기동 시 모델 준비로 20초쯤 걸린다)
- MySQL: localhost:3307 (호스트 MySQL 3306 과 겹치지 않게)

```bash
docker compose logs -f backend   # 로그
docker compose stop frontend     # 하나만 멈추고 로컬 dev 서버로 대체
docker compose down              # 전체 종료 (DB 데이터는 볼륨에 남는다)
docker compose down -v           # DB 데이터까지 삭제
```

## 구조

```text
브라우저 ──▶ frontend (nginx :5173)
              ├─ /api, /hls         ──▶ backend:8080 ──▶ mysql:3306
              ├─ /ai/*              ──▶ ai-rf:8000          (rf target)
              └─ /ai-minirocket/*   ──▶ ai-minirocket:8000  (minirocket target)
```

- 화면 상단의 **RF | MiniRocket 토글**이 분석 요청을 어느 AI 로 보낼지 고른다.
  다음 분석부터 적용되고, 선택은 브라우저에 기억된다.
  결과 화면은 모델에 맞게 표시한다 (RF: 확률 %, MiniRocket: Ridge margin. 확률 아님).

- `skydetect-frontend/nginx.conf` 가 `vite.config.js` 의 dev 프록시와 같은 규칙으로 중계한다.
  브라우저는 5173 하나만 보므로 same-origin 이 유지되어 세션/XSRF 쿠키가 그대로 동작한다.
- 두 AI 서버는 이 저장소의 `storage/`, `artifacts/` 를 같이 마운트한다. 로컬 실행과 산출물을 공유한다.
  모델 파일은 동일 커밋에서 각 이미지로 복사한다. A 파일을 런타임 마운트하지 않는다.
- 컨테이너 MySQL 은 호스트 MySQL 과 별개의 DB 다. 데이터는 `mysql-data` 볼륨에 유지된다.

## 주의

- `VITE_*` 값은 프론트 빌드 시점에 고정된다. `.env` 에서 바꾸면 `--build` 로 다시 빌드한다.
- 컨테이너는 핫 리로드가 없다. 한 서비스를 고치는 중이면 그 컨테이너만 멈추고
  평소처럼 `npm run dev` / `./gradlew bootRun` / `uvicorn` 을 띄우면 된다. 포트가 같아서 나머지와 그대로 연결된다.
- 호스트 포트 5173 · 8080 · 8000 · 8001 · 3307 이 비어 있어야 한다. `.env` 에서 바꿀 수 있다.

## 파이 옵션

`docker-compose.pi.yml`을 함께 사용하면 독립 Python+FFmpeg clip-bridge 이미지를
빌드한다. RF 이미지나 로컬 Python venv가 필요하지 않다. 파이 제어 명령과 환경값은
[하드웨어 안내](../hardware/raspberry-pi/README.md)를 따른다.
기존 서비스/DB를 그대로 두고 후보만 검증할 때는 `compose up` 대신 별도 태그로
`docker build --target rf` / `--target minirocket`을 사용한다.
