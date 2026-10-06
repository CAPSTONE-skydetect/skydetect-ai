# Docker로 전체 스택 실행

프론트 · 백엔드 · AI · MySQL 을 한 번에 띄운다. 세 저장소의 서버를 각각 켜지 않아도 된다.

## 전제

세 저장소를 같은 상위 폴더에 형제로 clone 한다. compose 가 상대경로로 빌드한다.

```text
<상위>/
├── skydetect-ai/         ← docker/docker-compose.yml 이 여기 있다
├── skydetect-backend/
└── skydetect-frontend/
```

각 저장소 루트에 `Dockerfile` 이 있어야 한다 (각 저장소에 커밋되어 있음).
각 저장소에서 테스트하고 싶은 브랜치를 체크아웃한 뒤 빌드하면 그 코드로 뜬다.

## 실행

```bash
cd skydetect-ai/docker
cp .env.example .env        # 최초 1회. 비밀번호 채우기
docker compose up -d --build
```

- 프론트: http://localhost:5173 (로그인: `.env` 의 `OPERATOR_USERNAME` / `OPERATOR_PASSWORD`)
- 백엔드 Swagger: http://localhost:8080/swagger-ui.html
- AI: http://localhost:8000/docs
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
              ├─ /api, /hls ──▶ backend:8080 ──▶ mysql:3306
              └─ /ai/*      ──▶ ai:8000   (/ai 접두어 제거)
```

- `skydetect-frontend/nginx.conf` 가 `vite.config.js` 의 dev 프록시와 같은 규칙으로 중계한다.
  브라우저는 5173 하나만 보므로 same-origin 이 유지되어 세션/XSRF 쿠키가 그대로 동작한다.
- AI 의 `storage/`, `artifacts/`, `models/` 는 호스트 폴더를 마운트한다. 로컬 실행과 산출물을 공유한다.
- 컨테이너 MySQL 은 호스트 MySQL 과 별개의 DB 다. 데이터는 `mysql-data` 볼륨에 유지된다.

## 주의

- `VITE_*` 값은 프론트 빌드 시점에 고정된다. `.env` 에서 바꾸면 `--build` 로 다시 빌드한다.
- 컨테이너는 핫 리로드가 없다. 한 서비스를 고치는 중이면 그 컨테이너만 멈추고
  평소처럼 `npm run dev` / `./gradlew bootRun` / `uvicorn` 을 띄우면 된다. 포트가 같아서 나머지와 그대로 연결된다.
- 호스트 포트 5173 · 8080 · 8000 · 3307 이 비어 있어야 한다. `.env` 에서 바꿀 수 있다.
