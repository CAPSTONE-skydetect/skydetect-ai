# GCP 배포 (main 브랜치 자동 배포)

프로젝트 `skydetect-514809`, 서울 리전(`asia-northeast3-a`). 접속 주소는 로드밸런서 IP `http://136.81.84.92`.

```text
브라우저 ─▶ HTTP 로드밸런서 ─┬─ /                                   ─▶ Cloud Storage + Cloud CDN (프론트 정적 파일)
                             └─ /api /hls /ai* /local-api /live      ─▶ 앱 VM skydetect-app (e2-medium)
                                                                         gateway(nginx) · backend(Spring) · mediamtx · clip-bridge
                                                                         ├─▶ Cloud SQL skydetect-db (MySQL 8.4, 사설 IP)
                                                                         └─▶ AI VM skydetect-latency (e2-standard-4, 내부망 전용)
                                                                              MultiRocket Stack 추론 서버
현장: 파이 ─랜선─▶ 노트북 ─RTSP(TCP)─▶ 앱 VM mediamtx:8554 (방화벽: 송출 노트북 공인 IP만)
```

## 자동 배포 (CI/CD)

| 저장소 | main 에 push 되면 | 반영 |
| --- | --- | --- |
| skydetect-ai | 테스트 후 `ai-multirocket:main`, `clip-bridge:main` 이미지를 Artifact Registry 에 올림 (`.github/workflows/deploy-gcp.yml`) | AI VM·앱 VM 의 `autodeploy.sh` 가 1분 안에 받아 재시작 |
| skydetect-backend | `backend:main` 이미지를 올림 | 앱 VM 의 `autodeploy.sh` |
| skydetect-frontend | 파이 모드로 빌드해 버킷에 올리고 CDN 캐시를 비움 | 즉시 |

- GitHub → GCP 인증은 Workload Identity Federation(풀 `github`, 제공자 `github`)이라 키 파일이 없다.
  `CAPSTONE-skydetect` 조직의 `refs/heads/main` 실행만 서비스 계정 `github-deployer` 를 쓸 수 있다.
- VM 은 GitHub 접속을 받지 않는다. `skydetect-autodeploy.timer` 가 1분마다 레지스트리의 `:main` 을 확인한다.
  확인 기록: `sudo journalctl -u skydetect-autodeploy.service`
- 이미지에는 `:main` 과 커밋 7자리 태그가 함께 붙는다. 되돌릴 때는 원하는 커밋 태그를 `:main` 으로 다시 붙인다.

## VM 에 두는 파일 (`/opt/skydetect`)

| VM | 파일 |
| --- | --- |
| AI VM | `compose.ai.yml`, `autodeploy.sh`, `autodeploy.env`(`COMPOSE_FILE=compose.ai.yml`, `SERVICES="ai-minirocket"`) |
| 앱 VM | `compose.app.yml`, `gateway.conf.template`, `mediamtx.yml`(템플릿에 송출 비밀번호를 채운 것), `relay_heartbeat.py`, `.env`, `autodeploy.sh`, `autodeploy.env`(`SERVICES="backend clip-bridge relay-heartbeat"`) |

`.env`(DB·운영자 계정, `AI_UPSTREAM`, `PI_FRONTEND_ORIGIN`)와 `mediamtx.yml` 의 송출 비밀번호는 저장소에 넣지 않는다.
설정 파일(`compose.*.yml`, `gateway.conf.template`)을 바꾸면 VM 에 복사하고 `docker compose up -d` 를 직접 실행한다.
자동 배포는 이미지만 갱신한다.

## 주의

- AI API(`/ai*/api/*`)는 인증이 없다. 공개 주소를 널리 공유하기 전에 접근 제한이 필요하다.
- 업로드 영상과 추적 산출물은 AI VM 의 `storage/`, `artifacts/` 에 쌓이고 자동으로 지워지지 않는다 (디스크 10GB).
- 파이 녹화는 클라우드에서 5분 뒤 지워진다. 클립은 그 안에 만든다.
- Windows Git Bash 에서 gcloud·docker 에 `/live/...` 같은 슬래시 인자를 넘길 때는 `MSYS_NO_PATHCONV=1` 을 켠다.
  (빌드 인자가 `C:/Program Files/Git/...` 로 바뀐 채 배포된 적이 있다)
