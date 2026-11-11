# Model unification

## Frozen scope

- Branch: `codex/ai-model-unification`, separate worktree `skydetect-ai-unified`.
- dev: `01e9fff22445b86ec328a97525f2ddc6411b4a46`.
- Hardware baseline: `a94e791620a7be3a8f5ccc7cbb12df63a5f429c6` (includes dev).
- RF reference: `16246fb028e836c451c95fc80fe68ef978d42505`.
- MiniRocket/training reference: `75f1b541c1034345c1e1116822d62a1c0bacbf82`.
- No retraining, threshold changes, frontend edits, live service replacement,
  database operations, or camera operations. Existing worktrees stay unchanged.
- One shared A implementation, preserving hardware baseline D0 default and its
  opt-in profiles. Model-specific contracts, inference and train/evaluate code
  remain separate. Two images preserve their existing dependency versions.
- Existing HTTP paths and model-specific response fields stay compatible.
  `SKYDETECT_MODEL` is a process-start setting, not a per-request toggle.
- Golden comparisons exclude elapsed time but include labels, scores, features,
  window order/rejections and abstention. Synthetic fixtures prove compatibility,
  not classification accuracy. A source hashes and regression tests guard tracking.
- New Stack training remains opt-in; no Stack promotion or new model fitting.

## 구현 결과

- `ai_server/contracts/common.py`: 동일했던 A/B 스키마의 단일 정의.
- `ai_server/classifiers/rf/`, `minirocket/`: 서로 다른 C 계약·추론·학습·평가.
- `ai_server/schemas.py`, `services/{classifier,prediction,train,evaluate}.py`:
  기존 import와 CLI를 위한 호환 경로. 기본 모델은 RF.
- `SKYDETECT_MODEL=rf|minirocket`은 프로세스 시작 시 선택한다. 실행 중 변경하지 않는다.
- 공통 `/analyze`, 업로드, 수동 ROI 추적, 다운로드 경로는 유지한다.
  RF의 `/classify/rule-filter`와 설명 필드, MiniRocket의 margin/창/보류 필드도 유지한다.
- 프론트 `/ai`와 `/ai-minirocket` 토글은 기존처럼 별도 서비스로 보낸다.
- 최신 MiniRocket 브랜치의 Stack 연구 CLI와 테스트를 가져왔다. API에는 연결하지 않았다.
- 두 모델 브랜치에서 공통이었던 FFmpeg 1920px 변환 상한·4스레드 처리를 반영했다.
  이는 업로드 변환 동작이며 파이 원본 녹화 품질/송출 설정은 변경하지 않는다.

## Docker와 하드웨어

한 AI checkout에서 `Dockerfile --target rf` / `--target minirocket`을 빌드한다.
RF는 기존 `requirements.txt`, MiniRocket은 기존 모델 브랜치에서 가져온
`requirements/minirocket.txt`를 사용한다. 두 NumPy 버전을 억지로 합치지 않는다.
모델 이미지는 각자 필요한 체크포인트 하나만 포함한다.

Pi overlay의 A 소스 bind mount를 제거했다. 두 이미지에 같은 A가 들어간다.
클립 워커는 `hardware/raspberry-pi/bridge/Dockerfile`의 Python+FFmpeg 전용 이미지다.
RF 이미지/ML 패키지에 의존하지 않는다. 워커·카메라 제어 소스와 녹화 설정은 그대로다.
기존 하드웨어 커밋은 이 브랜치의 조상으로 모두 포함돼 있다.

기존 운영 Compose, DB, 녹화 폴더, Pi, frontend/backend checkout은 변경하지 않았다.
새 후보 이미지만 별도 태그로 빌드했고 포트를 공개하거나 기존 서비스를 재시작하지 않았다.
이 브랜치를 검토한 뒤 실제 스택 전환/Windows/카메라 종단 검증은 별도로 수행해야 한다.

## 학습 담당자와 협업

현재 `model/rf`, `model/minirocket` 브랜치와 해당 작업 폴더는 수정/푸시하지 않는다.
작업 중인 증강/학습은 그대로 계속할 수 있다. 통합 브랜치가 합의된 뒤에는 모델별
구현을 `ai_server/classifiers/<model>/`에서 수정하고 공통 A/B를 복제하지 않는다.
기존 모델 브랜치를 통째로 다시 merge하면 삭제·이동 경로 충돌이 생길 수 있으므로,
후속 학습 변경은 담당자와 해당 파일 diff를 이식하고 두 모델 CI를 통과시킨다.
이번 통합이 향후 병합 충돌까지 없앤다는 뜻은 아니다.

새 학습을 실행할 때는 명시적 모듈/새 출력 경로를 권장한다. 아래 명령은 이번에
실행하지 않았으며 실제 패키지와 실행 승인이 있어야 한다.

```sh
python -m ai_server.classifiers.rf.train --data-path /data/train.csv --output-path /outputs/rf-new.pkl
python -m ai_server.classifiers.minirocket.train --package /data/package --output-path /outputs/mini-new.joblib
python -m research.minirocket_stack_training audit --collection /data/manifest.json
python -m research.minirocket_stack_training train --collection /data/manifest.json --output /outputs/stack-new
```

기존 `python -m ai_server.services.train`도 유지되지만 `SKYDETECT_MODEL`을 따른다.
PowerShell에서는 `$env:SKYDETECT_MODEL="minirocket"`, POSIX 셸에서는
`SKYDETECT_MODEL=minirocket python -m ...` 방식으로 설정한다.
학습 후 체크포인트/해시/검증 결과는 별도 변경으로 검토한다. 새 가중치를 자동 배포하지 않는다.

## 검증 결과 (2026-11-11)

| 검증 | 결과 |
| --- | --- |
| 고정 기존 이미지와 원본 브랜치 source/checkpoint 대조 | RF 9개, MiniRocket 10개 SHA-256 일치 |
| 7개 고정 궤적 입력, 모델별 분류 API 비교 | MiniRocket 완전 동일, RF 동일 판정/필드/설명; 확률 최대 차이 약 1.1e-16 |
| 요청/응답 OpenAPI와 경로 목록 | 두 모델 각각 기존 스냅샷과 일치 |
| RF 새 이미지 테스트 | 74 passed |
| MiniRocket 새 이미지 테스트 | 53 passed, 1 skipped (실제 B 패키지 없음) |
| Mac RF 환경 테스트 | 74 passed |
| 기존 하드웨어 로컬 unittest | 25 passed (mock + 합성 FFmpeg, 실제 카메라 아님) |
| RF / MiniRocket / 독립 clip 이미지 | 세 빌드 성공 |
| 독립 clip 비관리자 실행 | SQLite + 1080p 합성 영상 추출 60프레임/PTS 매핑 통과 |
| Compose 기본+Pi overlay | 구성 검증, 두 AI가 같은 build context, A 소스 mount 없음 |
| 공통 A/B/시계열 전처리·hardware runtime | 기준 커밋과 source hash 일치 |
| compileall / git diff --check | 통과 |

초기 테스트 문제도 수정했다: API 테스트의 잘못된 ROI 필드명, Mac/Linux sin 함수의
최하위 비트 차이로 인한 입력 해시 불일치(기존 Linux 입력 JSON을 고정), 테스트 이미지의
examples 모듈 누락. 기준 모델 출력/임계값을 바꿔 테스트를 통과시킨 것이 아니다.
고정 입력은 smooth, short, tiny, unstabilized, long_gap, low_conf, fps60을 포함한다.
비교 허용오차는 수치 1e-12이며 시간 측정 필드만 제외한다.

```sh
docker build --target rf -t skydetect-unified-test-rf .
docker build --target minirocket -t skydetect-unified-test-minirocket .
docker build -f docker/Dockerfile.test --build-arg RUNTIME_IMAGE=skydetect-unified-test-rf -t skydetect-unified-check-rf .
docker run --rm --network none skydetect-unified-check-rf
docker build -f docker/Dockerfile.test --build-arg RUNTIME_IMAGE=skydetect-unified-test-minirocket -t skydetect-unified-check-minirocket .
docker run --rm --network none skydetect-unified-check-minirocket
```

## 범위와 한계

- 위 결과는 구조 통합의 호환성 검증이지 A/C 성능 개선이나 모든 실제 영상의 출력 일치 증명이 아니다.
- 실제 자료 재학습 없음. Stack 단위 테스트의 소규모 합성 fit은 운영 모델/학습 데이터와 분리된다.
- `docs/minirocket_evaluation.md`, metrics, 기존 Stack protocol/results는 가져온 과거 기록이다.
  이번 통합으로 새 성능을 측정한 것이 아니다.
- 이미지 테스트는 Linux arm64, 호스트 RF 테스트는 macOS arm64에서 수행했다.
  Windows/amd64 및 원격 GitHub Actions 실행은 아직 확인하지 않았다.
- 새 전체 스택과 실제 Pi의 연속 송출/복구 검수는 이번에 다시 실행하지 않았다.
- CI는 빌드/회귀 검사만 한다. 이미지 레지스트리 게시·배포 자동화는 추가하지 않았다.

## 보존한 모델 SHA-256

- RF: `e0de459f27cbcf0fcb6265bf50f5e47f20102cedaafe7a5800b73314f83d1d3c`
- MiniRocket: `9024731a649b5e0a84ead3fcc45c9dc0a692ce305cea6e5ec8b7d720867a1e5a`

세부 코드/입력 해시는 `model_unification_sources.json`에 기록한다.
