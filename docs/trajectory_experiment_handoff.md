# 1차 자동화 작업 지시문 (실행 기록)

상태: 2026-10-17 현재 아래 지시의 P0와 후속 E1~E6 파일럿까지 실행했다.
새 위임 프롬프트로 사용하지 말고 당시 범위와 재현 조건을 확인하는 기록으로 남긴다.
최종 결과와 후속 우선순위는 `docs/trajectory_measurement_experiment_results.md`를 따른다.

## 그대로 전달할 프롬프트

`/Users/yuchan/Documents/형주플젝/skydetect-ai-trajectory-experiments`의
`codex/trajectory-measurement-experiments` 브랜치에서 작업하세요.
먼저 `docs/trajectory_measurement_experiment_plan.md`를 읽고 P0a/P0b 및
baseline용 runner를 구현하세요. E1 이후 추적 알고리즘 변경은 이번 작업에서 하지 마세요.

기준 dev SHA는 `a240387b269f80460785bc305aea5eef6ad6d00c`입니다.
현재 브랜치와 dirty 상태를 확인하고 사용자 변경을 덮어쓰지 마세요.
기존 `/Users/yuchan/Documents/형주플젝/skydetect-ai-apart-fix-cvat-comparison`의
미커밋 평가 코드는 읽기만 하고, 필요한 변경을 새 dev에 검토하여 이식하세요.
이전 `research/features.py` 전체로 새 dev를 덮어쓰지 마세요.

첫 단계에서 필요한 입력/환경을 inventory하고 ROI, tuning, XML 객체 매칭,
영상 offset과 평가 구간을 검증하세요. 기존 6개 sample 및 gt_handoff manifest를 사용하되
과거 수동 ROI가 없으면 GT로 대신 채우지 말고 해당 sample을 blocked로 보고하세요.
bird_2 영상의 실제 클래스는 drone이고 drone_1의 bird_2는 별개의 새입니다.

평가 확장 기능과 비교 영상을 이식하고 실제 B 런타임과 수치 일치를 검증하세요.
GT를 추적 입력에 넣지 마세요. 원본 자료는 바꾸지 말고 모든 새 결과는 별도 output에 쓰세요.
영상/모델/새 결과물을 Git에 넣지 마세요. dry-run, provenance/hash, 실패/resume,
sample당 timeout, stage 제한, 메모리 제한을 갖춘 runner를 만드세요.

현재 기본 설정의 D0를 6개 객체에 실행하고 동일 환경에서 한 번 더 반복해 재현성을 확인하세요.
저장된 H0 결과와 새 dev D0는 다른 기준선으로 구분하세요.
영상 비교는 GT와 실제 A 결과가 둘 다 보이는지 프레임을 렌더링해 검수하세요.
입력 부족, 모델 계약 불일치, 자원 한계, 테스트 실패는 결과에서 숨기지 마세요.
개선 기능은 아직 없으므로 '개선됐다'고 보고하지 마세요.

환경은 먼저 기존 프로젝트의 사용 가능한 runtime을 확인하세요. 새 가중치 다운로드,
대규모 의존성 설치, 재학습, 네트워크 자동화, 기존 서버 재시작은 하지 마세요.
기존 B/C 공식과 기본 API/UI 동작은 유지하세요. 커밋/푸시/병합은 별도 요청이 없으면 하지 마세요.

완료 시 다음을 보고하세요.

1. 변경 파일 및 테스트 결과, 기준 SHA와 실행 환경.
2. 6개 객체별 입력 확인 상태와 D0 재현 여부.
3. 위치/상대 위치/운동 신호/관측률/누락 및 B 결과 표.
4. 실제 GT/A 비교 영상과 보고서 경로.
5. 다음 단계 E1/E2 실행 가능 여부와 남은 차단 조건.

## 후속 위임 경계

P0 검토 후에만 E1/E2 실행을 지시한다. 그때 초기 조건, 지표, triage 기준을 잠그고
한 기능씩 비교한다. E3~E6까지 한 번에 누적 적용하거나 가장 좋아 보이는 일부 객체만 보고하지 않는다.
