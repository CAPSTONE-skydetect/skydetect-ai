# 2026-09-28 B 시계열 전달 및 실제 자료 평가

## 판정

그룹 감사, 지상 카메라 수정, train 그룹 CV, 별도 validation 비교,
MiniRocket 실제-only/실제+합성 비교, 계약 1.0.1 test 재변환과 평가,
C 전달 ZIP 생성까지 실행했다.

**실험을 실행했다는 것과 Sim-to-Real/증강이 성공했다는 것은 다르다.**
현재 합성 데이터는 분류 성능을 낮췄다. 실제-only 모델을 기준으로 전달하고,
합성 데이터는 실험용으로만 전달한다. 대량 합성 생성은 승인하지 않는다.

## 원본 그룹

- 사용자 확인: 같은 번호는 같은 원본 영상이고 괄호는 영상 안의 다른 객체다.
- 같은 영상의 객체와 모든 파생 창을 같은 그룹으로 유지했다.
- 현재 54궤적, 49영상 그룹. train 33궤적/29그룹, validation 11궤적/10그룹, test 10궤적/10그룹.
- 원본 ID, 확인 가능한 영상 바이트 해시, 중심 좌표 해시, 파일명 계열의 교차 분할 연결은 0건이다.
- 재감사에 따라 바뀐 그룹은 없다. 현재 자료에서는 기존 분할을 보존했다.
- train 6궤적의 원본 영상은 여전히 로컬에서 찾지 못해 영상 해시가 없다.
- **서로 다른 영상의 촬영 세션 독립성은 미확인이다.** 해시가 다르다고 독립 촬영인 것은 아니다.
- 과거 개발에서 모든 실제 자료를 관찰한 이력이 있어 이번 test는 새 독립 test가 아니다.

## 카메라 및 분포

물리 모델 v4는 유지했다. 별도 sequence runner는 1.4.0으로 갱신했다.
`AcquisitionProfile(ground_camera=True)`일 때 카메라 높이 1.5m,
최대 고도각 55도 조건에 맞춰 거리/시선각/화각을 함께 계산한다.
불가능한 근거리 요청에는 광학 화각을 조정하며 좌표를 사후 확대하지 않는다.
이 수치는 설계 제약이며 실제 장비를 역추정한 값이 아니다.
기존 실험 재현을 위해 legacy 모드는 그대로 남겨 두었다.

| 항목 | 기존 촬영 모델 | 지상 카메라 span2 | 지상 카메라 span3 |
|---|---:|---:|---:|
| train 3-fold 평균 분포 거리 | 0.9401 | 0.9628 | 1.1409 |
| 가용성 벌점 포함 점수 | 1.2178 | 1.3147 | 1.4186 |
| 유효 비행 비율 | 86.11% | 82.41% | 86.11% |
| 지하 카메라 / CV 108비행 | 42 | 0 | 0 |

거리와 벌점 포함 점수는 낮을수록 좋다. train CV에서 물리적 후보 둘 중 span2를
선택했지만 legacy보다 우수한 후보라는 뜻은 아니다. 선택을 고정한 후 새 합성
시드의 validation 비교를 수행했다. validation 거리도 0.8378 → 0.9036으로 악화됐다.
시계열 MMD는 새 0.1242 → 0.1213, 드론 0.0274 → 0.0299로 일관된 개선이 아니다.

최종 실험용 train 합성은 120비행 중 104비행에서 587창이 생성됐다. 지하 카메라는 0/120이다.
영상 그룹 가중 2초 화면 범위 중앙값은 다음과 같다. 실제와 합성의 시드는 다르며,
과거 36비행 예시와 직접 짝지은 전후 비교는 아니다.

| 종류 | 실제 train | 이번 합성 train |
|---|---:|---:|
| 새 | 0.2779 | 0.1599 |
| 드론 | 0.0546 | 0.1254 |

새의 화면 범위는 여전히 작고 드론은 크다. 클래스 공통 카메라 확대만으로
둘을 해결하지 못했다. 이후 연구에서는 관측된 행동/운용 조건의 차이를
검토해야 하며, 이번 test 결과를 보고 파라미터를 재조정하지 않았다.

## MiniRocket 비교

- 입력: 2초, 30Hz, float32 `(N,4,60)`, q_x/q_y/d_x/d_y.
- 전처리: trajectory-sequence-1.0.1, fingerprint eb9be8154ee0f404.
- 실제 train 156창, validation 59창, test 42창. 합성 train 587창.
- MiniRocket 9,996개 출력과 StandardScaler는 두 구성 모두 실제 train에만 fit.
- Ridge alpha 0.1: train 그룹 3-fold CV로 선택. 같은 alpha를 두 구성에 적용.
- 합성 실험은 손실 가중치 실제 75% / 합성 25%. 각 클래스/영상 그룹에 균형 가중치.
- 전체 손실 가중치 합을 두 구성에서 동일하게 유지했다.
- 0을 분류 임계값으로 고정. margin은 확률이 아니다.
- 창 margin 평균으로 객체별 판단, 객체들의 동일 가중 평균으로 영상별 판단.
- validation 영상 macro-F1로 구성 선택. 실제-only가 선택됐다.

| 구성 | validation 영상 macro-F1 | test 영상 macro-F1 | test 객체 macro-F1 | test 창 macro-F1 |
|---|---:|---:|---:|---:|
| 실제-only | 0.8990 | 0.7619 | 0.7619 | 0.7143 |
| 실제+합성 | 0.6970 | 0.5238 | 0.5238 | 0.5227 |

validation 객체 macro-F1은 실제-only 0.8167, 혼합 0.6333이다.
영상 집계는 시연의 객체별 성능과 같지 않으며 더 좋은 숫자만 보고하면 안 된다.
test는 10영상뿐이고 실제-only도 드론 4영상 중 2개만 맞췄다.
합성 추가의 test 영상 macro-F1 차이는 -0.2381이고 그룹 bootstrap 95% 구간은
[-0.5495, 0.0000]이다. 모집단의 정확한 성능이나 모든 합성 방식의 실패를 뜻하지 않는다.

모델/선택 JSON 해시 고정 후 test를 현재 계약으로 변환했다. test 10궤적 모두 유효 창을
얻었고, 계획된 test 창 42개에서 품질 조건에 의해 제거된 창은 없었다.
첫 float32 Ridge 실행에서 수치 경고가 발생해 설정을 바꾸지 않고 계산만 float64로
재실행했다. 원 실행을 보존했고, 두 실행의 집계 성능지표는 모두 동일했다.
따라서 test 열람 이력은 첫 실행과 수치 보정 재실행 모두 남아 있다.

## 산출물

로컬 디렉터리: `research/output/sequence_handoff_20260928/`

- `group_audit.json`, `source_review.csv`: 원본/분할 감사. 원본 영상은 변경하지 않았다.
- `protocol.json`, `fold_evidence.json`, `camera_cv.csv`: 사전 설정과 fold별 적합 근거.
- `selected_profile.json`, `camera_validation.csv`: 고정된 카메라 설정과 validation 결과.
- `synthetic_train_*`: 합성 원시 궤적, 배열, 비행 가용성 및 제외 이유.
- `model_evaluation/`: 원 float32 실행. 삭제하거나 성능을 골라 쓰지 않았다.
- `model_evaluation_float64/`: 수치 보정 평가, 모델, 전체 예측, test 재변환 근거.
- `C_handoff.zip`: C에게 실제 전달할 파일. Git pull만으로 전달되지 않는다.
- `C_handoff/README.md`: 데이터 수, 방법, 한계, 실행 명령, 그래프.
- `C_handoff/SHA256SUMS.json`: 파일별 무결성.

ZIP에는 실제/합성/혼합 train, validation, test NPZ, 계약, 익명 ID 메타데이터,
실제-only 기준 모델, 전처리/추론 코드와 평가 자료가 있다.
원본 영상, 개인 파일명, 로컬 절대 원본 경로는 포함하지 않는다.
혼합 train은 별개의 독립 train이 아니라 기존 실제 train에 합성을 붙인 대조 구성이다.

## 실행 및 의존성

기존 서버의 requirements/venv는 바꾸지 않았다. aeon 1.6.0의 numba 의존성이
NumPy 2.4와 충돌하므로 연구용 별도 환경에 NumPy 2.3.5를 설치했다.

```powershell
.\venv\Scripts\python.exe -m research.prepare_sequence_handoff --output research/output/sequence_handoff_new
.\research\output\minirocket_env\Scripts\python.exe -m research.evaluate_sequence_handoff --folder research/output/sequence_handoff_new
.\venv\Scripts\python.exe -m research.package_sequence_handoff --folder research/output/sequence_handoff_new --evaluation model_evaluation
```

새 실행을 위해서는 비어 있는 새 폴더를 사용한다. 기존 결과는 덮어쓰지 않는다.
이번처럼 이미 test를 평가한 후 재실험하면 독립 test라고 다시 주장하지 않는다.
MiniRocket 코어는 [aeon 공식 구현](https://www.aeon-toolkit.org/en/stable/examples/transformations/minirocket.html)을 사용했다.

## 다음 판단

현 시점의 실행 가능한 기준선은 실제-only MiniRocket이다. 합성은 검증 실패 사실을
명시한 실험용으로 남긴다. 촬영 세션 확인/누락 원본 확보, 새 독립 평가 영상 확보는
현재 파일만으로 대체할 수 없다. B/C 런타임 서비스 연결 및 보류 정책은 별도 작업이다.

후속 train 범위 선별과 날개 관측 후보의 결과는
[SEQUENCE_REFINEMENT_RESULTS.md](SEQUENCE_REFINEMENT_RESULTS.md)에 기록했다.

## 검증 기록

- 기존 및 신규 기본 테스트를 포함한 research 전체 실행: 192 passed, 407.45초.
- 이후 추가한 추론 계약/실제 MiniRocket 연동까지 포함한 별도 연구 환경 테스트: 9 passed, 30.69초.
- 서버 환경의 같은 추가 테스트: 8 passed, 1 skipped. MiniRocket은 서버에 설치하지 않아 해당 연동 검사만 생략됐으며 위 별도 환경에서 실행했다.
- ZIP CRC와 28개 파일의 SHA-256 일치, 실제 train/validation/test 그룹 교집합 0 확인.
- 혼합 학습 가중치 실제 합 117, 합성 합 39: 총 156, 비율 75:25 확인.
- 전달 폴더만을 작업 디렉터리로 사용한 `minirocket_inference.py` 실행 성공.
- 생성한 평가/궤적 이미지 두 장을 직접 열어 레이아웃 확인.
- 서버 venv의 NumPy는 기존 2.4.4로 유지. A 코드 및 서버 의존성은 수정하지 않았다.
