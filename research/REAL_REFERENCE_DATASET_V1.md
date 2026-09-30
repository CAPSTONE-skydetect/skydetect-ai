# 실제 A 기준 합성 보조·검증: 개발 데이터 계약 v1

이 문서는 B-1 평가 규칙 고정과 B-2 비교 데이터 패키징 결과다. **C 모델
세 가지를 학습하거나 합성의 성능 향상을 확인한 결과는 아니다.** 현재
산출물은 `research/output/real_reference_comparison_v1/`에 있으며 Git에서
제외된다. 전달용 `research/output/real_reference_comparison_v1.zip`도
생성했다(SHA-256 `445b4d0cac9339a24775841f11efe0aa230719a945e544868ecd9d3adbc299e4`).
C에게 넘길 때 폴더 전체 또는 ZIP과 `dataset_manifest.json`을 함께 전달한다.

## B-1. 고정한 평가 기준

| 항목 | 결정·현재 증거 |
| --- | --- |
| 입력 | `trajectory-sequence-1.0.1`, fingerprint `eb9be8154ee0f404`, 2초·30Hz·1초 stride, float32 `(N,4,60)` |
| 채널 | `q_x, q_y, d_x, d_y`; A의 CMC 보정 후 중심 좌표에서 계산. bbox와 정답·품질 메타데이터는 입력 채널이 아님 |
| 전처리 | `trajectory_sequence.py`의 원본 clock, gap, 재표본화, 창 제외, 좌표 중심화/스케일 규칙을 그대로 사용 |
| 그룹 | 동일 원본 영상의 다른 객체와 모든 파생 창은 한 그룹. `새30(1)/(2)`처럼 번호가 같고 괄호가 다른 파일도 같은 원본 영상 |
| 기존 분할 | 실제 54궤적, 49개 잠정 원본 영상 그룹. train 33궤적/29그룹, validation 11궤적/10그룹, 과거 test 10궤적/10그룹. 확인 가능한 원본 ID·파일 해시 기반 교차 분할 연결 0건 |
| validation 지위 | 개발 중 반복 사용한 실제 평가 자료. 새로운 독립 holdout이 아님 |
| test 지위 | 과거 개발 중 이미 열람했으므로 이번 패키지에서 제외. 새 미사용 실제 영상이 생기기 전 최종 일반화 성능 주장은 보류 |
| 촬영 세션 | 서로 다른 영상 사이의 동일 세션 여부는 미확인. train 6궤적은 원본 영상 바이트 해시가 없음 |
| 원본 영상 감사 | 제공된 51영상/54궤적의 시간·프레임·종횡비 대응 확인. 43개는 기존 바이트 해시와 raw A CSV까지 대조, 나머지 11개는 메타데이터 수준의 대응. 독립 중심 GT는 아님 |

단일 창을 독립 영상으로 세거나, validation/test의 실제 궤적을 증강의
부모로 사용하지 않는다. 영상 그룹 단위 CV를 사용하고 그룹별 수와
클래스별 오류를 함께 보고한다. 세션 독립성이 확인되지 않았다는 한계를
결과에서 삭제하지 않는다.

## B-2. C에 제공하는 대조 자료

| 파일 | 행(2초 창) | 독립 단위 | 의미 |
| --- | ---: | ---: | --- |
| `train_real_only.npz` | 156 | 실제 원본 영상 29그룹 | 실제 A train 기준선 |
| `train_real_plus_augmented.npz` | 6,156 | **같은 실제 29그룹** | 실제 156창 + 실제 부모를 가진 시뮬레이터 잔차 증강 6,000행. 증강 행의 학습 가중치 총량 10% |
| `train_synthetic_only.npz` | 587 | 합성 비행 104그룹 | 합성만으로 학습한 모델의 실제 전이 대조군. 현재 프로파일은 현실성 승인 상태가 아님 |
| `validation_real.npz` | 59 | 실제 원본 영상 10그룹 | 세 구성에 **동일하게 적용**하는 개발 validation |

train NPZ는 `X`, `y`, `group_id`, `sample_id`, `sample_weight`, `domain`을,
validation NPZ는 앞의 네 배열을 저장한다. 세 train 파일은 **별도 분할이
아니라 서로 다른 학습 구성**이다. 모델 입력에는 `X`만 사용한다. 나머지
배열은 학습 가중치·분할·집계·출처 분석에 사용한다.

`real_metadata.csv`는 실제 train/validation 창의 품질 및 전처리 정보,
`synthetic_metadata.csv`는 합성 창의 시드·subtype·관측 정보,
`augmentation_provenance.csv`는 증강 행의 실제 부모와 donor ID를
담는다. `source_inventory.csv`에는 원본 그룹과 확인 가능한 해시를
담되, 파일명·로컬 절대 경로·원본 영상 자체는 제공하지 않는다.
`simulator_profile.json`은 합성 train의 촬영 후보 설정이다. 모든 파일과
원천 자료의 SHA-256 및 그룹 수는 `dataset_manifest.json`에 남는다.

MiniRocket·scaler·분류기를 validation에 적합하지 않는다. 세 학습
구성에서 변환기를 어디에 적합했는지 C가 명시해야 한다. 기존 혼합
실험은 실제 train에만 변환기를 적합하고 Ridge 학습에 합성을 넣었다.
합성-only 대조군에서 실제 train으로 적합한 변환기를 재사용하면
합성-only가 아니다. 현재 세 구성의 C 학습·동일 실제 validation 비교는
**다음 단계(B-3)**다.

## 현재까지의 효과 판정

- 전체 합성 25% 혼합의 실제 validation 영상 macro-F1은 실제-only
  `0.8990` 대비 `0.6970`이었다. 이 방식은 현재 기본 학습 자료가 아니다.
- 실제 궤적 기반 작은 증강의 train 그룹 CV는 `0.8569 -> 0.9293`이었지만
  실제 validation은 `0.8990 -> 0.8990`이었다. 6,000행이 서로 다른
  6,000개의 현실 사례를 뜻하지 않으며 **개발용 후보**로만 제공한다.
- `synthetic-only -> real validation` 비교는 아직 수행하지 않았다.
- 본 패키지는 데이터 준비 완료를 뜻할 뿐, 시뮬레이터 현실성이나 C 최종
  성능이 확인됐다는 뜻이 아니다.

## 재생성·검사

기존 원천 산출물은 덮어쓰지 않는다. 새 이름의 빈 출력 경로를 지정한다.

```powershell
.\venv\Scripts\python.exe -m research.package_real_reference_dataset --output research/output/real_reference_comparison_v2
.\venv\Scripts\python.exe -m pytest research/tests/test_real_reference_package.py -q
Compress-Archive -Path research/output/real_reference_comparison_v2 -DestinationPath research/output/real_reference_comparison_v2.zip -CompressionLevel Optimal
```

기본 원천은 `sequence_handoff_20260928`, `anchor_training_6000_v3`,
`anchor_augmentation_v3`이며 세 경로 모두 CLI 인자로 바꿀 수 있다.
원천 manifest, 실제 train/validation, 증강본·출처 및 보정 gate의 해시와
그룹 관계가 맞지 않으면 새 패키지 생성을 거부한다. 과거 증강 생성 코드의
CLI 기본값 변경으로 소스 해시는 달라질 수 있지만, 이 패키지는 **이미
고정된 v3 산출물의 파일 해시와 계보**를 확인하고 재사용한다.
