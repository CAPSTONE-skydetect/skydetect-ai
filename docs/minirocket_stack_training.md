# Optional MiniRocket Stack training

This branch adds an opt-in research CLI. The collaborator's MiniRocket,
StandardScaler, Ridge, sequence preprocessing, group weights/metrics, and
`research/requirements-minirocket.txt` are inherited. The addition is the
image-width-scale member and logistic Stack head, plus a TrackSequence
collection loader. The existing service and baseline checkpoint are unchanged.

## Setup

Stack support was published to `model/minirocket` at `fe067103`. For a new clone:

```sh
git clone --branch model/minirocket https://github.com/CAPSTONE-skydetect/skydetect-ai.git
cd skydetect-ai
python3.14 -m venv .venv-stack
. .venv-stack/bin/activate
python -m pip install -r research/requirements-minirocket.txt pydantic==2.12.5
```

Use the project root as the working directory. No second checkout or
`PYTHONPATH` is needed. Verified with CPython 3.14.3 on macOS 26.4 arm64;
wheel availability and runtime on other OSes were not tested.

## 협업자 인수인계: 기존 자료 없이 새 데이터로 학습

**이 경로에서는 기존 영상/학습 패키지/체크포인트를 사용하지 않습니다.**
재사용하는 것은 코드와 알고리즘입니다. 새 학습이 생성하는 `stack.joblib`은
기존 가중치를 수정한 파일이 아니라 두 구성 모델과 결합기를 새로 fit 한 결과입니다.

| 구분 | 역할 |
| --- | --- |
| 협업자 기존 구현 | TrackSequence 전처리, MiniRocket/Scaler/Ridge, 그룹 가중치와 평가 |
| Stack 추가분 | 이동 크기 보존 표현, 두 모델의 OOF 점수를 학습하는 결합기 |
| 이번에 준비할 자료 | 새 궤적 JSON, 라벨, 원본/촬영 그룹, 분할 manifest |
| 사용하지 않을 자료 | 기존 `real_reference_comparison_v1` 및 기존 모델 가중치 |

### 1. 코드와 환경 준비

새로 받는 경우 위 Setup 명령을 사용합니다. 기존 clone을 쓰는 경우 작업 중인
변경을 먼저 보존한 뒤 아래처럼 업데이트합니다. 분기 이력이 다르면 강제 reset을
하지 말고 협업자와 확인합니다.

```sh
git fetch origin
git switch model/minirocket
git pull --ff-only origin model/minirocket
git log -1 --oneline
```

이후 위의 격리 환경 설치를 진행합니다. 모든 Python 명령은 저장소 루트에서
실행합니다. 실행 중 다른 체크아웃의 코드를 `PYTHONPATH`로 추가하지 않습니다.
macOS arm64/Python 3.14.3에서 검증했으며 다른 OS의 새 설치는 아직 미검증입니다.

### 2. 새 궤적과 manifest 준비

원본 영상과 오버레이는 모델 입력이 아닙니다. 사람이 정답/동일 객체 추적을
검수하는 근거로 보존하고, 학습 입력은 기존 계약의 TrackSequence JSON을 사용합니다.
아래처럼 코드 저장소 밖의 별도 데이터 폴더로 전달해도 됩니다.

```text
new_dataset/
  manifest.json
  tracks/
    bird_track_001.json
    drone_track_001.json
```

아래 JSON은 **필드 작성 예시 하나**입니다. 그대로 실행 가능한 데이터셋이
아니며, 해시 자리표시자를 실제 파일의 SHA-256으로 바꾸고 새/드론 및 각 분할의
여러 레코드를 추가해야 합니다.

```json
{
  "schema": "stack-tracksequence-collection-1",
  "records": [
    {
      "source_sha256": "REPLACE_WITH_ORIGINAL_VIDEO_SHA256",
      "source_group": "capture-bird-session-001",
      "session_group": "capture-bird-session-001",
      "parent_id": "original-video-001",
      "object_id": "bird-1",
      "segment_id": "frames-30-166",
      "label": "bird",
      "split": "train",
      "qa_disposition": "accepted",
      "track_path": "tracks/bird_track_001.json",
      "track_sha256": "REPLACE_WITH_TRACK_JSON_SHA256"
    }
  ]
}
```

- `source_sha256`: 원본 촬영 영상 파일의 실제 SHA-256입니다.
- `track_sha256`: 전달한 TrackSequence 파일 바이트의 SHA-256입니다. 내용이 같아도
  JSON을 재저장하면 해시가 달라질 수 있으므로 실제 전달 파일로 계산합니다.
- `source_group`: 분할/CV에 쓰는 그룹입니다. 같은 촬영 세션의 여러 영상은
  보수적으로 같은 그룹에 둡니다. 클래스명이나 파일마다 임의의 번호만 주지 않습니다.
- `session_group`: 알려진 촬영 세션 ID입니다. 같은 값이면 현재 로더는 같은
  `source_group` 및 split을 요구합니다. 불명은 `unknown`으로 기록할 수 있지만
  독립 촬영이라는 증거는 아닙니다. `source_group` 자체는 `unknown`을 허용하지 않습니다.
- `parent_id`: 같은 원본에서 나온 부분 구간/변형들이 공유할 고유 부모 ID입니다.
  다른 원본에 공통 문자열 `video-1` 등을 중복 사용하지 않습니다.
- `object_id`, `segment_id`: 원본 내 대상과 구간 식별자입니다. 같은 원본의
  서로 다른 새라도 서로 다른 학습/평가 분할에 배치하면 안 됩니다.
- `label`: 사람이 확인한 `bird` 또는 `drone`입니다. 모델의 예측값을 쓰지 않습니다.
- `split`: `train`, `validation`, `test` 중 하나입니다. 결과를 보기 전에 고정합니다.
- `qa_disposition`: `accepted`인 레코드만 시계열 입력으로 변환됩니다.
- `track_path`: manifest 파일 위치 기준 상대 경로 또는 절대 경로입니다.
  `normalization_scale`은 전처리가 계산하므로 보통 직접 적을 필요가 없습니다.

각 파일의 SHA-256은 아래처럼 계산할 수 있습니다. JSON에는 출력된 64자리 값을
사용하며, `source_sha256`은 오버레이가 아닌 원본 영상으로 계산합니다.

```sh
python -c 'import hashlib, sys; f = open(sys.argv[1], "rb"); print(hashlib.file_digest(f, "sha256").hexdigest())' /path/to/original_video.MOV
python -c 'import hashlib, sys; f = open(sys.argv[1], "rb"); print(hashlib.file_digest(f, "sha256").hexdigest())' /path/to/new_dataset/tracks/bird_track_001.json
```

**그룹 제약:** train과 validation 모두 유효한 새/드론 창이 필요합니다.
train에는 3-fold의 각 학습/보류 fold에 두 클래스가 남을 만큼 촬영 그룹이 있어야
합니다. 각 클래스 3그룹 확보만으로 통과를 보장하지 않으며 실제 분할은 `audit`가
검사합니다. 영상/구간을 쪼개 그룹 수를 늘리지 않습니다. test 평가도 두 클래스의
유효한 창을 요구합니다. 같은 그룹에 새와 드론이 섞이면 현재 평가 구현이
거부합니다. 이 제한을 피하려고 같은 촬영을 라벨별 그룹으로 나누지 않습니다.

### 3. 검사 후 학습

`/path/to/new_dataset/manifest.json`을 실제 경로로 바꿉니다. 아래 검사 명령은
모델을 fit 하지 않습니다.

```sh
python -m research.minirocket_stack_training audit \
  --collection /path/to/new_dataset/manifest.json
```

반환된 `window_counts`와 각 `audit` 행의 `status`, `rejected`를 확인합니다.
검사 명령이 성공해도 일부 궤적은 제외됐을 수 있으므로 전체 수 대비 제외 수와
이유를 검토합니다. timestamp/frame 불일치, 짧은 구간, 긴 gap 등은 입력 계약에
따라 거부됩니다. 통과시키려고 시각을 임의 수정하거나 누락을 보간하지 않습니다.
파일 해시 검사는 코드가 수행하지만 새/드론 정답과 촬영 세션의 사실 여부는
담당자가 확인해야 합니다.

다음 명령은 **실제 학습을 실행**합니다. 데이터/분할 검토와 실행 승인이 끝난 뒤
협업자 환경에서 실행합니다. 출력 폴더는 아직 존재하지 않아야 합니다.

```sh
python -m research.minirocket_stack_training train \
  --collection /path/to/new_dataset/manifest.json \
  --output research/output/stack-new-01
```

기존 표현 모델과 이동 크기 보존 모델을 새로 학습하고, 학습 그룹 안에서 생성한
OOF 점수로 결합기도 새로 학습합니다. validation은 평가에만 쓰며, test는 학습에
쓰지 않습니다. 실제 학습 기본값은 MiniRocket 10,000 kernels입니다.
완료 후 `state.json`의 `completed`, `status.json`의 설정/해시/검증 지표,
`stack.joblib`, `alpha_cv_*.csv`, `oof_*.csv`, `validation_tracks.csv`를 확인합니다.

### 4. 평가와 개별 궤적 판정

```sh
# 설정 확인/선택용 validation 평가
python -m research.minirocket_stack_training evaluate \
  --collection /path/to/new_dataset/manifest.json \
  --model research/output/stack-new-01/stack.joblib \
  --split validation

# 모델/설정을 고정한 후에만 미사용 test 평가
python -m research.minirocket_stack_training evaluate \
  --collection /path/to/new_dataset/manifest.json \
  --model research/output/stack-new-01/stack.joblib \
  --split test

# 한 TrackSequence 판정 (학습하지 않음)
python -m research.minirocket_stack_training predict \
  --model research/output/stack-new-01/stack.joblib \
  --track /path/to/new_dataset/tracks/bird_track_001.json
```

현재 `evaluate`는 전처리를 통과한 입력의 지표를 계산하므로 `audit`의 제외 수를
따로 함께 보고해야 합니다. 평가할 때도 train/validation 기록을 포함한 원래
전체 manifest를 사용합니다. test만 담은 manifest는 현재 CLI가 지원하지 않습니다.
margin은 확률이 아닙니다. 기존 데이터나 반복해서 본 영상을 폐기/재분할했다고
새 독립 test가 되지는 않습니다.

### 5. 기여와 성능 비교

발표/협업 설명은 다음과 같이 구분합니다.

> 기존 MiniRocket 전처리와 학습 구현을 기반으로, 이동 크기를 보존하는 두 번째
> 표현과 그룹 OOF 기반 Stack 결합 방식을 추가했습니다. 기존 데이터/가중치는
> 재사용하지 않고 새 데이터로 구성 모델과 결합기를 모두 처음부터 학습합니다.

추가 효과를 확인하려면 **같은 새 자료/분할로 학습한 Baseline과 Stack**을
비교합니다. 옛 가중치와 새 Stack만 비교하면 데이터 교체 효과가 섞입니다.
현재 CLI의 평가 출력은 Stack 결과이며, 새 Baseline과의 비교표 자동화는 별도입니다.
새/드론별 성능과 입력 제외율도 함께 보고합니다. 이 절차는 재학습 방법을 제공하는
것이지 성능 향상을 보장하지 않습니다. 코드 이식 시에는 합성 smoke만 실행했으며
수집한 실제 자료로 학습하지 않았습니다.

## Legacy package input (optional)

Only when deliberately reproducing the old package, transfer and extract
`real_reference_comparison_v1` separately. It is **not needed for new collection
training above**. Its inherited validator checks hashes and lineage;
Stack uses only the real-only training arrays and common validation.

```sh
python -m research.minirocket_stack_training audit --real-package /path/to/real_reference_comparison_v1
# Run the next command only after data and training are approved.
python -m research.minirocket_stack_training train --real-package /path/to/real_reference_comparison_v1 --output research/output/stack-run-01
python -m research.minirocket_stack_training evaluate --real-package /path/to/real_reference_comparison_v1 --model research/output/stack-run-01/stack.joblib
python -m research.minirocket_stack_training predict --model research/output/stack-run-01/stack.joblib --track /path/to/track_sequence.json
```

Alternatively replace `--real-package …` with `--collection /path/to/manifest.json`.
The schema is `stack-tracksequence-collection-1` with a `records` array.
Each row needs `source_sha256`, `source_group`, `session_group`, `parent_id`,
`object_id`, `segment_id`, `label` (`bird`/`drone`), `split`
(`train`/`validation`/`test`), `qa_disposition`, `track_path` (relative to
manifest, or absolute), and `track_sha256`. See the synthetic manifest generator:

```sh
python -m examples.stack_synthetic research/output/manifest-example
python -m research.minirocket_stack_training audit --collection research/output/manifest-example/manifest.json
```

Only `accepted` QA rows produce windows. Hashes must bind actual bytes.
The builder computes each window's normalization scale from the unchanged
sequence contract. Optional `normalization_scale` is a list in accepted-window
order and is verified rather than used as an override. Rejected clocks,
stabilization, short intervals and gaps are reported; no time repair or fake
arrays are used. Source/video/group/session/parent lineage cannot cross splits.
Use conservative source groups; unknown session does not imply independence.
The inherited group metric rejects mixed bird/drone groups: **do not split one
capture by label** to make it pass. Such a collection needs a separately
reviewed group evaluation contract. Test input is reserved unless explicitly
evaluated with `evaluate --split test` on a collection.

The output path must **not exist**. It receives a checkpoint, status and
alpha/OOF/validation tables, not a replacement for the shipped baseline.
Record accepted/rejected audit rows and group assignments before running.
Only load trusted joblib files; deserialization can execute code.

## Recipe and limits

- Original q/d channels and the scale-restored member both use 4×60 inputs.
  The second member multiplies centered q by per-window scale ×400 and
  recomputes displacement. This is apparent image motion, not physical speed.
- Each member alpha is selected with inherited train-only, group-stratified
  three-fold CV: 0.1, 1, 10, 100, smallest on ties. Seed 20260928. Fold feature
  transforms and member fits never see held fold data. Full real-package
  member fitting preserves supplied weights; collection uses inherited
  class/group-balanced weights. No augmentation.
- Fixed member alphas generate OOF margins; track means fit StandardScaler and
  LogisticRegression C=1 using class/group weights. OOF is meta-training, not
  independent evaluation. Validation is evaluation-only. Inference averages
  window margins with the existing sign at zero; scores are not probabilities.
- `examples.stack_synthetic --smoke` is synthetic-only and uses 168 kernels
  in that subprocess. The real training CLI keeps 10,000. Synthetic smoke
  metrics are not performance evidence.
