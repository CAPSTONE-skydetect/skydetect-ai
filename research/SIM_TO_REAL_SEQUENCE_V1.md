# 4단계: 2초 시계열을 위한 실제 관측 기반 시뮬레이터 보정

상태: 실제 train 구간 측정, 그룹 CV, clock 오류 수정 및 재검증 완료. **현재 보정 프로파일의 대량 생성 채택은 보류한다.** C의 실제 분류 성능 검증은 별도다.
시계열 계약: `trajectory-sequence-1.0.1` (2초, 30Hz, 4채널).
새 생성 경로: `sequence-guidance-1.3.0`, 물리 운동은 기존 v4 힘 모델을 사용한다.
첫 실행은 `output/sim_to_real_v1`, train 진단을 반영한 두 번째 실행은 `output/sim_to_real_v2`에 보존한다.
첫 실행에서 validation을 확인했으므로 두 번째 결과는 반복 사용한 개발 평가다.

## 현재 판단 요약

최신 validation 비교는 `output/sim_to_real_v4`다. 실제·합성·기존 생성기 기준 자료를 모두 수정된 clock 계약으로 변환했다.
train 거리는 0.8271 -> 0.8173, validation은 0.7164 -> 0.8561이다. 그룹 CV도 0.8970 -> 0.9401로 악화됐다.
AR2 시간 잡음 후보는 기존 AR1 방식보다 CV 점수가 좋지 않아 선택되지 않았다. 새 행동 prior 역시 전체 유사도 개선을 입증하지 못했다.
따라서 이 후보를 기존 생성 경로에 자동 적용하지 않는다. 기존 경로의 현실성이 승인된 것도 아니다.

측정·검증 방법과 판단은 [구간 측정 및 재보정 명세](STAGE4_MOTION_REFINEMENT.md),
수치와 그림은 [최신 실행 보고서](output/sim_to_real_v4/report.md)에 있다. output은 기존 정책대로 git에서 제외되는 로컬 산출물이다.
`acceptance_review.json`은 `not_ready_for_bulk_generation`이며, 선택된 설정은 '후보 중 최선'이지 현실성 검증을 통과한 설정이 아니다.

후속 원인 분리와 단일 잔차 후보 검증은 [원인 진단 및 결정](SEQUENCE_GAP_FINDINGS.md)에 정리했다.
1.3의 이동량 조건부 잔차는 기본값 0으로 비활성 상태다. train CV 평균 0.9401 -> 0.9416으로 개선되지 않아 채택하지 않았고 validation도 다시 열지 않았다.

아래 1~6절은 1.1 경로의 초기 설계 기록이다. 1.2에서 추가한 `GuidanceProfile`, AR2/white 후보, 종횡비 표본화,
train 그룹 CV, clock 수정의 현재 실행법은 `STAGE4_MOTION_REFINEMENT.md`를 따른다. 이전 숫자는 구 계약으로 계산됐으므로 최신 숫자와 직접 비교하지 않는다.

## 1. 이번 변경과 실제 사용하는 경로

```text
실제 train NPZ + train 원본 A JSON
                 |
                 v
화면 이동 범위, 누락, 2차 차분 강도 측정
                 |
                 v
8초 잠재 비행 생성: 힘 모델 + 연속된 행동 명령
                 |
                 v
공통 카메라/관측 후보 6개를 train 분포와 비교
                 |
                 v
selected_profile.json 고정 및 SHA-256 저장
                 |
                 v
새로운 비행 seed + 실제 validation으로 평가
```

기존 `BatchRunner.simulate`는 보정 전 재현 경로로 보존했다. **이 문서의 진단용 합성 생성은 `sequence_simulator.py`를 사용한다.**
기존 feature CSV 생성 명령을 실행하면 이번 보정이 자동 적용되는 것은 아니다.

| 코드 | 역할 |
|---|---|
| sequence_simulator.py | 행동 명령, 기존 힘 모델 적분, 실제 단위의 카메라 투영, 관측 오차 |
| dynamics.py | 멀티콥터에 선택적 세계 좌표 속도 명령 추가. 없으면 기존 위치 목표 제어 |
| sequence_comparison.py | 새 2초 입력을 비교하는 진단 통계와 시계열 MMD |
| calibrate_sequence_simulator.py | train 보정, 후보 선택, 설정 고정, validation 평가와 보고서 |
| trajectory_sequence.py | 실제·합성 모두가 공유하는 기존 4채널 전처리 |

모델 입력 특징을 직접 옮기거나 합성 궤적을 2D에서 확대해 점수를 맞추지 않는다. 투영 이전 카메라의 위치/FOV를 바꾸어 관측 좌표 자체가 달라진다.

## 2. 행동 명령

멀티콥터: 이동 -> 감속 -> 위치 유지 -> 반대 방향 이동 -> 상승/하강/선회 명령을 연결한다.
속도 명령을 실제 속도에 즉시 대입하지 않고 가속도·jerk 제한, 자세 제어, 모터 응답과 힘 적분을 거친다.
위치 유지 구간에서는 진입 시의 실제 위치를 목표로 고정한다. 즉시 정지하는 것은 아니다.

조류: 추진 -> 짧은 활공 -> 선회 -> 활공 하강 -> 추진 재개를 연결한다.
추진은 기존 날갯짓 힘 모델, 활공은 추력이 0인 기존 wing 모델이다. 하강 명령이 원하는 경사를 강제로 보장하지 않으며 활공의 수직 운동은 양력·항력·중력으로 결정된다.

멀티콥터 템플릿 확률은 전환형 40%/25%, 매끄러운 순항 20%, 호버 15%다. 명령 지속시간은 0.8~2.4초다.
조류는 전환형 25%/15%, 활공 20%, 활공 선회 10%, 연속 추진 30%다. 전환 명령은 2~4초 지속한다.
조류의 짧은 train 구간에 과도한 선회·속도 변화가 생성되는 첫 실행 문제를 보고 연속 추진을 추가했다.
단일 모드 템플릿은 전체 비행 동안 유지한다. 이는 사람이 정한 prior이며 실제 행동 주석으로 추정한 확률이 아니다.
지금의 소량 영상에서 새·드론을 쉽게 구분하려고 순항·활공·호버를 삭제하지 않는다. 중심 좌표에 클래스 전용 주기 신호를 더하지 않는다.
초기 지원은 pigeon/seagull/falcon과 consumer/racing/hover 멀티콥터다. fixed-wing 구현은 기존 경로에 남으며 이번 보정 코호트에서는 제외한다.

## 3. 카메라와 관측 보정의 의미

| 값 | 설정 근거 | 해석 |
|---|---|---|
| 2초 화면 이동 폭의 10/50/90% 값 | 실제 train, 클래스당 1/2 및 원본 그룹 동등 가중치 | 원근과 속도가 섞인 겉보기 규모의 기준 |
| 배율 후보 0.75/1.3/2 | 두 번째 실행 전 고정 탐색 | train에서 부족한 화면 이동 범위를 반영한 공학적 탐색 |
| FOV 25~70도, 고도각 5~55도, 방위각 전 범위 | 공통 설계 prior | 실영상의 카메라 보정값을 추정한 것은 아님 |
| 카메라 거리 | 공통 기준 15m/s의 2초 이동(30m)과 목표 화면 규모에서 산출, 20~1500m 제한 | 실제 촬영 거리를 복원한 값이 아님 |
| jitter 후보 | 연속 관측의 2차 차분 강도에서 만든 상한의 0.35/1배 | 실제 운동이 섞인 민감도 범위이며 순수 추적 오차 아님 |
| random dropout | 실제 train 프레임 누락 비율을 기존 difficulty 가중에 맞춰 완화 | 원인별 누락을 식별한 추정이 아님 |
| burst 발생 | train track의 여러 프레임 연속 누락 비율, 최대 0.4 | 8초 합성 구간과 실자료 길이가 달라 시간당 발생률 추정으로 볼 수 없음 |
| drift 확률 0.25 / camera residual 확률 0.5 | 양쪽 클래스 공통 stress prior | 실제 drift/CMC 오류 발생률의 측정값 아님 |

카메라는 비행의 잠재 위치 중앙값을 향하는 **고정 카메라**다. 비행을 프레임에 배치하는 합성 촬영 규칙이며 실제 카메라 추적 제어를 재현한 것이 아니다.
첫 실행은 각 비행의 속도를 기준으로 거리를 맞춰 새·드론의 속도 차이를 관측에서 상쇄하는 문제가 있었다. 1.1은 객체 실제 속도와 무관한 공통 기준을 사용한다.
어느 클래스인지에 따라 카메라/잡음 프로파일을 다르게 선택하지 않는다. 객체의 물리 속도와 크기가 다르므로 같은 규칙에서도 관측 분포가 달라질 수 있다.
거리·배율·방향, 진짜 운동·추적 흔들림은 중심 좌표만으로 유일하게 분리할 수 없다. 이 결과는 물리 파라미터의 식별이 아니라 관측 분포의 근사다.

## 4. 누수 방지와 평가 범위

- 기존 real_sequences_v1의 원본 그룹 split을 유지한다. 원본의 추가 임의 분할은 없다.
- fit 함수가 읽는 원본 A JSON은 train뿐이다. metadata.csv/manifest에는 split 목록이 있지만 test NPZ와 test 원본 좌표는 열지 않는다.
- 원본 파일·NPZ·메타데이터 해시 및 4채널 계약 ID를 확인한다.
- `selected_profile.json` 저장 후 해시를 기록한 다음 validation을 연다. validation을 본 뒤 같은 실행에서 후보를 다시 고르지 않는다.
- 보정용 합성 비행은 클래스당 12개, 평가용은 클래스당 24개이며 seed 범위를 분리한다.
- 실제 train은 29개 잠정 원본 그룹, validation은 10개다. 기존에 진단한 자료이므로 미사용 독립 최종 평가가 아니다.
- 촬영 세션 독립성과 정답 행동은 아직 수동 확인이 남아 있다. 자동 그룹 분리만으로 모든 누수 가능성을 제거했다고 주장하지 않는다.
- 확인용 합성 궤적을 저장하지만 이 코호트를 그대로 최종 C train/validation/test라고 배포하지 않는다.

## 5. 지표와 판단

진단 통계는 C에 넘길 새 수작업 특징 세트가 아니다. C 입력은 계속 4채널 원시 시계열이다.

| 범주 | 지표 | 해석 |
|---|---|---|
| 투영 | screen_span, screen_path, screen_speed_median | 정규화 전 화면 너비 단위 이동 범위·경로 길이·겉보기 속도 |
| 형태 | straightness, transverse_ratio | 직선성, 주 이동축과 수직인 방향으로 퍼진 정도 |
| 형태 | turn_radians_mean, reversal_fraction | 관측된 연속 변위의 방향 변화와 120도 초과 반전 비율 |
| 시간 | step_cv, low_motion_fraction | 변위 변동, 화면 너비의 0.003/초 이하 저이동 비율 |
| 시간 | acf_lag3 | 선형 추세 제거 변위의 0.1초 자기상관 |
| 시간 | band_3_10_ratio | 변위 잔차의 3~10Hz 에너지 비율. 날갯짓 검출 확률이 아님 |
| 전체 시계열 | RBF MMD² | 4채널 전체 시계열 분포 차이. 커널 스케일은 real train에서 고정 |
| 관측 가용성 | 유효 비행 비율, 관측률, 창 제외 사유 | 보정 후 어려운 사례가 선택적으로 사라지는지 점검 |

Wasserstein은 각 원본 그룹의 총가중치를 같게 하여 계산한다. 정규화 분모는 **train real IQR**를 기본으로,
분모가 아주 작을 때 train 5~95% 범위의 10%와 고정 하한으로 안정화한다. 하한은 비율형 저이동/반전에 0.01, 나머지 0.001이다.
따라서 그래프를 단순한 모든 지표의 W/IQR라고만 표기하지 않고 `W / frozen train scale`로 표기한다.

선택 점수는 클래스·세 범주의 평균 거리 + `2*(1-유효 합성 비행 비율)`다.
단일 지표만 맞추거나 적격 궤적만 남기는 후보가 유리해지지 않도록 가용성도 반영한다.
포함률은 합성 5~95% 구간에 실제 값이 들어오는 비율이다. 분포를 무한히 넓혀 높일 수 있으므로 거리와 함께 본다.

Bootstrap은 클래스별로 실제 원본 그룹과 합성 비행 그룹을 재표집한다. 같은 track의 겹친 창을 독립으로 세지 않는다.
작은 validation과 고정된 후보 선택 아래의 구간이며, 보정 파라미터 선택 자체의 불확실성까지 포함하지 않는다.
MMD는 편향형 RBF 기술 통계로 사용한다. 별도의 가설검정이나 현실 동등성 통과 기준으로 사용하지 않는다.

판단: train 개선만으로 보정을 성공이라 부르지 않는다. validation에서 범주·클래스별 악화,
시계열 MMD와 관측 가용성까지 함께 보고 5단계 합성 추가 후보로 사용할지를 결정한다.
궁극적인 유효성은 C의 실제 기반 학습 대비 합성 추가 후 동일 실제 평가 성능으로 판단한다.

자동 검토는 validation 거리, 클래스별 MMD, bootstrap 구간, 관측 가용성의 악화를 점검한다.
검토를 통과해도 `candidate_for_downstream_ablation`까지만 표시하며 대량 생성이나 현실 동등성을 자동 승인하지 않는다.
이 검토 규칙은 개발용 보수적 판단 장치다. 모든 촬영 조건의 보장을 주는 통계 검정이 아니다.

## 6. 실행과 새 궤적 생성

```powershell
.\venv\Scripts\python.exe -m research.calibrate_sequence_simulator --output research/output/sim_to_real_v2 --previous-run research/output/sim_to_real_v1
```

이미 생성된 폴더를 덮어쓰지 않는다. 다시 실행할 경우 새 폴더를 지정한다.
아래 생성 예시는 진단과 개발용이다. 현재 선택 프로파일로 C 최종 학습 자료를 대량 생성하지 않는다.

아래 예시는 이전 1.1 결과를 읽는 예다. 현재 코드의 버전 검사에서 구 설정은 거부하며 현재 설정 형식은 새 명세를 따른다.

```python
import json
from pathlib import Path
from research.sequence_simulator import AcquisitionProfile, latent_flight, observe_flight, SEQUENCE_SIMULATOR_VERSION
from research.trajectory_sequence import window_track

saved = json.loads(Path('research/output/sim_to_real_v2/selected_profile.json').read_text())
assert saved['generator_version'] == SEQUENCE_SIMULATOR_VERSION
profile = AcquisitionProfile(**saved['profile'])
flight = latent_flight('drone', 'consumer_quad', seed=70001, duration=8)
sample = observe_flight(flight, profile)
windows, rejections = window_track(sample['track'])
# windows[i]['X']: float32 (4, 60)
```

`protocol.json`: 실행 전 설정·평가 방식·입력 해시.
`candidate_scores_train_only.csv`: 6개 후보의 모든 train 점수.
`selected_profile.json`: validation 이전에 고정한 설정.
`comparison.csv`, `results.json`, `report.md`: 전후 결과와 악화된 항목.
`*_tracks.jsonl`: 관측 JSON뿐 아니라 world_truth, optical_truth, 명령·카메라·관측 메타데이터.
`*_attempts.csv`, `*_rejections.json`: 제외 사례까지 포함한 가용성 결과.

## 참고

- [Tobin et al., Domain Randomization](https://arxiv.org/abs/1703.06907): 촬영·관측 조건 다양화 접근의 근거. 본 데이터에서 성공한다는 근거는 아니다.
- [Gretton et al., A Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html): MMD 정의. 본 구현은 원 논문의 검정 절차 전체를 수행하지 않는다.
- 기존 물리 모델·파라미터 출처는 `SIMULATION_V4.md`와 `parameters.py`에 유지한다. 새 행동 명령 지속시간과 카메라 범위는 이 프로젝트의 미보정 설계 prior다.
