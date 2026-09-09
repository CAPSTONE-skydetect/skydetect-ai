# Research Simulation V4: 모델과 근거

이 문서는 소스와 함께 추적하는 모델 명세다. 실행 결과는 `output/physics_v4/`에 별도로 생성한다. **물리 방정식과 수치 구현의 검증(verification)은 실제 관측과의 일치 검증(validation)이 아니다.** 현재 실제 A 궤적, 조류 GPS, 기체 비행 로그로 보정한 모델은 아니다.

## 변경 범위

- `SIMULATOR_VERSION=4.0.0`, `FEATURE_VERSION=3.0.0`. 특징 산술식과 11개 특징의 구조는 변경하지 않았다.
- 기존 v3 학습/테스트 CSV는 이 모델의 결과가 아니다. 새 실행의 기본 출력 폴더는 `sim_v4`, `dataset_10000_sim_v4`로 분리한다. 다른 simulator version의 manifest가 있는 폴더는 덮어쓰지 않는다.
- 파일 이름의 `features_v3`는 특징 버전을 유지한다는 뜻이다. 원시 JSONL 파일 이름도 기존 입출력 호환성을 위해 유지하지만, 반드시 metadata의 `simulator_version`을 확인해야 한다.
- `dynamics.py`: 힘 기반 운동 적분. `generators.py`: 객체·환경·투영 연결. `behavior.py`: 확률적 명령 일정. `parameters.py`: 기준값·단위·출처·가정 수준.
- 원시 결과에는 개체의 실제 사용 파라미터, 스케일링, 힘, 자세, 포화 상태, 행동 일정, 파라미터 manifest 해시가 포함된다.

## 1. 새와 고정익

세계 좌표 z축은 위쪽이며 SI 단위다. 대기 상대속도는 `v_air = v_ground - wind`이다.

```text
q = rho * |v_air|^2 / 2
AR = span^2 / wing_area
CD = CD0 + CL^2 / (pi * AR * e)
L = q * wing_area * CL
D = q * wing_area * CD
m * dv_ground/dt = L * lift_direction + (T-D) * flight_direction - m*g*up
```

유도항력식은 [NASA Glenn](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/)의 표준 공력 관계를 사용한다. 그러나 `CD0`, `e`, `CL_max`의 수치는 해당 종을 측정한 값이 아니라 공학적 가정이다.

이전처럼 속도와 상승률을 결과에서 잘라내지 않는다. 목표 속도·선회·상승 명령이 주어지면 양력계수 및 추력 한도 내에서 힘을 생성한다. 저속에서 양력이 부족하면 하강할 수 있고 `lift_saturated`로 확인할 수 있다. `min_speed`, `max_speed`는 초기/명령 범위이지 실제 상태의 절대 범위가 아니다. `lift_saturated`는 요구 양력 부족의 지표이며, 실속 이후의 비정상 유동을 정확히 계산했다는 뜻도 아니다.

새는 날갯짓 위상에 따라 양력과 추력이 변한다. 같은 위상으로 관측 bbox도 변한다. 위상 변조를 끄는 대조군과 비교하면 질량중심의 가속도와 속도도 달라진다. `glide`와 `thermal_circle`은 추진력 0인 활공이다. 고정익 드론은 날갯짓 변조 없이 추진계 응답시간을 거쳐 추력이 변한다.

**남는 축약:** 새와 고정익은 여전히 병진 운동·bank 조종 모델의 공력 계산을 공유한다. 새의 관절·날개 비틀림·깃털·날개 관성·몸체 6자유도와 비정상 공력을 풀지는 않는다. `CL`은 자세/받음각의 완전한 동역학 대신 경로 추종기가 요청한다. 날갯짓 파형도 측정 데이터를 재현한 파형이 아니라 평균 주변의 사인 변조다. 따라서 “생체역학적으로 검증된 새 모델”이라고 표현하면 안 된다.

## 2. 쿼드로터

```text
f_i = k_thrust * rotor_speed_i^2
d rotor_speed_i/dt = (command_i - rotor_speed_i) / motor_tau
m * dv/dt = R * [0,0,sum(f_i)] + F_drag - m*g*up
J * dOmega/dt = torque - Omega x (J*Omega)
dR/dt = R * skew(Omega)
```

위치·속도, 회전행렬·몸체 각속도, 로터 속도 4개를 상태로 둔다. 로터 위치와 회전 방향에서 roll/pitch/yaw 토크를 계산하고, 모터 응답 지연과 로터 추력 상한을 적용한다. 회전 적분에는 SciPy `Rotation`을 사용한다. 자세 오차 기반 PD 제어와 속도/위치 외부 루프를 사용한다. 가속도·jerk 제한은 **명령**에 적용되며 실제 가속도는 힘의 결과다.

강체 모델의 참고는 [Lee, Leok & McClamroch](https://arxiv.org/abs/1003.2005)다. 논문의 제어기·안정성 증명을 그대로 구현하거나 재현한 것은 아니다. 공력은 `-k_drag*|v_air|*v_air` 형태의 등방성 항력이고 로터 후류·블레이드 공력·배터리 방전·지면효과는 없다. 추력 할당도 역행렬 후 개별 포화를 적용하는 축약이다. `allocation_saturated`는 이 제한이 발생한 구간이다. 바람·항력을 알고 있다는 이상적인 feed-forward 가정도 남아 있다.

## 3. 행동과 환경

- Foraging: 전 샘플이 공유하던 4초 사인 경로 대신, 잘린 gamma 분포의 유지시간과 무작위 회전각으로 갱신하는 목표 방향을 사용한다.
- Sharp turn: 90도 고정 대신 45~135도와 부호를 샘플링한다. 방향은 명령이며 실현 선회는 공력·자세·추력의 제약을 받는다.
- Dash/escape: 1.5배 고정 대신 1.2~1.65배 속도 명령, 감속은 0.5~0.8배로 다양화한다.
- Thermal: 현재 궤도에 붙어 있는 링 모양 상승풍 대신 고정된 중심에서 거리에 따라 약해지는 Gaussian core를 사용한다. 정상 유동의 축약이며 열기포·난류·환경에 적응하는 새의 의사결정을 모델링한 것은 아니다.
- Wind: OU 시간상관 돌풍을 유지하되, 바람을 객체 속도에 즉시 더하는 방식 대신 대기 상대속도를 통한 힘으로 반영한다. 공간 난류 스펙트럼이나 실제 관측 기상자료와 맞춘 모델은 아니다.

지속시간을 명시하는 이동 모델은 [상태 지속시간 관련 연구](https://pmc.ncbi.nlm.nih.gov/articles/PMC3751962/)에서 동기를 얻었다. **현재 구현은 학습된 HMM/HSMM이 아니며**, 행동 확률과 파라미터는 GPS로 추정하지 않았다. Glide/thermal 등의 큰 행동 모드는 한 트랙 안에서 고정되고, 그 안의 명령 일부만 변화한다. 향후 행동 전환까지 추정하려면 독립적인 실제 이동 자료가 필요하다.

## 4. 수치별 근거

| 항목 | 현재 채택한 값과 해석 | 근거 수준 |
|---|---|---|
| Pigeon 형태 | 0.456 kg, 날개폭 0.647 m, 날개면적 0.064 m² | [Krishnan et al., 2022 Table 1](https://nora.nerc.ac.uk/id/eprint/533121/1/rsif.2022.0168.pdf)의 비둘기 표본 기준. 종 전체 분포가 아님 |
| Seagull 형태 | 0.387 kg, 0.965 m, 0.101 m² | 같은 표의 **검은다리세가락갈매기 Rissa tridactyla** 기준. 기존 seagull 라벨을 유지했지만 모든 갈매기의 대표값은 아님 |
| Pigeon flap 기준 | 8.3 Hz, 이후 0.85~1.15 배율 | [Ros et al., 2015 결과](https://journals.biologists.com/jeb/article/218/3/480/14476/Pigeons-produce-aerodynamic-torques-through)의 저속 선회 기준값을 외삽. 순항 분포로 검증되지 않음 |
| Consumer quad | 1.388 kg, 목표 속도 상한 20 m/s, 목표 pitch 상한 42° | [DJI Phantom 4 Pro S-mode 공식 사양](https://www.dji.com/support/product/phantom-4-pro?from=landing_page&site=brandsite)의 일부 기준. 이 기체 전체의 디지털 트윈이 아님 |
| Consumer arm | 0.175 m | 공식 모터 대각선 0.350 m의 절반으로 유도 |
| Pigeon 목표 순항 15 m/s, falcon 목표 상한 40 m/s | 유지/설정한 비행 명령 범위 | 실측 종별 속도 분포가 아닌 가정 |
| 다른 subtype의 질량·관성·추력비·항력 | `parameters.py`의 값 | 아직 개별 기체/종의 측정과 연결되지 않은 가정 |
| 개체 변이 | 길이 0.9~1.1배, 면적 길이², 질량 길이³, 관성 길이⁵ | 기하학적 상사 가정. 실제 생물의 allometry 분포나 95% 구간이 아님 |
| gust, bank, flap 진폭/타 종 주파수 | `parameters.py`의 값 | 공학적 가정 |
| dropout 3%, drift 65%, ROI 0.8~1.5 | 관측 스트레스 테스트 기본값 | 실제 A 로그로 추정하지 않음. 수치에 논문 출처를 임의로 붙이지 않음 |

전체 기체 파라미터의 값·단위·출처 ID·적용 조건은 `parameter_manifest()`에 있다. `reported_anchor`, `derived_anchor`, `extrapolated_prior`, `design_prior`, `stress_test_prior`를 구분한다. 개체 스케일링 이후의 실제 값은 sample metadata에 기록된다. 공식 제원이나 문헌 표본이 출발점이더라도 무작위 변형 이후 개체가 실재 기체라는 뜻은 아니다.

## 5. 검사와 해석

```powershell
.\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\venv\Scripts\python.exe -m research.physics_validation --seeds 2
```

두 번째 명령은 모델 학습을 하지 않고 다음을 실행한다.

- 자유낙하와 무제어 호버의 해석해 비교: 위치 오차 1e-9 m 미만.
- 무풍·무추력 활공: 모든 관측 시점 사이에서 역학적 에너지가 감소하는지 확인.
- 동일한 문제를 내부 120/240/480 Hz로 적분: 240/480 결과의 4초 후 위치 차이 0.15 m 미만이며, 120/480 오차의 70% 미만인지 확인.
- 날갯짓 변조 on/off 대조: 질량중심 속도 차이 RMS > 0.02 m/s 및 수직 가속도 주파수 피크가 명령 주파수 ±0.15 Hz인지 확인. 이 수치는 효과가 구현됐는지를 판정하며 생물학적 진폭 타당성 기준이 아니다.
- 조류 3종×5행동, 드론 4종, 시나리오 4종, 길이 60/180/420, seed 2개: 총 456개 궤적. NaN, 지면 종료, 자세 행렬 오차, 관측 거절, 힘 포화를 기록한다.
- 무풍 활공에서 항력 가정값을 0.5/1/2배 바꾼 민감도 검사.

검사 임계값은 수치 일관성 기준으로 정한 공학적 기준이지 실제 분포와 유사하다는 합격선이 아니다. 출력을 보고 거절된 궤적을 숨기지 말아야 한다. 긴 활공·공격적인 명령 등에서 지면 종료나 양력 부족이 생길 수 있으며, 그 빈도는 실제 행동 빈도로 해석할 수 없다.

현재 bbox는 여전히 투영 실루엣 근사와 A-like ROI 관측 모델이다. 날개 자세·가림·motion blur를 렌더링하고 A 추적기에 실제 통과시킨 결과가 아니다. 따라서 bbox 기반 특징의 현실성 문제는 이 개선으로 해결되지 않는다. 날갯짓 모양이나 완벽한 보정 같은 합성 고유 패턴을 C가 배울 위험도 남는다.

논문에서 주장할 수 있는 범위는 “명시적인 물리 제약과 출처 수준을 가진 합성 궤적 생성기, 수치 검증과 민감도 검사”까지다. 실제 분류 성능·sim-to-real 일반화·A 관측과의 유사도는 독립적인 실제 영상/궤적 테스트가 있어야 주장할 수 있다.
