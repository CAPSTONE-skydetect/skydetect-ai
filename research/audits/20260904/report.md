# B Research 시뮬레이션·특징 추출 감사 보고서

작성일: 2026-09-04  
대상 브랜치: `feat/29-b-training-dataset`  
대상 커밋: `94a8303e806e60135b678b8a6ea6f15425a3a23a`  
A 비교 기준: `codex/manual-roi-a-integration`의 `0d29495187c26befc8e5952daf0ba06f546c51d9`

## 1. 결론

현재 research는 여러 조건을 빠르게 생성하고 실험하는 연구용 도구로는 쓸 만하다. 시작점·목표점·길이·이벤트·dropout·tracking drift·camera residual·드론 subtype·조류 behavior metadata를 갖춘 점도 좋다.

그러나 현재 산출물을 그대로 C의 주 학습 데이터로 넘기고 “A 산출물과 현실적으로 동등한 분포”라고 주장하기에는 부족하다. 가장 큰 이유는 다음 네 가지다.

1. 관측 좌표계가 목표 위치에 따라 프레임마다 바뀐다. 물체가 정지해도 목표점만 바뀌면 화면에서 움직인다.
2. 실제 frame gap을 만들었으나 특징 추출은 모든 인접 row를 같은 `dt`로 처리한다.
3. 속도·가속도·heading·maneuverability의 정의에 순서 정보 손실, 노이즈 증폭, 해상도 종횡비 문제와 수식 불안정성이 있다.
4. 합성 내부의 bird/drone 분리도만 측정하며, 실제 A 데이터에 대한 대표성·coverage·downstream 성능 검증은 아직 없다.

따라서 현 단계의 평가는 “다양화 기능이 작동하는 프로토타입”이다. “학습용으로 검증된 시뮬레이터” 단계는 아니다. 이 판단은 물리 모델이 단순해서라기보다, 생성 좌표의 의미와 특징 계산이 A 계약과 아직 일치하지 않는 데서 나온다.

## 2. 현재 파이프라인의 실제 구조

현재 실행 흐름은 다음과 같다.

```text
조건 샘플링
  -> 3차원 bird/drone 상태 적분
  -> goal-dependent 선형 2차원 매핑
  -> bbox/conf 독립 노이즈
  -> camera residual
  -> tracking drift
  -> frame dropout / low confidence
  -> 인접 row 차분
  -> 5개 feature CSV
  -> synthetic bird/drone 내부 통계
```

이 흐름에서 “물체의 실제 운동”, “카메라 투영”, “A 추적 오차”, “B 보간·특징 추출”이 계층별로 존재한다는 방향은 맞다. 각 계층을 독립적으로 조정하고 metadata를 남길 수 있다는 것도 장점이다.

문제는 각 계층의 수학적 의미와 적용 순서가 아직 완전히 분리되지 않았다는 점이다. 예를 들어 goal은 물체 제어뿐 아니라 화면 좌표의 축척까지 바꾸고, tracking noise는 실제 운동 feature에 그대로 미분된다. 그 결과 C가 학습하는 값이 실제 기동인지 시뮬레이터의 관측 규칙인지 구분하기 어렵다.

## 3. 가장 먼저 고쳐야 하는 정확성 문제

### 3.1 goal 변경이 가상 카메라 축척을 변경한다

`BaseAgent.get_observation()`은 다음과 같이 계산한다.

```python
max_x = max(env.x_goal[0] * 1.1, 130.0)
max_z = max(env.x_goal[2] * 1.5, 110.0)
cx = x / max_x
cy = 1.0 - z / max_z
```

goal은 조종 목표일 뿐 카메라 내부 파라미터가 아니다. 그런데 현재는 dash, brake, sharp turn, bird behavior가 goal을 바꿀 때 카메라 축척도 즉시 바뀐다.

제어 실험에서 물체 위치를 `[100, 100, 80]`으로 고정하고 goal x만 300에서 550으로 바꾸자 `cx`가 `0.3030 -> 0.1653`으로 이동했다. 물체는 움직이지 않았지만 관측상 약 `82.57 bbox-width/s` 속도가 생겼다.

140개 ideal 표본을 조사했을 때 95개에서 이런 “물체 위치를 고정한 projection-only shift”가 0.02를 넘었다. 가장 큰 값은 약 0.937이었다. 이 값은 허용 기준이 아니라 문제의 발생 빈도를 보기 위한 진단 기준이다.

이 문제는 sudden dash와 sharp turn에서 특히 심각하다. C는 시나리오를 구분하는 goal 변경 흔적을 bird/drone 운동으로 학습할 수 있다.

필요한 수정 방향은 고정된 카메라 모델이다. 최소 구현이라도 샘플 시작 시 `fx, fy, cx0, cy0`와 camera pose를 고정하고, 세계 좌표를 camera 좌표로 변환한 뒤 깊이로 나누어야 한다. OpenCV가 설명하는 pinhole 관계는 `u = fx * Xc/Zc + cx0`, `v = fy * Yc/Zc + cy0`다. goal은 이 계산에 들어가면 안 된다.

### 3.2 frame gap을 만든 뒤 시간 차이를 무시한다

dropout 구현은 row를 실제로 제거하며 `frame_index`와 `timestamp_ms`를 보존한다. A 계약을 흉내 내는 방법으로는 맞다.

하지만 `CoreFeatureExtractor`와 `RealTrackVerifier`는 모든 인접 observation에 하나의 `dt = 1/fps`를 적용한다. 실제 차이가 3프레임인 두 점도 1프레임 이동으로 해석한다.

제어 실험에서는 매 프레임 `cx += 0.001`, `w=0.05`인 완전한 일정 속도 직선을 사용했다.

| 입력 | v_mean | v_std | a_mean |
|---|---:|---:|---:|
| 연속 frame 0..9 | 0.5996 | 0.0000 | 0.0000 |
| frame 0,1,2,5,6,9 | 1.0794 | 0.5875 | 26.9838 |

운동은 같고 관측을 일부 제거했을 뿐인데 속도 평균이 약 80% 증가하고 가속도가 0에서 26.98로 바뀌었다. 이 왜곡은 noisy dataset의 label과 feature에 체계적으로 들어간다. real verifier도 같은 수식을 복사했기 때문에 동일한 잘못을 재현한다.

보간을 먼저 하든 불규칙 시계열 미분을 하든, 반드시 `timestamp_ms` 또는 `frame_index/fps`의 실제 차이를 사용해야 한다. 보간 후 feature와 gap-aware feature의 관계도 별도 시험해야 한다.

### 3.3 재현성 seed가 NumPy의 실제 난수원에 연결되지 않는다

`_run_single_simulation()`은 `default_rng(unique_seed)`로 조건을 샘플링한다. 반면 Environment, BirdDyn, DroneDyn, `get_observation()`은 전역 `np.random`을 사용한다.

같은 scenario, subtype, sample index를 연속 두 번 호출한 실험에서 sample id와 시작점은 같았지만 observations는 달랐다. 따라서 지금의 “결정론적 seed” 설명은 일부 파라미터에만 해당한다.

metadata에도 `unique_seed` 자체가 저장되지 않는다. 같은 샘플을 재현하거나 paired ideal/noisy 실험을 하기 어렵다.

하나의 `np.random.Generator`를 모든 계층에 주입하고 seed, simulator version, noise version을 metadata에 기록해야 한다. ideal/noisy pair는 같은 latent physical trajectory를 공유하고 observation noise만 달라야 효과를 분리해 측정할 수 있다.

### 3.4 non-recovery drift가 끝에서 순간적으로 사라진다

`recover=False`이면 drift 구간 안에서는 최대 offset을 유지한다. 그러나 `end` 다음 프레임에는 원본 좌표로 즉시 돌아간다.

제어 실험에서 `cx`는 frame 10에서 0.52, frame 11에서 0.50이 됐다. “회복하지 않는 drift”가 사실상 한 프레임짜리 강제 복귀 jump를 만든다.

recover가 false라면 이후 track 끝까지 offset을 유지하거나, 별도의 tracker reset/reinitialization 이벤트와 confidence/source 변화를 동반해야 한다.

## 4. 특징 계산의 문제

### 4.1 속도는 해상도 독립적인 body-length 속도가 아니다

현재 속도는 다음 식이다.

```text
sqrt(diff(cx)^2 + diff(cy)^2) / (next_w * dt)
```

`cx`는 영상 폭, `cy`는 영상 높이로 각각 정규화된다. 16:9 영상에서 동일한 10픽셀 이동은 x와 y에서 서로 다른 정규화 거리를 갖는다. 현재 식은 이 둘을 동일 좌표 단위로 더한다.

`w`로 나누는 아이디어는 크기 변화에 대한 근사 정규화로 유용할 수 있다. 그러나 A의 bbox width는 초기 ROI, KLT 점 분포, tracker recovery에 따라 변한다. 물체의 실제 body length와 동일하지 않다. 조류에서는 날개 펼침과 자세 변화의 영향도 크다.

최소한 pixel-equivalent 좌표로 변환하거나 종횡비를 보정해야 한다. bbox 정규화도 `w(t+1)` 하나 대신 양 끝 크기의 평균이나 `sqrt(area)`를 후보로 비교해야 한다. “BL/s”라는 단위는 실제 A 자료에서 scale invariance가 확인된 뒤에만 사용해야 한다.

### 4.2 원시 1차 차분은 tracking jitter를 기동으로 바꾼다

노이즈만 준 정지 물체 301프레임을 특징 추출기에 넣었더니 다음 값이 나왔다.

| v_mean | v_std | a_mean | heading_change_ratio |
|---:|---:|---:|---:|
| 3.0881 | 1.7177 | 49.9327 | 0.5697 |

즉, 현재 Gaussian bbox-center jitter만으로도 상당한 “속도, 가속도, 방향 변화”가 생성된다. 140개씩 조사한 표본의 `a_mean` 중앙값은 ideal 67.65, noisy 109.22였다. 여기에는 실제 기동, goal-induced projection shift, 관측 jitter가 모두 섞여 있으므로 물리 가속도로 해석할 수 없다.

수치 미분은 고주파 관측 노이즈를 증폭한다. B에서는 gap 보간 정책과 함께 smoothing 또는 regularized derivative를 검증해야 한다. smoothing 강도는 synthetic에서 임의로 고르지 말고, A track과 사람이 표시한 중심 또는 고품질 구간의 잔차를 기준으로 정해야 한다.

### 4.3 heading feature가 “방향 전환 빈도”를 측정하지 않는다

현재 `heading_change_ratio`는 연속 heading의 차이가 아니라 모든 heading이 전역 circular mean에서 얼마나 떨어져 있는지 계산한다. 따라서 시간 순서를 잃는다.

제어 실험에서 다음 두 궤적은 5개 feature가 허용 오차 내에서 완전히 같았다.

- heading +45도 50회 후 -45도 50회: 방향 전환 1회
- +45도와 -45도를 매번 교대: 방향 전환 99회

C가 필요로 하는 급회전·지터·선회 빈도는 이 feature로 구분할 수 없다. `wrap(heading[t]-heading[t-1])` 기반의 turning-rate 통계, 임계각 초과 비율, 곡률, 지속 선회 길이를 후보로 삼아야 한다. 거의 정지한 step에서는 heading 자체가 정의되지 않으므로 최소 변위 mask도 필요하다.

SciPy의 `circmean`은 mean resultant vector가 0이면 구현 의존 값을 반환한다고 명시한다. 여러 방향이 균등한 복잡한 궤적일수록 현재 계산의 기준 방향이 불안정해질 수 있다.

### 4.4 a_mean은 속력 변화만 재며 방향 가속도를 놓친다

현재 가속도는 `abs(diff(speed)) / dt`다. 일정 속력으로 급회전하는 물체의 구심 가속도는 0으로 나온다.

관측 평면에서라도 속도 벡터 `[vx, vy]`를 만든 뒤 `||v[t]-v[t-1]|| / dt`를 계산해야 방향 변화에 의한 가속도를 포함할 수 있다. 노이즈와 gap 처리가 먼저 안정돼야 이 값도 의미를 갖는다.

### 4.5 maneuverability_sigma는 명칭과 수식의 관계가 뒤집혀 있다

현재 식은 사실상 다음과 같다.

```text
(v_mean / 5) / (heading_change_ratio + 1e-6)
```

방향 변화가 0에 가까울수록 값이 무한히 커진다. 일정한 직선 궤적 제어 실험에서 약 119,928이 나왔고 C의 `max_maneuverability_sigma=30`에 의해 `high_noise`로 거절됐다. 매우 안정적인 직선 track이 고노이즈로 판정되는 것이다.

이 값은 `v_mean`과 `heading_change_ratio`의 결정론적 조합이므로 새 정보를 추가하지도 않는다. Random Forest에는 중복·비선형 shortcut을 제공한다. 물리적 정의와 방향성이 합의되지 않았다면 제거하거나, 검증 가능한 turn rate·lateral acceleration·curvature로 교체하는 편이 낫다.

### 4.6 고정 스케일 V_MAX=5가 데이터와 맞지 않는다

진단 표본에서 `v_mean > 5`인 샘플은 ideal 140개 중 20개, noisy 140개 중 54개였다. 그런데 visualizer의 x축은 0..5로 고정돼 있어 이 표본들이 플롯 밖으로 사라진다.

`V_MAX=5`는 clip도 calibration도 아니며, maneuverability 값만 임의로 스케일한다. 실제 train 분포에서 robust scale을 학습해 model artifact와 함께 저장하거나, 단위가 확정된 후 물리적 범위를 정해야 한다.

## 5. 조류 물리·행동 모델의 한계

BirdDyn의 bank-to-yaw 관계 `yaw_rate = g*tan(phi)/s`는 level coordinated turn의 간단한 근사로 방향은 타당하다. 속도 적응, bank limit, OU gust, 종별 크기·속도 차이를 둔 것도 연구 초기에는 유용하다.

다만 다음 한계가 있다.

1. “논문 Table 1”, “논문 식 (3),(4)”, “[cite:18]”이라는 주석에 실제 논문명, DOI, 표·식 대응표가 없다. 현재 숫자가 문헌 기반인지 경험적 조정값인지 감사할 수 없다.
2. 초기 heading은 모든 샘플에서 +x다. 목표 위치를 다양화했어도 출발 과도응답과 화면상 진행 방향 편향이 남는다.
3. bank angle 자체에 hard clip이 없다. target만 clip하고 Gaussian noise를 더하므로 상태 `phi`는 지정된 `phi_max`를 넘을 수 있다.
4. pitch는 고도 오차를 적분하지만 damping 또는 vertical speed 제약이 없다. 상승·하강 응답의 overshoot를 제어하는 물리 파라미터가 부족하다.
5. glide는 양력, 침하율, 활공비와 연결되지 않고 속도·조종 gain·노이즈 배율만 바꾼다.
6. flap_jitter는 wingbeat 동역학 대신 목표 고도를 사인파로 6~18m 이동시킨다. 실제 몸 중심의 작은 주기 운동과 tracker bbox 변화가 구분되지 않는다.
7. thermal_circle은 목표점을 원으로 움직이게 할 뿐 agent가 실제로 반경·각속도를 달성했는지 검사하지 않는다.
8. foraging_zigzag와 sudden_escape도 목표점 명령이며, 실현된 turn rate·가속도·에너지·최소 선회 반경을 검증하지 않는다.
9. species는 pigeon/seagull/falcon 세 점 파라미터다. 같은 종의 개체차, 나이, 날개짓 상태, 군집 비행, 착륙·이륙, 바람에 대한 orientation 차가 없다.
10. 화면 bbox는 자세나 날개 주기에 따른 aspect ratio를 모델링하지 않는다.

조류를 완전한 생체역학 모델로 만들 필요는 없다. C가 쓰는 관측 feature를 재현할 수준의 중간 모델이면 된다. 다만 behavior label을 붙였으면 “명령한 behavior”와 “실제로 나타난 관측 통계”를 모두 저장하고, 후자를 기준으로 acceptance test를 해야 한다.

## 6. 드론 subtype 모델의 한계

DroneDyn은 목표 속도 벡터에 접근하는 제한 가속도 controller다. consumer/racing/hover/fixed-wing의 `s_star`, `a_max`, response gain을 분리한 것은 이전 단일 quad보다 낫다.

하지만 subtype 이름에 비해 실제 운동 구조의 차이는 작다.

1. `sigma_u`는 config에 정의되지만 step에서 한 번도 사용되지 않는다.
2. hover_quad는 hover 상태를 갖지 않는다. 목표가 현재 위치일 때 `get_unit_vector_to_goal()`은 현재 heading을 반환하므로 계속 비행한다. 제어 실험에서 무풍·무노이즈 hover_quad는 2초 동안 16m 이동했고 속도는 8m/s였다.
3. sudden dash는 goal을 멀리 옮길 뿐 목표 속도 크기를 높이지 않는다. 같은 방향에서 14m/s로 시작한 consumer quad는 dash 직후에도 14m/s였다.
4. fixed_wing도 quad와 같은 velocity-vector controller를 사용한다. 최소 속도, stall, bank-limited turn, climb rate, coordinated turn이 없다.
5. quad도 yaw·tilt·thrust·drag 없이 3차원 속도 벡터를 직접 수정한다. subtype별 acceleration envelope 또는 jerk 차이가 충분히 드러나는지 보장되지 않는다.
6. 모든 subtype의 초기 속도는 10~14m/s로 동일하다. racing, hover, fixed-wing의 시작 과도응답이 subtype shortcut이 될 수 있다.
7. 바람은 모든 기체에서 동일한 단순 합산이고 크기·drag·제어 보상 차이가 없다.
8. multi_mode의 “hover”가 새와 드론 모두에게 같은 goal 조작으로 적용돼 클래스별 행동 의미가 흐려진다.

fixed-wing에는 최소한 비영 속도, bank/turn-rate limit, climb-rate limit을 갖는 별도 kinematic model이 필요하다. quad에는 commanded acceleration과 hover state를 두고 subtype별 acceleration/jerk/position-hold residual 분포를 실측으로 맞추는 편이 낫다.

## 7. 관측·추적 노이즈 모델의 한계

현재 noise layer의 장점은 camera motion, tracker drift, dropout을 metadata로 나눈 점이다. A가 accepted visible observation만 history로 넘긴다는 최신 계약과 dropout 방식도 맞는다.

부족한 점은 다음과 같다.

1. `conf`가 uniform random이다. A의 confidence는 KLT point ratio, forward-backward error, background inlier, appearance, motion recovery와 source별 상한으로 계산된다.
2. noise 크기가 bbox 크기와 무관하다. `cx/cy sigma=0.003`은 큰 표적에는 작고 작은 표적에는 매우 큰 오차다.
3. bbox center, width, height noise가 독립이다. 실제 ROI scale drift와 중심 drift는 상관될 가능성이 높다.
4. dropout은 위치, 크기, contrast, motion blur, camera shake와 조건부로 연결되지 않는다.
5. burst 뒤 재획득 시 jump, source 전환, confidence 회복 곡선, ID switch가 없다.
6. camera shake는 x와 y에 같은 사인파를 0.6 비율로 넣어 반복적인 대각 패턴을 만든다.
7. `shake_frequency`는 sample 전체 progress에 대한 cycle 수다. 이름만 보면 Hz로 오해할 수 있고 track 길이가 달라지면 실제 주파수가 달라진다.
8. camera residual은 affine-like shift/scale만 있으며 rotation, rolling shutter, parallax와 CMC failure mode가 없다.
9. 좌표·bbox를 0..1 또는 0.005..0.2로 clip한다. 경계에 붙은 값은 실제 추적 패턴이 아니라 saturation artifact다. ideal 140개 중 20개, 총 1,132 point가 화면 경계에 있었다.
10. 화면 밖으로 나간 물체도 clipped 좌표로 계속 관측된다. 실제 A라면 visible=false가 되거나 track이 종료돼야 한다.
11. `distance=max(y,50)` 때문에 camera 뒤쪽 또는 지나치게 가까운 위치도 유효 bbox로 바뀐다.
12. ideal 모드도 BirdDyn/DroneDyn의 process noise와 gust를 사용한다. “ideal”은 observation-noise-off에 가까우며 결정론적 truth trajectory는 아니다.

A의 debug observation에는 tracking source, visibility, forward-backward error, appearance score, foreground contrast, motion score가 있다. 작은 real calibration set에서 이 조건부 분포를 측정해 simulator noise를 맞춰야 한다.

## 8. A와 B 사이의 의미 불일치

A 비교 브랜치의 계약상 history에는 accepted visible observation만 들어가고 prediction-only frame은 gap으로 남는다. `cx/cy`는 CMC가 적용되면 camera-motion-compensated coordinate이며 `conf`는 detector confidence가 아니라 observation confidence다.

research의 `coordinate_space=post_cmc_residual_observation` 선언은 방향상 맞다. 그러나 실제 A의 residual 크기나 spectrum에서 추정한 값이 아니다.

또한 A의 bbox는 KLT point spread를 혼합해 갱신하며 초기 ROI의 0.65~2.2배로 제한된다. research bbox는 실제 물체 크기와 y 거리로 계산한다. 두 bbox는 생성 원리가 다르다. 따라서 bbox-normalized speed를 비교하기 전에 A의 bbox scale bias부터 측정해야 한다.

현재 B runtime 파일인 `feature_core.py`, `feature_signal.py`, `interpolate.py`, `fractal.py`는 사실상 placeholder다. research와 real verifier에는 동일한 잘못된 수식이 복사돼 있다. 공용 feature implementation을 하나 만든 뒤 synthetic과 real 모두 같은 함수를 호출하도록 해야 식 drift를 막을 수 있다.

## 9. 통계·검증 방식의 부족

`StatisticalEngine`은 synthetic bird와 synthetic drone 사이에서 Welch t-test와 Cohen's d를 계산한다. 이 분석으로 알 수 있는 것은 “현재 simulator가 만든 두 label의 feature 평균이 다른가”다.

이 결과만으로 다음은 알 수 없다.

- 실제 A bird/drone에도 같은 차이가 있는가
- synthetic이 실제 feature 범위를 덮는가
- C가 simulator artifact를 학습했는가
- 새 영상·카메라·추적 품질에서도 분류되는가
- 확률값이 보정돼 있는가

`all_features_valid = 모든 p<0.05`도 validity 기준으로 부적절하다. 표본 수가 크면 작은 차이도 유의해지고, 표본 수가 작으면 큰 차이도 놓칠 수 있다. 분류 feature는 모든 단변량 평균이 달라야 할 필요도 없다.

Cohen's d에서 Gaussian·동분산을 가정해 계산한 overlap estimate는 현재 다봉성 scenario/subtype mixture에 맞지 않을 수 있다. z-score verifier도 실제 track 하나를 synthetic class 전체의 mean/std에 비교하므로 multi-modal coverage를 제대로 측정하지 못한다.

더 큰 문제는 `statistical_engine.py`와 `visualizer.py`가 v2가 아니라 `simulation_features.csv`를 고정 입력으로 읽는다는 점이다. 메인 pipeline은 noisy `simulation_features_v2.csv`를 생성한다. 현재 “최신 noisy 분석”이라고 생각해도 실제로는 ideal CSV를 읽을 가능성이 높다.

`RealTrackVerifier`는 실제 TrackSequence의 `timestamp_ms`를 무시하고 사용자가 지정한 단일 fps, 기본 50을 사용한다. 실제 영상이 30fps인데 50으로 읽으면 속도 계열은 약 5/3배, 가속도 계열은 약 25/9배 스케일될 수 있다. source, stabilization, confidence, resolution, quality metadata도 최종 real CSV에서 사라진다.

## 10. 학습 데이터 구성의 문제

현재 C 학습 코드는 CSV의 5개 feature와 label 전체로 Random Forest를 fit한다. holdout 생성, group split, 교차검증, class probability calibration, model card가 없다.

feature CSV에는 sample id, subtype, scenario, behavior, seed, noise profile, frame count가 사라진다. 이 상태에서는 다음을 할 수 없다.

- 같은 latent trajectory family를 한 split에 묶기
- 특정 subtype/scenario held-out 평가
- noisy/ideal pair 비교
- failure sample의 생성 원인 추적
- dataset version과 모델 재현

Random row split을 나중에 추가하는 것만으로는 부족하다. `sample_id`도 ideal/noisy 또는 simulator version을 포함하지 않아 pair·버전 식별자로 충분하지 않다.

권장 테이블은 feature row와 metadata manifest를 sample key로 연결하는 구조다. 최소 column은 `sample_id, family_id, seed, generator_version, label, subtype, scenario, behavior, fps, target_frame_count, observed_count, gap metrics, confidence metrics, camera profile, drift profile`이다.

## 11. 현재 테스트가 보장하는 것과 보장하지 않는 것

현재 repository tests는 C RuleFilter의 상태·threshold 동작을 검사한다. research generator, projection, dropout-aware feature, bird/drone dynamics, reproducibility, metadata, split에 대한 자동 테스트는 없다.

이번 감사에서는 source를 수정하지 않고 실제 함수를 호출하는 두 진단 프로그램을 실행했다.

- `probe_core.py`: projection invariance, gap invariance, hover, dash, seed 재현성, heading 순서, drift 종료, C filter 연동
- `probe_batch.py`: scenario × 7 subtype × 5 index × ideal/noisy, 총 280 sample 진단

280 sample은 결함을 재현하기 위한 smoke audit이며 분포 적합성을 판정할 표본은 아니다. 실제 A track을 넣지 않았으므로 이번 감사로 sim-to-real 유사성을 측정했다고 해석하면 안 된다.

기존 단계별 validation CSV에서 “범위가 넓어졌다”, “gap이 생겼다”, “mode별 bbox 방향이 맞다”를 본 것은 기능 작동 확인으로는 유효하다. 그 검증은 카메라 모델의 불변성, 시간 미분의 정확성, 실제 A 분포 적합성까지 보장하지 않았다.

## 12. 권장 수정 순서와 통과 조건

### P0: 데이터 의미를 먼저 고정

1. A/B contract의 좌표 공간, confidence 의미, bbox 의미, fps/timestamp 규칙을 문서화한다.
2. 고정 camera model을 적용하고 goal과 projection을 분리한다.
3. 공용 gap-aware interpolation/feature 함수를 만든다.
4. 모든 RNG를 하나의 seed chain으로 통일한다.
5. CSV와 metadata를 sample key로 함께 보존한다.

통과 조건:

- 고정 3D point에서 goal만 바꿔도 `cx/cy/w/h`가 변하지 않는다.
- 같은 seed와 config는 byte-equivalent observation을 만든다.
- 일정 속도 선형 truth는 dropout 전후 보간 후 feature가 허용 오차 내에서 같다.
- fps 30/50/60으로 동일 연속시간 truth를 샘플링해도 단위 feature가 허용 오차 내에서 같다.

### P1: feature를 수학적으로 다시 정의

1. 종횡비를 반영한 관측 평면 속도
2. 실제 `delta_t` 기반 속도·가속도
3. noise-aware smoothing/derivative
4. consecutive heading delta와 low-speed mask
5. vector acceleration, curvature, bbox area/scale trend
6. maneuverability_sigma 제거 또는 명시적 물리 정의

통과 조건:

- 정지 + A에서 측정한 jitter의 feature가 실제 기동과 구분된다.
- 1회 turn과 99회 turn이 turning feature에서 크게 구분된다.
- 안정 직선이 high_noise로 거절되지 않는다.
- 해상도만 바꾼 같은 pixel-space motion의 feature가 불변이다.

### P2: 실제 A에서 noise와 조건 분포를 추정

작은 실제 영상 세트에 대해 ROI를 수동 지정하고, overlay를 확인하며, good/fair/poor quality로 층화한다. 사람이 일부 프레임에 중심과 bbox를 표시해 A 오차를 추정한다.

그 결과로 다음 conditional model을 맞춘다.

- source별 confidence
- bbox size와 center error
- motion magnitude/blur/contrast와 dropout
- occlusion 길이와 reacquisition jump
- CMC residual의 이동·회전 spectrum
- track termination과 ID switch

통과 조건은 임의 숫자로 선결정하지 않는다. real calibration set의 bootstrap confidence interval과 task tolerance에서 정한다.

### P3: 조류·드론 동역학을 필요한 범위만 보강

1. initial heading/pose를 카메라 시야 조건과 함께 다양화
2. quad hover/acceleration/jerk state 구현
3. fixed-wing minimum speed, bank/turn/climb limit 구현
4. bird behavior별 실현 trajectory constraint 정의
5. 모든 문헌 파라미터에 출처·단위·허용범위 기록
6. 명령 profile과 실현 profile을 함께 저장

물리적 정확도를 6-DOF까지 높이는 것이 목표는 아니다. C feature에 영향을 주는 관측 통계가 실제 A 범위를 덮는지에 집중한다.

### P4: sim-to-real 검증과 C 평가

최종 평가는 다음 네 층으로 해야 한다.

1. 단변량: ECDF, quantile, Wasserstein distance와 bootstrap interval
2. 다변량: PCA/UMAP 시각화, MMD 또는 classifier two-sample test
3. coverage: real sample의 synthetic neighborhood 거리와 subtype/scenario별 coverage
4. downstream: synthetic-only로 학습하고 완전히 분리한 real video group에서 AUROC, macro-F1, balanced accuracy, confusion matrix, calibration을 평가

domain classifier accuracy가 50%에 가깝다는 사실만으로 충분하지 않다. class-conditional `P(feature | label)`도 맞아야 하며 실제 held-out 성능이 마지막 판단 기준이다. 영상, 촬영 세션, camera, 원본 source 단위로 group split해야 한다.

신뢰성 주장은 단계적으로 써야 한다.

- coverage가 부족하면 “현재 simulator가 포괄하지 못한 real regime”를 찾는다.
- coverage는 좋지만 real 성능이 낮으면 class-conditional 관계 또는 label shortcut을 의심한다.
- A quality에 따라 성능이 급락하면 quality-conditioned model, rejection 또는 uncertainty 정책이 필요하다.
- synthetic-only보다 소량 real fine-tuning이 안정적으로 좋아지면 현실적인 hybrid 전략을 채택한다.

## 13. 학술적으로 방어 가능한 주장 범위

현재 코드로 방어 가능한 주장은 다음 정도다.

> 여러 기동·관측 교란 조건을 갖는 합성 궤적 생성기를 구현했고, 각 다양화 기능의 작동 여부를 내부 검증했다.

현재 코드로는 다음 주장을 하기 어렵다.

> 합성 데이터가 실제 A 궤적 분포를 대표하며, 이 데이터로 학습한 C가 실제 새와 드론을 신뢰성 있게 분류한다.

후자의 주장을 하려면 real held-out evaluation이 필수다. 합성 분포를 넓게 만드는 domain randomization은 유용한 전략이지만, 분포 정렬만으로 target 성능이 자동 보장되지는 않는다. 특히 label별 조건 분포가 다르면 domain-invariant 표현도 실패할 수 있다는 이론 결과가 있다.

## 14. 참고 근거

- OpenCV, Camera Calibration and 3D Reconstruction: 고정 camera intrinsic/extrinsic과 perspective projection 관계
  - https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html
- van Breugel, Kutz, Brunton, Numerical differentiation of noisy data: 관측 노이즈 미분의 ill-posed 성격과 smoothing/faithfulness tradeoff
  - https://arxiv.org/abs/2009.01911
- Tobin et al., Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World
  - https://arxiv.org/abs/1703.06907
- Zhao et al., On Learning Invariant Representations for Domain Adaptation: marginal alignment만으로 target generalization이 충분하지 않은 조건
  - https://proceedings.mlr.press/v97/zhao19a.html
- Liu et al., Learning Deep Kernels for Non-Parametric Two-Sample Tests: MMD 및 classifier 계열 two-sample test의 근거
  - https://proceedings.mlr.press/v119/liu20m.html
- Garcia de Marina et al., Guidance algorithm for smooth trajectory tracking of a fixed wing UAV flying in wind flows: fixed-wing path tracking과 bank-angle physical constraint 사례
  - https://arxiv.org/abs/1610.02797

## 15. 재현 파일

- `research/audits/20260904/probe_core.py`
- `research/audits/20260904/probe_batch.py`
- `research/audits/20260904/core_results.json`
- `research/audits/20260904/batch_results.json`

실행 명령:

```powershell
.\venv\Scripts\python.exe -B -X utf8 research\audits\20260904\probe_core.py
.\venv\Scripts\python.exe -B -X utf8 research\audits\20260904\probe_batch.py
```

두 프로그램은 repository source를 수정하지 않으며 진단 결과를 표준 출력으로만 보낸다.
