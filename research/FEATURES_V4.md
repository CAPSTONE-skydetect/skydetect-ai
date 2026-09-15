# Research Feature V4

Feature v4는 A의 bbox가 객체 외곽 크기를 안정적으로 나타낸다는 가정을 제거한다. 모든 분류 특징은 post-CMC 중심점과 실제 관측 시각으로 계산한다. `w`, `h`와 bbox 면적은 특징 수식에 사용하지 않는다.

## 입력과 전처리

관측점은 `p_i = (cx_i * image_width, cy_i * image_height)` 픽셀 좌표로 변환한다. `timestamp_ms`를 우선 사용하고, timestamp가 전부 없을 때만 명시적인 `fps`와 `frame_index`를 사용한다. 짧은 gap은 보간하고 긴 gap은 독립 구간으로 분리한다. 고주파 aliasing을 줄인 뒤 Savitzky-Golay smoothing을 적용한다.

```text
dt_i = t_i - t_(i-1)
velocity_i = (p_i - p_(i-1)) / dt_i
speed_i = norm(velocity_i)
```

`speed_median`과 가속도는 각각 `pixel/s`, `pixel/s^2` 단위의 apparent image-plane motion이다. 실제 물리 속도나 거리 보정 속도가 아니다.

## 특징 계약

| 특징 | 정의 | 해석 |
|---|---|---|
| `speed_median` | 속력의 시간 가중 중앙값 | 일반적인 겉보기 속도 |
| `speed_cv` | `std(speed) / mean(speed)` | 일정 배율에 불변인 속도 변동성 |
| `acceleration_median` | `abs(diff(speed)) / dt`의 시간 가중 중앙값 | 일반적인 속력 변화율 |
| `acceleration_p95` | 가속도 95백분위수 | 드문 급가속의 크기 |
| `turn_rate_median` | wrapped heading 변화량을 시간으로 나눈 값의 중앙값 | 일반적인 방향 전환율 |
| `turn_rate_p95` | turn rate 95백분위수 | 드문 급회전의 크기 |
| `curvature_cv` | `kappa = abs(delta_heading) / arc_step`의 변동계수 | 이동 거리당 굴곡의 불규칙성 |
| `tortuosity` | 전체 이동거리 / 시작-끝 직선거리 | 경로 우회 정도 |
| `heading_change_ratio` | 전체 전환 관측 시간 중 유효 turn rate가 임계값을 넘는 시간 비율 | 의미 있는 방향 변화 빈도 |

정지 또는 관측 잡음 수준 이하의 구간은 heading 계산에서 제외한다. 시작점과 끝점이 거의 같은 순환 경로의 tortuosity는 설정된 상한으로 제한한다. 모든 백분위수와 중앙값은 여러 연속 구간을 시간 가중하여 계산한다.

## 근거와 제한

- Luesutthiviboon et al. (2026), *Improving visual differentiation of drones and birds in aerial surveillance using trajectory features*: 실제 적외선 영상에서 apparent velocity, acceleration, heading change rate, tortuosity를 사용했다. https://doi.org/10.1007/s00521-026-12080-5
- Yao et al. (2026), *Non-Appearance-Based Discrimination of UAVs and Birds in Optical Remote Sensing*: 속도 CV, 곡률 CV, heading change 비율의 결합이 평균 통계 중심 구성보다 우수하다고 보고했다. 원 논문의 bbox 정규화 속도는 사용하지 않는다. https://doi.org/10.3390/drones10020098
- `acceleration_p95`와 `turn_rate_p95`는 급격한 운동을 보존하기 위한 연구 후보이며, 위 논문에서 독립적으로 검증된 특징은 아니다.

Feature v4는 거리 모호성을 해결하지 않는다. 일정한 좌표 배율에 불변인 CV·각도·tortuosity 특징과 거리 영향을 받는 apparent speed/acceleration을 함께 제공하고, 실제 A 궤적 ablation으로 채택 여부를 결정해야 한다.
