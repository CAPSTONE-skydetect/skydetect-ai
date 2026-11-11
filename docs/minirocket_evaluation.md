# MiniRocket + Ridge 개발 비교 (C)

- 패키지 manifest SHA-256: `d5dbdd50a28f4113ca751cf12758f92788bc822f62eef5847f2c73ee54f50f8c`
- 입력 계약: `trajectory-sequence-1.0.1` / `eb9be8154ee0f404`
- validation: 창 59개, 원본 영상 그룹 10개
- validation 은 반복 사용된 개발 자료다. 최종 일반화 성능이 아니다.
- 점수는 Ridge margin 이며 확률이 아니다.

| 학습 구성 | 변환기 fit | alpha | 학습 창/그룹 | 창 macro-F1 | track macro-F1 | 그룹 macro-F1 (95% CI) | 새 recall | 드론 recall |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| real_only | real_only | 0.1 | 156/29 | 0.8101 | 0.8167 | 0.8990 (0.67–1.00) | 0.8333 | 1.0000 |
| real_plus_augmentation | real_only | 0.1 | 6156/29 | 0.8625 | 0.9060 | 0.8990 (0.67–1.00) | 0.8333 | 1.0000 |
| synthetic_only | synthetic_only | 100 | 587/104 | 0.5683 | 0.5299 | 0.5833 (0.29–0.89) | 0.6667 | 0.5000 |

## 그룹 단위 오분류

- real_only: group-fba48c3bb553e271(bird, +0.851)
- real_plus_augmentation: group-fba48c3bb553e271(bird, +0.419)
- synthetic_only: group-bc9f60f39f587180(bird, +0.759), group-c29305e39602da00(drone, -0.006), group-f772bea7233d63d4(drone, -0.764), group-fba48c3bb553e271(bird, +1.199)

## 추론 지연

- real_only: 첫 호출 1.8 ms, 준비 후 창당 0.549 ms
- real_plus_augmentation: 첫 호출 1.7 ms, 준비 후 창당 0.588 ms
- synthetic_only: 첫 호출 1.9 ms, 준비 후 창당 0.577 ms
