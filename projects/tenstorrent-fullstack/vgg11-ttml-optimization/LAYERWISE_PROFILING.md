# Layer-wise Profiling and Conv1 Block Tuning

P100a에서 VGG11의 device-resident forward를 레이어 단위로 동기화하여 병목을 확인하고, 레이어별 Conv2D configuration을 조정한 실험을 기록한다.

## 레이어별 측정 결과

다음 값은 Conv1 전용 block tuning 이전의 프로파일 스냅샷이다.

| 레이어 | 평균 latency |
|---|---:|
| Conv1 | 3.94 ms |
| MaxPool1 | 1.67 ms |
| Conv2 | 0.93 ms |
| MaxPool2 | 0.53 ms |
| Conv3 | 0.55 ms |
| Conv4 | 1.69 ms |
| MaxPool3 | 0.39 ms |
| Conv5 | 0.67 ms |
| Conv6 | 0.88 ms |
| MaxPool4 | 0.28 ms |
| Conv7 | 0.43 ms |
| Conv8 | 0.41 ms |
| MaxPool5 | 0.24 ms |
| Reshape | 0.06 ms |
| To tile layout | 0.07 ms |
| **동기화된 레이어 시간 합계** | **12.74 ms** |

Conv1은 동기화된 레이어 시간 합계의 약 30.9%를 차지하여 가장 큰 단일 병목으로 나타났다. 다음으로 Conv4 약 13.3%, MaxPool1 약 13.1% 순이다.

## 전체 forward와 합계가 다른 이유

레이어별 측정에서는 각 연산 직후 device synchronization을 수행한다. 따라서 각 항목에는 개별 synchronization과 host-side timer 호출 비용이 반복해서 포함되고, 비동기 dispatch가 연속되는 정상 실행 경로도 끊긴다.

이에 따라 레이어별 시간의 합계 12.74 ms는 단일 synchronization으로 측정한 전체 forward latency와 직접 비교할 수 없다. 전체 forward 약 8.3 ms를 보고할 때는 레이어 프로파일 합계를 더하지 않고 별도의 진단 지표로 취급한다.

## Conv1 병목 해석

Conv1의 입력은 224 × 224의 큰 spatial dimension을 가지지만 입력 채널은 3개뿐이다. 이후 레이어보다 행렬 연산의 채널 병렬성이 낮고 activation 및 tiling 비용의 비중이 크기 때문에, 자동 block 선택이 항상 최적이라고 보장할 수 없다.

Conv1에만 다음 설정을 적용하였다.

```python
self.conv1 = TTConv2d(
    3,
    64,
    device,
    conv_config=ttnn.Conv2dConfig(
        weights_dtype=WEIGHT_DTYPE,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        act_block_h_override=256,
        enable_act_double_buffer=False,
        enable_weights_double_buffer=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        output_layout=ACTIVATION_LAYOUT,
    ),
)
```

Conv2–Conv8은 자동 block 선택과 현재 sharding을 유지한다.

```python
ttnn.Conv2dConfig(
    weights_dtype=WEIGHT_DTYPE,
    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
    act_block_h_override=0,
    enable_act_double_buffer=False,
    enable_weights_double_buffer=False,
    reshard_if_not_optimal=False,
    deallocate_activation=False,
    output_layout=ACTIVATION_LAYOUT,
)
```

## 결과

| 구성 | Device-only forward |
|---|---:|
| 모든 Conv block height 자동 선택 | 약 10.0 ms/batch |
| Conv1 block height 256, 나머지 자동 선택 | **약 8.3 ms/batch** |

동일한 측정 경계에서 약 1.7 ms, 약 17% 감소하였다. 이 변화는 입력 준비 및 host-to-device 전송을 타이머 밖에 둔 상태에서 측정되었으므로 이전의 18 → 10 ms 측정 경계 정정과 구분된다.

현재 결과는 Conv1에 대한 layer-specific configuration이 전체 모델에 동일한 configuration을 적용하는 것보다 효과적임을 보여준다. 다만 1.7 ms가 전부 Conv1 kernel 시간 감소인지, 이후 MaxPool1 또는 sharding 경로의 변화도 포함하는지는 후속 프로파일이 필요하다.

## 다음 검증

- [ ] 변경 후 Conv1–MaxPool1 구간을 다시 프로파일링
- [ ] Conv1 `act_block_h_override` 128, 192, 256 및 자동값 비교
- [ ] 각 조건을 최소 100회 반복하고 평균, p50, p95 기록
- [ ] 출력 tensor의 수치와 분류 정확도 일치 확인
- [ ] Conv1 출력 memory layout과 shard specification 기록
- [ ] `reshard_if_not_optimal` 및 `deallocate_activation`을 독립적으로 분리 측정
