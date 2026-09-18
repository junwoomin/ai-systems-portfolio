# Layer-wise Profiling, Conv1 Block and Shard Tuning

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

이에 따라 레이어별 시간의 합계 12.74 ms는 단일 synchronization으로 측정한 전체 forward latency와 직접 비교할 수 없다. 전체 forward 약 6.6 ms를 보고할 때는 레이어 프로파일 합계를 더하지 않고 별도의 진단 지표로 취급한다.

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
        act_block_h_override=128,
        enable_act_double_buffer=False,
        enable_weights_double_buffer=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        output_layout=ACTIVATION_LAYOUT,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ),
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
```

Conv2–Conv8은 자동 block 선택을 유지한다. HEIGHT_SHARDED는 Conv1, Conv2 및 Conv4에 적용하고, 나머지 Conv는 기본 shard 선택을 유지한다.

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
| Conv1 block height 256, 나머지 자동 선택 | 약 8.3 ms/batch |
| Conv1 block 256 + Conv1 L1 HEIGHT_SHARDED | 약 7.3 ms/batch |
| Conv1 block 128 + Conv1·2·4 HEIGHT_SHARDED | **약 6.6 ms/batch** |

마지막 단계에서는 Conv1 block height를 256에서 128로 변경하는 동시에 HEIGHT_SHARDED 적용 범위를 Conv1에서 Conv1·Conv2·Conv4로 확대하였다. 동일한 측정 경계에서 약 0.7 ms, 약 9.6%를 추가로 줄였다. 자동 block 구성의 약 10.0 ms와 비교하면 총 3.4 ms, 약 34% 감소이다.

두 설정을 동시에 변경했으므로 7.3 → 6.6 ms를 block height 128 또는 추가 sharding 중 하나의 단독 효과로 해석할 수 없다. 이 변화는 입력 준비 및 host-to-device 전송을 타이머 밖에 둔 상태에서 측정되었으므로 이전의 18 → 10 ms 측정 경계 정정과도 구분된다.

## HEIGHT_SHARDED의 효과

Conv1 입력은 큰 spatial dimension과 낮은 channel 수를 가진다. HEIGHT_SHARDED는 flattened activation의 height 방향을 여러 Tensix core에 나누고 각 shard를 해당 core의 L1에 배치한다. Conv1뿐 아니라 초기·중간 고해상도 구간의 Conv2와 Conv4에도 같은 shard 방향을 명시함으로써 자동 reshard 또는 interleaved tensor 재분배 비용을 줄일 가능성이 있다.

여기서 `ttnn.L1_MEMORY_CONFIG`는 일반 activation용 L1 배치이며 `L1_SMALL`과는 다른 영역이다. `shard_layout`은 분할 방향을 지정하고, `memory_config`은 결과 tensor의 저장 계층을 지정한다.

현재 결과는 레이어별 block 및 shard configuration이 전체 모델에 동일한 configuration을 적용하는 것보다 효과적임을 보여준다. 다만 마지막 약 0.7 ms 감소에서 Conv1 block height 변경과 Conv2·Conv4 sharding 확대가 각각 얼마나 기여했는지는 후속 ablation이 필요하다.

## 다음 검증

- [ ] L1 + HEIGHT_SHARDED 변경 후 Conv1–MaxPool1 구간을 다시 프로파일링
- [ ] Conv1 `act_block_h_override` 64, 128, 192, 256 및 자동값 비교
- [ ] 각 조건을 최소 100회 반복하고 평균, p50, p95 기록
- [ ] 출력 tensor의 수치와 분류 정확도 일치 확인
- [ ] Conv1 입력·출력 memory layout과 shard specification 기록
- [ ] L1/HEIGHT_SHARDED 각각을 독립적으로 분리 측정
- [ ] Conv1만, Conv1·2, Conv1·2·4 HEIGHT_SHARDED 조건 비교
- [ ] `reshard_if_not_optimal` 및 `deallocate_activation`을 독립적으로 분리 측정
