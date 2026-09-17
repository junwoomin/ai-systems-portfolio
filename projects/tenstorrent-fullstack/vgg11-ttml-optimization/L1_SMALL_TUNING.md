# L1_SMALL Size Tuning Experiment

TT-NN의 `Conv2D`와 `MaxPool2D`에서 configuration tensor 저장 위치, `l1_small_size`, prepared weight/bias 재사용이 VGG11 순전파 latency에 미치는 영향을 확인한 실험입니다.

## Configuration

Conv2D:

```python
config_tensors_in_dram = False
```

MaxPool2D:

```python
config_tensor_in_dram = False
```

`False`로 설정하면 해당 연산의 configuration tensor를 DRAM이 아닌 L1_SMALL에 배치합니다.

## Results

| `l1_small_size` | Conv2D config | MaxPool config | Prepared weight/bias cache | Latency |
|---:|---|---|---|---:|
| 24 KiB | L1_SMALL | L1_SMALL | 미적용 | 약 35 ms |
| 24 KiB | DRAM | L1_SMALL | 미적용 | 약 28 ms |
| 48 KiB | L1_SMALL | L1_SMALL | 미적용 | 약 28 ms |
| 96 KiB | L1_SMALL | L1_SMALL | 미적용 | 약 29 ms |
| 48 KiB | L1_SMALL | L1_SMALL | 적용, block height 32 강제 | 약 21 ms |
| 48 KiB | L1_SMALL | L1_SMALL | 적용, block height 자동 선택 (`0`) | **약 18 ms** |

21 ms 행은 prepared weight/bias 재사용을 추가한 결과이며, 마지막 행은 `act_block_h_override`를 32에서 0으로 바꾼 최신 측정입니다. 앞의 L1_SMALL 크기 비교와 최적화 조건이 다르므로 하나의 L1 크기 ablation으로 해석하면 안 됩니다.

## L1_SMALL 크기 관찰

24 KiB에서 두 configuration tensor를 모두 L1_SMALL에 배치하면 약 35 ms였습니다. 같은 크기에서 Conv2D configuration tensor를 DRAM으로 이동하면 약 28 ms로 감소했습니다.

이는 DRAM이 L1_SMALL보다 빠르다는 뜻이 아니라, 24 KiB에 두 종류의 configuration tensor를 동시에 배치했을 때 allocation pressure, placement 변화 또는 다른 실행 계획이 영향을 주었을 가능성을 보여줍니다.

두 configuration tensor를 모두 L1_SMALL에 둔 상태에서 예약 크기를 48 KiB로 늘리면 약 28 ms로 회복했습니다. 96 KiB에서는 약 29 ms로 추가 개선이 없었습니다.

가능한 가설:

- 24 KiB는 현재 workload의 configuration tensor에 부족함
- 48 KiB는 configuration tensor를 수용하면서 일반 L1 공간도 유지함
- 96 KiB는 Circular Buffer, activation 및 runtime working buffer용 일반 L1을 줄일 수 있음
- 48 KiB와 96 KiB의 1 ms 차이는 측정 변동일 수 있음

## Prepared weight/bias 재사용

최신 코드는 각 `TTConv2d` 인스턴스가 첫 호출에서 TTNN이 준비한 weight와 bias를 저장합니다.

```python
if not self.weights_prepared:
    x, output_dim, prepared = ttnn.conv2d(
        **common_args,
        return_weights_and_bias=True,
    )
    self.weight, self.bias = prepared
    self.weights_prepared = True
else:
    x, output_dim = ttnn.conv2d(
        **common_args,
        return_weights_and_bias=False,
    )
```

실행 흐름:

```text
첫 호출
  → weight/bias 준비
  → prepared tensor 저장

이후 호출
  → 저장된 prepared tensor 재사용
  → 반복 준비 단계 생략
```

48 KiB에서 Conv2D와 MaxPool config를 모두 L1_SMALL에 둔 조건은 캐시 적용 전 약 28 ms, 적용 후 약 21 ms였습니다.

```text
48 KiB + all-L1 + cache 미적용 → 약 28 ms
48 KiB + all-L1 + cache 적용   → 약 21 ms
```

prepared tensor 재사용으로 28 ms → 21 ms, 약 **25% 감소**했습니다. 이후 `act_block_h_override=0`에서 21 ms → 18 ms, 약 **14.3% 추가 감소**했습니다. 첫 호출에는 준비 비용이 남아 있으므로 두 값 모두 warm-up 이후 steady-state 순전파 결과로 기록합니다.

## Activation block height 자동 선택

기존에는 `act_block_h_override=32`로 activation block height를 강제했습니다. 이를 0으로 변경하면 TTNN이 workload와 sharding 조건에 맞는 값을 자동 선택합니다.

```python
act_block_h_override = 0
```

현재 관측에서는 강제값 32보다 자동 선택이 빨랐습니다. 실제로 선택된 block height와 개선 원인은 profiler 및 program configuration을 통해 추가 확인해야 합니다.

## Current Best Configuration

```python
l1_small_size = 48 * 1024

Conv2D:
    config_tensors_in_dram = False

MaxPool2D:
    config_tensor_in_dram = False

TTConv2d:
    act_block_h_override = 0
    prepare weights and bias once
    reuse prepared tensors after the first call
```

현재 최저 관측값은 약 **18 ms/batch**입니다.

## Interpretation

최신 결과는 세 병목 후보를 구분합니다.

1. L1_SMALL이 너무 작으면 configuration tensor allocation pressure가 발생할 수 있습니다.
2. L1_SMALL이 충분해도 매 호출마다 weight/bias를 다시 준비하면 반복 비용이 남을 수 있습니다.
3. 고정된 activation block height가 현재 shape와 sharding에 최적이 아닐 수 있습니다.

48 KiB는 all-L1 구성에 필요한 공간을 제공했고, prepared tensor 캐시는 반복 준비 비용을 줄였습니다. block height 자동 선택은 추가로 3 ms를 줄였습니다. 다만 profiler 없이 내부 원인을 확정할 수는 없습니다.

## 추가 검증

- [ ] 첫 호출 latency와 steady-state latency를 분리 기록
- [ ] 각 조건을 별도 process에서 반복 실행
- [ ] 동일한 software/firmware commit 사용
- [ ] 최소 100개 batch 측정
- [ ] 평균, p50, p95 및 표준편차 기록
- [ ] program cache와 compile warm-up 조건 통일
- [ ] cache 적용 전후 출력 수치와 정확도 비교
- [ ] TTNN profiler로 weight preparation 호출과 DRAM transfer 비교
- [ ] block height 0/16/32/64 ablation 및 실제 선택값 기록
- [ ] 일반 L1과 L1_SMALL 사용량 기록

## 메모리 이동 측정과의 구분

이 문서의 `all-L1`은 Conv/Pool config tensor가 모두 L1_SMALL에 배치된다는 뜻입니다. 모든 activation의 L1 유지를 의미하지 않습니다. 35 → 28 ms를 activation spill 제거 효과로 해석하지 않습니다.

[메모리 이동 측정 및 병목 가설](MEMORY_BOTTLENECK_EVIDENCE.md)에서 유효 대역폭과 activation 왕복 비용 추정, 레이어 경계 비교 계획을 확인할 수 있습니다.
