# Config Tensor Placement and L1_SMALL Tuning

TT-NN Conv2D와 MaxPool2D의 configuration tensor 저장 위치 및 `l1_small_size`가 VGG11 순전파 latency에 미치는 영향을 기록한다.

## 설정의 의미

Conv2D는 다음 옵션을 사용한다.

```python
config_tensors_in_dram = True
```

MaxPool2D는 다음 옵션을 사용한다.

```python
config_tensor_in_dram = True
```

`True`는 해당 configuration tensor를 DRAM에 배치한다. `False`는 L1_SMALL에 배치한다. 이 옵션은 일반 activation tensor의 DRAM/L1 배치와 별개이다.

## 관측 결과

각 결과는 실험 이력이며, 모든 행이 완전히 동일한 실행 세션에서 측정된 것은 아니다. 아래 결과는 입력 변환과 host-to-device 전송이 `FrozenVGG11.__call__()` 내부에 있던 이전 측정 경계에서 얻은 값이다. 이후의 10 ms 이하 device-only forward 결과와 직접 비교하지 않는다.

| `l1_small_size` | Conv config | MaxPool config | Prepared weight/bias | Block height | Latency |
|---:|---|---|---|---|---:|
| 24 KiB | L1_SMALL | L1_SMALL | 미적용 | 32 강제 | 약 35 ms |
| 24 KiB | DRAM | L1_SMALL | 미적용 | 32 강제 | 약 28 ms |
| 48 KiB | L1_SMALL | L1_SMALL | 미적용 | 32 강제 | 약 28 ms |
| 96 KiB | L1_SMALL | L1_SMALL | 미적용 | 32 강제 | 약 29 ms |
| 48 KiB | L1_SMALL | L1_SMALL | 적용 | 32 강제 | 약 21 ms |
| 48 KiB | L1_SMALL | L1_SMALL | 적용 | 자동 선택 (`0`) | 약 18 ms |
| 기본값 0 | DRAM | DRAM | 적용 | 자동 선택 (`0`) | **약 18 ms** |

## 해석

24 KiB에서 두 configuration tensor를 모두 L1_SMALL에 배치했을 때 약 35 ms가 관측되었다. Conv config만 DRAM으로 이동하거나 L1_SMALL을 48 KiB로 늘리면 약 28 ms로 회복되었다. 이는 24 KiB 조건에서 configuration tensor allocation pressure 또는 다른 program configuration이 발생했을 가능성과 일치한다.

96 KiB는 48 KiB보다 빠르지 않았다. L1_SMALL은 각 Tensix L1의 일부를 예약하므로, 과도한 예약은 Circular Buffer와 working buffer에 사용할 일반 L1 공간을 줄일 수 있다.

그러나 prepared tensor 재사용과 block height 자동 선택을 적용한 상태에서는 다음 두 구성이 모두 약 18 ms를 기록하였다.

```text
48 KiB + Conv/Pool config L1_SMALL → 약 18 ms
기본 0 + Conv/Pool config DRAM     → 약 18 ms
```

따라서 현재 측정 정밀도에서는 config tensor를 L1_SMALL에 배치하는 추가 성능 이점이 확인되지 않았다. 현재 코드는 예약 공간과 관리 복잡도를 줄이기 위해 기본 L1_SMALL 크기와 DRAM config tensor를 사용한다.

“반복적인 DRAM → L1 이동이 병목이었다”는 설명은 가능한 가설이다. 현재 latency 비교만으로 실제 이동 횟수나 비용을 확정할 수 없으므로 TTNN profiler의 data-movement 및 allocator 기록이 필요하다.

## 현재 코드

```python
context.open_device(
    device_ids=[DEVICE_ID],
)

conv_config = ttnn.Conv2dConfig(
    config_tensors_in_dram=True,
    act_block_h_override=0,
)

ttnn.max_pool2d(
    config_tensor_in_dram=True,
)
```

## 성능 개선 기여도

| 변경 | 관측 변화 | 해석 |
|---|---:|---|
| Prepared weight/bias 재사용 | 약 28 → 21 ms | 반복적인 weight/bias 준비 비용 감소 |
| Block height 32 → 자동 선택 | 약 21 → 18 ms | 현재 tensor shape와 sharding에 적합한 실행 구성 선택 |
| 48 KiB all-L1 config → 기본 0 및 DRAM config | 약 18 → 18 ms | L1_SMALL 없이 동일한 최저 latency 유지 |

## 입력 경로 분리 이후 결과

입력의 NCHW → NHWC 변환, contiguous 복사, BF16 변환 및 P100a DRAM 전송을 forward 타이머 앞으로 이동한 뒤 device-resident forward는 약 10 ms/batch로 측정되었다.

```text
이전 측정: 입력 준비 + H2D + device forward ≈ 18 ms
현재 측정: device forward only                 ≈ 10 ms
관측 차이                                    ≈  8 ms
```

이는 L1_SMALL 변경에 따른 추가 개선이 아니라 측정 범위의 정정이다. 이후 Conv1 block tuning으로 약 8.3 ms, 일반 L1의 HEIGHT_SHARDED activation 배치로 약 7.3 ms가 측정되었다. 여기서 일반 L1 activation과 configuration tensor용 L1_SMALL은 서로 다른 설정이다. L1_SMALL 실험을 재평가하려면 모든 조건에서 동일하게 입력 TTNN tensor를 미리 P100a DRAM에 배치한 뒤 device forward만 측정해야 한다.

## 추가 검증

- [ ] 동일 process에서 DRAM/DRAM과 L1/L1을 번갈아 측정
- [ ] 각 조건을 별도 process에서 반복 실행
- [ ] 최소 100개 batch의 평균, p50, p95 및 표준편차 기록
- [ ] 실제 configuration tensor allocation과 이동 횟수 확인
- [ ] block height 자동 선택값 기록
- [ ] 출력 수치 및 정확도 일치 확인
- [ ] 일반 L1과 L1_SMALL 사용량 기록
