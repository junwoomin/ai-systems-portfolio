# P100a Memory Transfer Measurements and Bottleneck Hypotheses

VGG11 최적화 과정에서 측정한 메모리 이동 대역폭을 정리하고, activation spill과 operator 실행 비용을 구분하기 위한 분석 기록입니다.

**현재 근거로는 VGG의 주 병목을 확정할 수 없습니다.** 다만 측정한 tensor 조건에서는 device 내부 이동이 host 왕복보다 높은 유효 대역폭을 보였으며, 수 ms의 성능 차이를 작은 activation 한 번의 payload 이동만으로 설명하기는 어렵습니다.

## 1. 측정 출처와 범위

사용자가 P100a 환경에서 측정해 공유한 결과입니다. 이 문서를 작성하는 환경에서 하드웨어 재측정은 수행하지 않았습니다. benchmark 소스는 저장소에 추가했으며, 전체 원본 로그, software commit과 반복 측정 통계는 아직 첨부되지 않았습니다.

벤치마크 코드: [benchmarks/p100a-bandwidth](benchmarks/p100a-bandwidth/README.md)

```bash
cd benchmarks/p100a-bandwidth
python all_bandwidth.py \
    --device-id 0 \
    --l1-small-kb 16 \
    --cpu-size-mb 512 \
    --pcie-size-mb 256 \
    --l1-size-mb 8 \
    --repeat 10 \
    --warmup 2
```

| 경로 | 보고된 payload 대역폭 | 측정 범위 / 조건 |
|---|---:|---|
| CPU RAM memcpy | 5.94 GB/s | 512 MiB, PyTorch copy, CPU threads 16 |
| CPU → P100a DRAM | 7.30 GB/s | 256 MiB, TTNN host-to-device 경로 |
| P100a DRAM → CPU | 1.92 GB/s | 256 MiB, TTNN device-to-host 경로 |
| P100a L1 → DRAM | 119.2 GB/s | 8 MiB, HEIGHT_SHARDED → INTERLEAVED |
| P100a DRAM → L1 | 45.1 GB/s | 8 MiB, INTERLEAVED → HEIGHT_SHARDED |

단위는 GB/s = 10⁹ byte/s, MiB = 2²⁰ byte입니다. CPU copy의 read+write traffic 환산치는 약 11.89 GB/s이며, 다른 행의 payload 대역폭과 혼용하지 않습니다.

Device 경로에서 보고된 shard 구성은 110 cores, shard shape [596, 64]입니다. 이 구성의 값을 다른 activation shape나 core 배치에 그대로 적용할 수는 없습니다.

## 2. 대역폭이 의미하는 것

위 값은 해당 API와 tensor 구성에서 측정한 **유효 payload 대역폭**입니다. GDDR의 raw aggregate bandwidth나 PCIe 링크 자체의 최대 대역폭 측정값이 아닙니다.

L1 ↔ DRAM 경로는 저장 위치뿐 아니라 sharded/interleaved 배치도 달라집니다. API dispatch, allocation, NoC 이동, synchronization 중 어떤 비용이 시간 측정에 포함되는지는 benchmark 코드의 측정 구간을 확인해야 합니다.

관측된 L1 → DRAM 대역폭은 반대 방향의 약 2.64배입니다. 이는 두 API 경로의 비대칭이며, GDDR read 자체가 write보다 2.64배 느리다는 증거는 아닙니다. 분배 방식, access pattern, program 구성 등의 기여도는 미확인입니다.

Host 경로도 H2D가 D2H보다 약 3.80배 높습니다. host tensor 생성·복사, DMA 및 동기화 범위를 확인하기 전에는 PCIe 또는 IOMMU를 원인으로 지정할 수 없습니다. IOMMU 활성화 자체도 성능 저하의 증거는 아닙니다.

CPU의 5.94 GB/s 역시 해당 PyTorch copy 조건의 결과입니다. RAM 하드웨어 성능 판단에는 별도 STREAM 측정과 thread 수·NUMA·메모리 채널 조건 확인이 필요합니다.

## 3. Activation 한 번의 spill 비용 추정

예시 activation: batch 8, 28 × 28, channels 256, BF16.

```text
8 × 28 × 28 × 256 × 2 = 3,211,264 bytes = 3.0625 MiB
```

위 대역폭이 이 tensor에도 동일하게 유지된다고 가정한 단순 환산:

| 경로 | 계산 | 추정 시간 |
|---|---|---:|
| L1 → DRAM | 3,211,264 / 119.2e9 | 0.0269 ms |
| DRAM → L1 | 3,211,264 / 45.1e9 | 0.0712 ms |
| 왕복 | 두 시간의 합 | 약 0.0982 ms |

8 MiB L1 → DRAM 이동도 같은 방식으로 약 0.0704 ms에 해당합니다.

이 계산은 **다른 크기의 tensor에 측정 대역폭을 적용한 추정**입니다. 작은 전송의 고정 비용, padding, shard 구성, 반복 횟수와 메모리 경합을 반영한 실측값이 아닙니다. 따라서 실제 VGG 전체의 spill 비용이 0.1 ms라고 결론 내릴 수 없습니다.

확인할 수 있는 범위는 “이 예시 tensor 한 번의 payload 왕복만으로 5~10 ms 차이를 설명하기 어렵다”까지입니다.

## 4. 기존 VGG 결과와 혼동하지 않을 것

| 기록된 실험 | 관측값 | 해석 범위 |
|---|---|---|
| 24 KiB, Conv/Pool config 모두 L1_SMALL | 약 35 ms | config tensor 배치 조건 |
| 24 KiB, Conv config DRAM / Pool config L1_SMALL | 약 28 ms | 위 조건에서 Conv config 위치 변경 |
| 48 KiB, 두 config 모두 L1_SMALL | 약 28 ms | L1_SMALL 예약 크기 변경 |
| 48 KiB + prepared weight/bias 재사용 | 약 21 ms | 준비된 tensor 재사용 추가 |
| 위 조건 + act_block_h_override=0 | 약 18 ms | activation block height override 변경 |

**35 → 28 ms는 activation의 DRAM spill을 제거한 실험 결과가 아닙니다.** Configuration tensor가 사용하는 L1_SMALL과 activation residency는 구분해야 합니다.

기존 문서의 “all-L1”은 Conv/Pool의 **config tensor 두 종류가 모두 L1_SMALL**이라는 의미입니다. 모든 activation이 L1에 유지된다는 뜻이 아닙니다. 현재 저장소 코드는 MaxPool 이후 DRAM memory configuration 변환을 포함합니다.

따라서 activation spill 제거 또는 sharded L1 chain 유지의 성능 효과는 별도 비교 실험으로 검증해야 합니다.

## 5. 병목 가설

| 가설 | 현재 근거 | 필요한 검증 |
|---|---|---|
| 반복 weight/bias 준비 비용이 큼 | prepared tensor 재사용 후 28 → 21 ms 보고 | 준비 관련 host/device 시간 및 출력 비교 |
| 고정 block height가 비효율적임 | override 32 → 0에서 21 → 18 ms 보고 | 실제 선택값, layer별 kernel 시간 |
| Spill 경계에서 다음 operator 비용도 바뀜 | memory configuration과 sharding이 바뀌는 경로 | 변환 단독 및 다음 Conv를 포함한 경계 전체 비교 |
| L1_SMALL 배치가 실행 계획에 영향을 줌 | 24/48/96 KiB에서 latency 차이 | allocator 상태와 program configuration |
| Host 왕복이 전체 시간에 기여함 | device 내부 이동보다 낮은 유효 대역폭 | 실제 forward 측정 구간의 H2D/D2H 호출과 시간 |

Spill 경계에서는 conversion 외에 reshard, halo 준비, allocation 또는 program 선택이 달라질 가능성이 있습니다. 현재 로그만으로 이 단계들이 항상 실행된다거나 주원인이라고 단정할 수 없습니다.

L1에 tensor를 유지하는 것은 memory residency 최적화입니다. 여러 operator를 하나의 kernel로 결합했다는 증거가 없으므로 operator fusion으로 표현하지 않습니다. 또한 같은 HEIGHT_SHARDED 명칭이어도 core grid와 shard shape가 달라지면 추가 변환이 필요할 수 있습니다.

## 6. 다음 실험: Pool → Conv 경계 비교

현재 18 ms 구성을 기준으로 batch, input, dtype, weight, L1_SMALL 크기, program cache와 warm-up 조건을 고정합니다. 각 조건은 별도 process에서 반복합니다.

| 조건 | 변경 사항 | 측정 항목 |
|---|---|---|
| A: 현재 경로 | Pool 이후 DRAM 변환 유지 | Pool, conversion, 다음 Conv, forward 전체 |
| B: L1 유지 후보 | 한 경계의 DRAM 변환만 제거 | 동일 항목과 실제 입출력 memory/shard config |
| C: 선택적 spill 후보 | OOM 또는 layout 제약이 확인된 경계에만 변환 유지 | 성공 여부, peak L1 사용량, forward 전체 |

conv3 → conv4는 조사할 경계 후보입니다. 정확한 OOM 로그와 allocation 상태가 확보되기 전에는 그 경계가 반드시 spill이 필요한 지점이라고 확정하지 않습니다.

기록할 값:

- 첫 호출과 warm-up 이후 평균, p50, p95, 반복 간 분산
- layer 이름, input/output shape, dtype, memory layout, core grid, shard shape
- conversion 시간과 다음 Conv까지 포함한 경계 전체 시간
- host wall time과 device kernel time
- 출력 수치 오차와 정확도

레이어마다 synchronize하면 overlap과 실행 흐름이 바뀔 수 있습니다. 진단용 개별 측정의 합과 원래 forward 전체 시간을 따로 기록합니다. Profiler에서 서로 겹치는 구간을 단순 합산하지 않습니다.

## 결론

이번 microbenchmark는 **작은 activation 한 번의 이동 대역폭만으로 VGG의 수 ms 차이를 설명하기 어렵다는 정량적 근거**입니다. P100a 내부 메모리가 모든 조건에서 충분히 빠르다거나, DRAM bandwidth가 주 병목이 아니라고 확정하는 자료는 아닙니다.

다음 최적화 판단은 실제 레이어 경계에서 conversion과 다음 operator 비용을 함께 측정한 결과에 근거합니다. L1 유지와 선택적 spill은 검증할 후보이며, 현재 확인된 최저 순전파 결과는 기존 18 ms/batch입니다.

관련 기록: [프로젝트 개요](README.md) · [L1_SMALL 및 Conv 설정 실험](L1_SMALL_TUNING.md)
