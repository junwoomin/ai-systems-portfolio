# P100a Memory Transfer Measurements and Bottleneck Hypotheses

VGG11 최적화 과정에서 측정한 메모리 이동 대역폭과 병목 가설을 기록한다. 입력 변환과 host-to-device 전송이 기존 forward 측정 구간에 포함되어 있었음이 확인되었으며, 이를 분리한 뒤 약 10 ms/batch가 측정되었고, Conv1 block 및 shard tuning 후 약 7.3 ms/batch까지 감소하였다.

## 측정 코드

[P100a Bandwidth Benchmark](benchmarks/p100a-bandwidth/README.md)

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

## 측정 결과

| 경로 | 보고된 payload 대역폭 | 측정 조건 |
|---|---:|---|
| CPU RAM memcpy | 5.94 GB/s | 512 MiB, PyTorch copy |
| CPU → P100a DRAM | 7.30 GB/s | 256 MiB, TTNN host-to-device |
| P100a DRAM → CPU | 1.92 GB/s | 256 MiB, TTNN device-to-host |
| P100a L1 → DRAM | 119.2 GB/s | 8 MiB, HEIGHT_SHARDED → INTERLEAVED |
| P100a DRAM → L1 | 45.1 GB/s | 8 MiB, INTERLEAVED → HEIGHT_SHARDED |

이는 API, allocation, memory configuration 변환과 synchronization을 포함한 application-level effective bandwidth이다. GDDR6 또는 PCIe의 raw peak bandwidth가 아니다.

L1 → DRAM과 DRAM → L1의 차이는 GDDR read/write 자체의 비대칭을 의미하지 않는다. DRAM → L1 측정에는 interleaved tensor를 여러 core의 HEIGHT_SHARDED L1으로 분배하는 경로가 포함된다.

## Activation spill 비용 추정

예시 activation은 batch 8, 28 × 28, channels 256, BF16이다.

```text
8 × 28 × 28 × 256 × 2
= 3,211,264 bytes
= 3.0625 MiB
```

측정 대역폭을 그대로 적용한 단순 추정치는 다음과 같다.

| 경로 | 추정 시간 |
|---|---:|
| L1 → DRAM | 약 0.0269 ms |
| DRAM → L1 | 약 0.0712 ms |
| 왕복 | 약 0.0982 ms |

이 값은 다른 tensor shape와 shard configuration에 측정 대역폭을 적용한 계산값이다. 실제 VGG layer latency가 아니며, 고정 dispatch 비용과 program 선택 변화도 포함하지 않는다.

## Config tensor와 activation의 구분

`config_tensors_in_dram` 및 `config_tensor_in_dram`은 Conv와 MaxPool의 configuration tensor 저장 위치를 제어한다. 일반 activation의 memory configuration과는 다른 설정이다.

다음 35 → 28 ms 변화는 activation spill 제거 결과가 아니다.

```text
24 KiB, Conv/Pool config L1_SMALL → 약 35 ms
24 KiB, Conv config DRAM          → 약 28 ms
```

현재 최적 구성은 다음과 같다.

```text
l1_small_size              : 기본값 0
Conv config                : DRAM
MaxPool config             : DRAM
prepared weight/bias cache : 적용
Conv1 act_block_h_override : 256
Conv2–Conv8 block height   : 자동 선택
Conv1 activation           : L1, HEIGHT_SHARDED
입력 위치                  : P100a DRAM
device-only forward        : 약 7.3 ms/batch
```

이전 약 18 ms에는 NCHW → NHWC 변환, contiguous host copy, BF16 변환 및 CPU → P100a DRAM 전송이 포함되어 있었다. 해당 작업을 타이머 앞으로 이동하자 약 10 ms가 측정되었다. 관측된 약 8 ms는 host-side 입력 준비와 H2D 경로의 합이며, 순수 PCIe 전송 시간이나 커널 개선량으로 단정할 수 없다. 이후 동일한 device-only 범위에서 Conv1 block tuning으로 약 8.3 ms, Conv1 L1 HEIGHT_SHARDED 배치로 약 7.3 ms가 측정되었다.

48 KiB에서 두 config tensor를 L1_SMALL에 둔 구성도 이전 측정 경계에서 약 18 ms였으므로, 현재 데이터는 L1_SMALL config 배치의 추가 성능 이점을 보여주지 않는다.

## 현재 병목 가설

| 가설 | 현재 근거 | 필요한 검증 |
|---|---|---|
| 반복 weight/bias 준비 비용 | prepared tensor 재사용 후 28 → 21 ms | 첫 호출과 steady-state profiler 비교 |
| 고정 block height 비효율 | override 32 → 0에서 21 → 18 ms | 실제 선택값과 layer별 kernel 시간 |
| 작은 L1_SMALL 예약의 압력 | 24 KiB all-L1 config에서 35 ms | allocator와 program configuration 비교 |
| Config tensor 이동·관리 오버헤드 | DRAM config에서도 18 ms 유지 | 실제 allocation 및 이동 이벤트 확인 |
| Conv1 activation 배치 | L1 HEIGHT_SHARDED 적용 후 8.3 → 7.3 ms | Conv1과 MaxPool1을 분리 프로파일링 |
| Activation spill 경계의 후속 비용 | sharded/interleaved 변환 경로 존재 | conversion과 다음 Conv를 포함한 경계 측정 |
| Host 입력 준비 및 H2D 비용 | 측정 경계 분리 후 18 → 10 ms | permute·cast·H2D 각각 개별 측정 |

현재 결과는 “DRAM이 L1보다 빠르다”는 결론을 지지하지 않는다. Configuration tensor의 배치와 activation의 residency를 구분해야 한다.

## 다음 실험

현재 7.3 ms device-only 구성을 기준으로 batch, input, dtype, weight, program cache와 warm-up 조건을 고정한다.

- DRAM/DRAM config와 L1/L1 config를 동일 process에서 반복 비교
- configuration tensor allocation과 data movement event 기록
- Pool → memory conversion → 다음 Conv 구간 측정
- 첫 호출과 steady-state latency 분리
- 평균, p50, p95 및 반복 간 분산 기록
- 출력 수치와 정확도 확인

관련 문서: [프로젝트 개요](README.md) · [Config Tensor Placement and L1_SMALL Tuning](L1_SMALL_TUNING.md)
