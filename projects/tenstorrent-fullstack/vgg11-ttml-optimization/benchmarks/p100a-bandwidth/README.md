# P100a Bandwidth Benchmark

P100a VGG11 병목 분석에 사용한 effective bandwidth microbenchmark입니다.

## 측정 경로

- CPU RAM memcpy
- CPU host RAM → P100a DRAM
- P100a DRAM → CPU host RAM
- P100a sharded L1 → DRAM interleaved
- P100a DRAM interleaved → sharded L1

측정값은 TTNN API, allocation, memory configuration 변환과 synchronization을 포함하는 wall-clock 기반 payload 대역폭입니다. GDDR6 또는 PCIe의 raw peak bandwidth가 아닙니다.

## 전체 실행

```bash
python all_bandwidth.py \
    --device-id 0 \
    --l1-small-kb 16 \
    --cpu-size-mb 512 \
    --pcie-size-mb 256 \
    --l1-size-mb 8 \
    --repeat 10 \
    --warmup 2
```

## 개별 실행

```bash
python cpu_ddr4_bandwidth.py --size-mb 512 --repeat 20 --warmup 3

python host_device_bandwidth.py \
    --device-id 0 \
    --size-mb 256 \
    --repeat 10 \
    --warmup 2 \
    --l1-small-kb 16

python l1_dram_bandwidth.py \
    --device-id 0 \
    --size-mb 8 \
    --channels 64 \
    --repeat 20 \
    --warmup 3 \
    --l1-small-kb 16
```

## 결과 해석

- GB/s는 logical payload bytes / elapsed wall-clock time입니다.
- CPU memcpy는 payload와 read+write traffic을 함께 출력합니다.
- L1 ↔ DRAM 테스트는 HEIGHT_SHARDED와 INTERLEAVED 변환을 포함합니다.
- 크기, channel 수, core grid와 shard shape가 달라지면 결과도 달라질 수 있습니다.
- 장치 테스트는 P100a에서 실행해야 하며, 이 저장소에 추가할 때는 Python 문법 검사만 수행했습니다.

분석: [Memory Transfer Measurements and Bottleneck Hypotheses](../../MEMORY_BOTTLENECK_EVIDENCE.md)
