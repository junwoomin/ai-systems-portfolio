# VGG11 TTML Runtime Optimization on P100a

Tenstorrent P100a에서 VGG11의 TTNN/TTML 순전파 병목을 분석하고, prepared weight/bias 재사용과 Conv2D 실행 설정 조정으로 latency를 줄인 프로젝트이다.

## 현재 결과

| 단계 | 주요 설정 | 순전파 latency |
|---|---|---:|
| 초기 기준 | 초기 실행 구성 | 약 28 ms/batch |
| L1_SMALL 부족 조건 | 24 KiB, Conv/Pool config 모두 L1_SMALL | 약 35 ms/batch |
| Prepared tensor 재사용 | weight/bias 준비 결과 캐시, block height 32 | 약 21 ms/batch |
| 현재 최적 구성 | 기본 L1_SMALL, Conv/Pool config DRAM, prepared tensor 재사용, block height 자동 선택 | **약 18 ms/batch** |

현재 최적 구성은 초기 28 ms 대비 약 **35.7%**, 21 ms 구성 대비 약 **14.3%** 감소하였다.

## 현재 실행 구성

TTML device는 별도의 L1_SMALL 예약 크기를 전달하지 않고 연다. 현재 사용 중인 TT-Metal branch의 기본값은 0이다.

```python
context.open_device(
    device_ids=[DEVICE_ID],
)
```

Conv2D와 MaxPool2D의 configuration tensor는 DRAM에 배치한다.

```python
ttnn.Conv2dConfig(
    config_tensors_in_dram=True,
    act_block_h_override=0,
)

ttnn.max_pool2d(
    config_tensor_in_dram=True,
)
```

각 Conv2D는 첫 호출에서 준비된 weight와 bias를 저장하고 이후 호출에서 재사용한다.

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

`act_block_h_override=0`은 block height를 0으로 만드는 설정이 아니라 강제값을 해제하고 TTNN의 자동 선택을 사용하는 설정이다.

## Config tensor 배치 해석

초기에는 Conv와 MaxPool configuration tensor를 L1_SMALL에 배치하면 빠를 것으로 예상하였다. 그러나 24 KiB에서 두 config tensor를 모두 L1_SMALL에 배치했을 때 약 35 ms가 관측되었고, 예약 크기를 늘리거나 일부 config tensor를 DRAM으로 이동하면 약 28 ms로 회복하였다.

현재는 L1_SMALL 예약을 사용하지 않고 두 config tensor를 모두 DRAM에 둔 상태에서도 약 18 ms가 유지된다. 따라서 **최종 18 ms를 L1_SMALL 적용 효과로 해석할 수 없다.** 현재 결과에서 성능 개선의 직접적인 근거가 있는 항목은 prepared weight/bias 재사용과 block height 자동 선택이다.

L1_SMALL 조건에서 발생한 저하는 configuration tensor의 반복적인 이동, 작은 예약 영역의 allocation pressure 또는 program configuration 변화와 관련될 가능성이 있다. 세 원인의 기여도는 profiler로 추가 검증해야 한다.

또한 `config_tensors_in_dram`은 configuration tensor의 저장 위치를 제어한다. 일반 activation의 L1 residency와는 별개의 설정이다.

## 측정 범위

`train_ttml.py`의 시간 측정 구간은 다음과 같다.

```text
synchronize
  → Frozen VGG11 backbone forward
  → TTML classifier forward
  → synchronize
```

18 ms는 충분한 warm-up 이후의 모델 순전파 시간이며 다음 항목을 제외한다.

- dataset loading 및 host preprocessing
- cross-entropy loss
- backward
- optimizer step
- metric 계산
- checkpoint 저장

첫 Conv 호출의 weight/bias 준비 비용도 steady-state 18 ms에 포함되지 않는다.

## 실험 환경

```text
Device       : Tenstorrent P100a
Architecture : Blackhole
Model        : VGG11
Input        : 224 × 224
Batch size   : 8
Model dtype  : bfloat16
Dataset      : Oxford-IIIT Pet
Classes      : 37
Training     : Frozen VGG11 backbone + trainable TTML classifier
Warm-up      : 10 batches
```

## 관련 문서

- [Config Tensor Placement and L1_SMALL Tuning](L1_SMALL_TUNING.md)
- [Memory Transfer Measurements and Bottleneck Hypotheses](MEMORY_BOTTLENECK_EVIDENCE.md)
- [P100a Bandwidth Benchmark](benchmarks/p100a-bandwidth/README.md)
- [TTML L1_SMALL Configuration Patch](../ttml-l1-small-config/README.md)

L1_SMALL patch는 설정을 실험할 수 있도록 TTML API를 확장한 기록이다. 현재 최적 실행에서는 `l1_small_size` 인자를 사용하지 않는다.

## 파일 구조

```text
├── L1_SMALL_TUNING.md
├── MEMORY_BOTTLENECK_EVIDENCE.md
├── benchmarks/
│   └── p100a-bandwidth/
└── src/
    ├── dataset.py
    ├── test_dataset.py
    ├── train_ttml.py
    ├── vgg.py
    └── vgg_ttml.py
```

## 실행

```bash
ttml
cd projects/tenstorrent-fullstack/vgg11-ttml-optimization/src
python train_ttml.py
```
