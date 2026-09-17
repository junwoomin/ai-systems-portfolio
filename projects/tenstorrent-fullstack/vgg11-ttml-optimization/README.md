# VGG11 TTML Optimization on P100a

Tenstorrent P100a에서 VGG11의 TTNN/TTML 순전파 병목을 분석하고, L1_SMALL 배치와 Conv2D prepared weight/bias 재사용으로 latency를 줄인 프로젝트입니다.

## 현재 결과

| 단계 | 설정 | 순전파 latency |
|---|---|---:|
| Baseline | 기본 설정 | 약 28 ms/batch |
| 초기 L1_SMALL 실험 | L1_SMALL 적용 | 약 27 ms/batch |
| Prepared tensor 재사용 | 48 KiB, Conv/Pool config L1, prepared weight/bias 재사용, block height 32 강제 | 약 21 ms/batch |
| 현재 최적 구성 | 위 조건에서 `act_block_h_override=0` | **약 18 ms/batch** |

현재 최적 구성은 baseline 대비 약 **35.7%**, 21 ms 구성 대비 약 **14.3%** 감소했습니다.

18 ms는 충분한 warm-up 이후의 관측값입니다. 각 Conv2D의 첫 호출에서는 weight와 bias 준비 비용이 발생하며, 이후 호출부터 준비된 tensor를 재사용합니다.

## 측정 범위

`train_ttml.py`의 측정 구간:

```text
synchronize
  → Frozen VGG11 backbone forward
  → TTML classifier forward
  → synchronize
```

따라서 18 ms는 다음 항목을 제외한 **모델 순전파 시간**입니다.

- dataset loading 및 host preprocessing
- cross-entropy loss
- backward
- optimizer step
- metric 계산
- checkpoint 저장

전체 training step latency와 직접 동일시하면 안 됩니다.

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

## 현재 최적화 구성

TTML device를 48 KiB L1_SMALL 예약 영역으로 엽니다.

```python
context.open_device(
    device_ids=[DEVICE_ID],
    l1_small_size=48 * 1024,
)
```

Conv2D와 MaxPool2D configuration tensor를 모두 L1_SMALL에 배치합니다.

```python
ttnn.Conv2dConfig(
    config_tensors_in_dram=False,
    act_block_h_override=0,
)

ttnn.max_pool2d(
    config_tensor_in_dram=False,
)
```

각 Conv2D는 첫 호출에서 준비된 weight와 bias를 받아 보관합니다.

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

이 구조는 첫 호출 이후 매 순전파에서 weight와 bias 준비를 반복하지 않도록 합니다.

`act_block_h_override=0`으로 block height 강제를 해제합니다. 여기서 0은 높이가 0인 block을 뜻하는 것이 아니라 TTNN의 자동 선택 경로를 사용한다는 의미입니다.

## 실험 해석

48 KiB와 두 config tensor의 L1 배치만 적용한 별도 실험에서는 약 28 ms가 관측됐습니다. prepared weight/bias 재사용과 block height 32 강제 조건에서는 약 21 ms였고, `act_block_h_override=0`으로 자동 선택을 사용하자 약 18 ms가 관측됐습니다.

따라서 각 개선을 **L1 배치**, **prepared tensor 재사용**, **block height 자동 선택**으로 분리해 기록합니다. 다만 실험 세션과 다른 설정을 완전히 통제한 ablation은 추가 검증이 필요합니다.

- [L1_SMALL Size Tuning Experiment](L1_SMALL_TUNING.md)
- [Memory Transfer Measurements and Bottleneck Hypotheses](MEMORY_BOTTLENECK_EVIDENCE.md): 메모리 이동 실측, spill 비용 추정, 검증할 병목 가설
- [P100a Bandwidth Benchmark](benchmarks/p100a-bandwidth/README.md): CPU, host↔device, L1↔DRAM 유효 대역폭 측정 코드
- [TTML L1_SMALL Configuration Patch](../ttml-l1-small-config/README.md)

## 코드에 포함된 기타 설정

현재 코드에는 다음 설정도 포함되어 있지만, 각각의 성능 효과는 아직 분리 측정하지 않았습니다. `act_block_h_override=0`의 21 ms → 18 ms 변화는 별도 단계로 기록했습니다.

- `enable_act_double_buffer=False`
- `enable_weights_double_buffer=True`
- `reshard_if_not_optimal=True`
- `output_layout=ROW_MAJOR_LAYOUT`
- MaxPool 이후 DRAM memory configuration 변환

## 파일 구조

```text
├── L1_SMALL_TUNING.md
├── MEMORY_BOTTLENECK_EVIDENCE.md
├── benchmarks/
│   └── p100a-bandwidth/
│       ├── README.md
│       ├── all_bandwidth.py
│       ├── cpu_ddr4_bandwidth.py
│       ├── host_device_bandwidth.py
│       └── l1_dram_bandwidth.py
└── src/
    ├── dataset.py
    ├── test_dataset.py
    ├── train_ttml.py
    ├── vgg.py
    └── vgg_ttml.py
```

- `dataset.py`: Oxford-IIIT Pet dataset과 DataLoader
- `test_dataset.py`: dataset loading 검증
- `train_ttml.py`: TTML classifier 학습 및 순전파 latency 측정
- `vgg.py`: TTNN Conv, MaxPool, prepared weight cache 및 VGG 연산
- `vgg_ttml.py`: frozen VGG11 backbone과 TTML classifier 연결

## 실행 전제

- [TTML / tt-train Build Setup](../ttml-tt-train-build/README.md)
- [TTML L1_SMALL Configuration Patch](../ttml-l1-small-config/README.md)

## 실행

```bash
ttml
cd projects/tenstorrent-fullstack/vgg11-ttml-optimization/src
python train_ttml.py
```

`DATA_ROOT`, `WEIGHTS`, batch size와 epoch 수는 `train_ttml.py`에서 환경에 맞게 설정합니다.
