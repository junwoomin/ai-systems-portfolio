# VGG11 TTML Optimization on P100a

Tenstorrent P100a에서 VGG11의 TTNN/TTML 순전파 병목을 분석하고, L1_SMALL 배치와 Conv2D prepared weight/bias 재사용으로 latency를 줄인 프로젝트입니다.

## 현재 결과

| 단계 | 설정 | 순전파 latency |
|---|---|---:|
| Baseline | 기본 설정 | 약 28 ms/batch |
| 초기 L1_SMALL 실험 | L1_SMALL 적용 | 약 27 ms/batch |
| 현재 최적 구성 | 48 KiB, Conv/Pool config L1, prepared weight/bias 재사용 | 약 21 ms/batch |

현재 최적 구성은 baseline 대비 약 **25%**, 초기 L1_SMALL 결과 대비 약 **22.2%** 감소했습니다.

21 ms는 충분한 warm-up 이후의 관측값입니다. 각 Conv2D의 첫 호출에서는 weight와 bias 준비 비용이 발생하며, 이후 호출부터 준비된 tensor를 재사용합니다.

## 측정 범위

`train_ttml.py`의 측정 구간:

```text
synchronize
  → Frozen VGG11 backbone forward
  → TTML classifier forward
  → synchronize
```

따라서 21 ms는 다음 항목을 제외한 **모델 순전파 시간**입니다.

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

## 실험 해석

48 KiB와 두 config tensor의 L1 배치만 적용한 별도 실험에서는 약 28 ms가 관측됐습니다. 같은 메모리 구성에서 prepared weight/bias 재사용을 추가한 현재 결과는 약 21 ms입니다.

따라서 최신 28 ms → 21 ms 변화는 단순히 L1_SMALL 크기만의 효과로 기록하지 않습니다. 현재 증거는 **L1 배치와 반복 준비 비용 제거를 함께 적용한 구성**의 결과이며, 각 요인의 독립 효과는 추가 ablation으로 검증해야 합니다.

- [L1_SMALL Size Tuning Experiment](L1_SMALL_TUNING.md)
- [TTML L1_SMALL Configuration Patch](../ttml-l1-small-config/README.md)

## 코드에 포함된 기타 설정

현재 코드에는 다음 설정도 포함되어 있지만, 각각의 성능 효과는 아직 분리 측정하지 않았습니다.

- `act_block_h_override=32`
- `enable_act_double_buffer=False`
- `enable_weights_double_buffer=True`
- `reshard_if_not_optimal=True`
- `output_layout=ROW_MAJOR_LAYOUT`
- MaxPool 이후 DRAM memory configuration 변환

## 파일 구조

```text
├── L1_SMALL_TUNING.md
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
