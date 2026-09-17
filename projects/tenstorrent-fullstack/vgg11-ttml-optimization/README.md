# VGG11 TTML L1_SMALL Experiment on P100a

Tenstorrent P100a에서 VGG11의 TTNN/TTML 순전파를 실행하고, TTML device에 L1_SMALL을 적용했을 때의 latency 변화를 확인한 프로젝트입니다.

## 측정 결과

| 조건 | 순전파 latency | W (전력)|
|---|---|---:|
| 기본 설정 | 약 28 ms/batch | 약 61W |
| L1_SMALL 적용 | 약 27 ms/batch | 약 61W |

L1_SMALL 적용 후 약 1 ms, 비율로는 약 **3.6%** 감소했습니다.

현재 확인된 최적화 효과는 **L1_SMALL 적용에 따른 28 ms → 27 ms 변화만**입니다. 다른 최적화 항목은 아직 개별 실험하지 않았으므로 성능 향상 원인으로 기록하지 않습니다.

## 측정 범위

현재 `train_ttml.py`의 측정 구간:

```text
synchronize
  → Frozen VGG11 backbone forward
  → TTML classifier forward
  → synchronize
```

따라서 27~28 ms는 다음 항목을 제외한 **모델 순전파 시간**입니다.

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

## L1_SMALL 적용

현재 업로드된 코드에서는 TTML device를 열 때 L1_SMALL을 설정합니다.

```python
context.open_device(
    device_ids=[DEVICE_ID],
    l1_small_size=8 * 1024,
)
```

Conv configuration tensor가 L1_SMALL을 사용하도록 설정합니다.

```python
ttnn.Conv2dConfig(
    config_tensors_in_dram=False,
    # ...
)
```

두 설정의 전달 경로와 TTML source 수정 내용은 별도 문서에 기록했습니다.

- [TTML L1_SMALL Configuration Patch](../ttml-l1-small-config/README.md)

## 코드에 포함된 기타 설정

현재 코드에는 다음 설정도 포함되어 있지만, 각각의 성능 효과는 아직 분리 측정하지 않았습니다.

- `act_block_h_override=32`
- `enable_act_double_buffer=False`
- `enable_weights_double_buffer=True`
- `reshard_if_not_optimal=True`
- `output_layout=ROW_MAJOR_LAYOUT`
- MaxPool 이후 DRAM memory configuration 변환

따라서 이 프로젝트에서는 위 설정들이 1 ms 단축을 만들었다고 주장하지 않습니다.

## 파일 구조

```text
src/
├── dataset.py
├── test_dataset.py
├── train_ttml.py
├── vgg.py
└── vgg_ttml.py
```

- `dataset.py`: Oxford-IIIT Pet dataset과 DataLoader
- `test_dataset.py`: dataset loading 검증
- `train_ttml.py`: TTML classifier 학습 및 순전파 latency 측정
- `vgg.py`: TTNN Conv, MaxPool 및 VGG 연산
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
