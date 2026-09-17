# VGG11 TTML Forward Optimization on P100a

Tenstorrent P100a에서 VGG11의 TTNN/TTML 순전파 병목을 분석하고, TTML device가 L1_SMALL을 실제로 사용할 수 있도록 수정한 뒤 latency를 단계적으로 줄인 프로젝트입니다.

## 결과

| 단계 | 주요 조건 | 순전파 latency |
|---|---|---:|
| Baseline | Conv configuration tensor를 DRAM에 배치 | 약 60 ms/batch |
| L1_SMALL 적용 | TTML source patch + L1_SMALL configuration tensor | 약 50 ms/batch |
| 현재 코드 | L1_SMALL과 Conv 설정 추가 최적화 | 약 28 ms/batch |

- Baseline 대비 현재 latency 감소: 약 **53.3%**
- L1_SMALL 최초 적용 효과: 약 **16.7%**
- L1_SMALL 적용 후 추가 감소: 약 **44.0%**

## 측정 범위

현재 `train_ttml.py`의 측정 구간은 다음 부분입니다.

```text
synchronize
  → Frozen VGG11 backbone forward
  → TTML classifier forward
  → synchronize
```

따라서 28 ms는 다음 항목을 제외한 **모델 순전파 시간**입니다.

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

## 현재 코드의 주요 설정

### TTML device

현재 업로드한 코드가 실제로 사용하는 값은 8 KiB입니다.

```python
context.open_device(
    device_ids=[DEVICE_ID],
    l1_small_size=8 * 1024,
)
```

24 KiB 지원 경로를 TTML에 추가했더라도, 현재 코드의 실험값은 `8 * 1024`이므로 결과를 기록할 때 두 값을 혼동하지 않습니다.

### TTNN Conv2d

```python
ttnn.Conv2dConfig(
    weights_dtype=ttnn.bfloat16,
    config_tensors_in_dram=False,
    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
    act_block_h_override=32,
    enable_act_double_buffer=False,
    enable_weights_double_buffer=True,
    reshard_if_not_optimal=True,
    output_layout=ttnn.ROW_MAJOR_LAYOUT,
)
```

핵심은 다음과 같습니다.

- `config_tensors_in_dram=False`: Conv configuration tensor를 L1_SMALL에 배치
- `act_block_h_override=32`: activation block height 조정
- `enable_weights_double_buffer=True`: weight double buffering 활성화
- `reshard_if_not_optimal=True`: 연속 Conv에서 필요 시 sharding 재구성
- `ROW_MAJOR_LAYOUT`: 다음 Conv와 연결되는 출력 layout 유지

## 남아 있는 병목 후보

현재 MaxPool은 configuration tensor를 DRAM에 두고, 결과를 다시 DRAM memory configuration으로 변환합니다.

```python
config_tensor_in_dram=True
```

```python
x = ttnn.to_memory_config(
    x,
    ttnn.DRAM_MEMORY_CONFIG,
)
```

따라서 다음 최적화에서는 아래 항목을 각각 분리해 측정해야 합니다.

- MaxPool configuration tensor의 L1 배치 가능 여부
- Pool 이후 DRAM round-trip 제거 가능 여부
- Conv 사이 sharded tensor 유지
- activation double buffering
- `act_block_h_override` 후보값 비교
- compile/warm-up 이후 steady-state latency
- full training step의 backward/optimizer 병목

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

먼저 TTML / tt-train 환경과 L1_SMALL source patch가 필요합니다.

- [TTML / tt-train Build Setup](../ttml-tt-train-build/README.md)
- [TTML L1_SMALL Configuration Patch](../ttml-l1-small-config/README.md)

## 실행

```bash
ttml
cd projects/tenstorrent-fullstack/vgg11-ttml-optimization/src
python train_ttml.py
```

`DATA_ROOT`, `WEIGHTS`, batch size와 epoch 수는 `train_ttml.py`에서 환경에 맞게 설정합니다.

## 다음 실험

- [ ] 동일 조건에서 최소 100개 batch latency 수집
- [ ] 평균뿐 아니라 p50/p95 기록
- [ ] 60 ms, 50 ms, 28 ms 코드 버전별 commit/tag 분리
- [ ] full training step latency 측정
- [ ] TTNN profiler로 Conv/Pool/reshard/DRAM transfer 비중 분석
- [ ] MaxPool 이후 DRAM 변환 제거 실험
- [ ] RTX 5070 Ti와 동일 조건 비교
