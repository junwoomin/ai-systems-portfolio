# VGG11 TTML Runtime Optimization on P100a

Tenstorrent P100a에서 VGG11의 TTNN/TTML 순전파 병목을 분리하고, prepared weight/bias 재사용과 Conv2D 실행 설정 조정 및 정확한 측정 경계 설정으로 device forward latency를 분석한 프로젝트이다.

## 현재 결과

| 단계 | 변경 사항 | 측정값 | 측정 범위 |
|---|---|---:|---|
| 초기 기준 | 초기 실행 구성 | 약 28 ms/batch | 입력 변환·H2D 포함 |
| Prepared tensor 재사용 | weight/bias 준비 결과 캐시 | 약 21 ms/batch | 입력 변환·H2D 포함 |
| Block height 자동 선택 | `act_block_h_override=0` | 약 18 ms/batch | 입력 변환·H2D 포함 |
| 현재 측정 | 입력 준비와 H2D를 타이머 밖으로 분리 | **약 10 ms/batch** | P100a device forward |

18 ms와 10 ms의 차이인 약 8 ms는 다음 작업이 기존 측정 구간에 포함되어 있었음을 보여준다.

- NCHW → NHWC `permute`
- `contiguous()` 복사
- FP32 → BF16 변환
- CPU tensor → P100a DRAM 전송

따라서 **18 → 10 ms를 P100a 커널 자체의 44.4% 최적화로 해석하면 안 된다.** 모델과 커널은 동일하며, 측정 대상을 host-side 입력 준비가 포함된 호출 시간에서 device-resident forward로 정정한 결과이다.

## 현재 입력 경로

DataLoader는 CPU의 NCHW tensor를 반환한다. 입력 변환과 P100a 전송은 각 배치에서 필요하지만, device forward 타이머를 시작하기 전에 수행한다.

```python
batch_size = images.shape[0]
images = images.permute(0, 2, 3, 1).contiguous()
images = images.to(torch.bfloat16)
x = ttnn.from_torch(
    images,
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=context.get_device(),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

ttnn.synchronize_device(context.get_device())
model_start = time.perf_counter()

features = backbone(x, batch_size)
outputs = classifier(features)

ttnn.synchronize_device(context.get_device())
model_time = time.perf_counter() - model_start
```

`FrozenVGG11`은 더 이상 PyTorch tensor를 내부에서 변환하지 않는다. P100a DRAM에 이미 배치된 TTNN tensor를 입력으로 받는다.

## 현재 실행 구성

TTML device는 별도의 L1_SMALL 예약 크기 없이 연다.

```python
context.open_device(
    device_ids=[DEVICE_ID],
)
```

Conv2D와 MaxPool2D configuration tensor는 DRAM에 배치한다.

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

## 측정 범위

현재 약 10 ms 측정 구간은 다음과 같다.

```text
P100a DRAM의 입력 TTNN tensor
  → Frozen VGG11 backbone forward
  → TTML classifier forward
  → device synchronize
```

다음 항목은 제외된다.

- dataset loading
- NCHW → NHWC 변환
- contiguous host copy
- BF16 host 변환
- host-to-device 전송
- cross-entropy loss
- backward 및 optimizer step
- metric 계산과 checkpoint 저장

첫 Conv 호출의 weight/bias 준비 비용도 warm-up 이후 steady-state 10 ms에는 포함되지 않는다.

전체 응용 지연시간을 평가할 때는 device forward 10 ms만 보고하면 안 된다. 별도로 end-to-end batch latency와 입력 준비/H2D latency를 함께 보고해야 한다.

## Config tensor 배치 해석

초기에는 Conv와 MaxPool configuration tensor를 L1_SMALL에 배치하면 빠를 것으로 예상하였다. 그러나 24 KiB에서 두 config tensor를 모두 L1_SMALL에 배치했을 때 약 35 ms가 관측되었고, 예약 크기를 늘리거나 일부 config tensor를 DRAM으로 이동하면 약 28 ms로 회복하였다.

현재는 L1_SMALL 예약을 사용하지 않고 두 config tensor를 DRAM에 둔다. L1_SMALL 실험은 입력 변환과 H2D가 포함된 이전 측정 경계에서 수행되었으므로, 현재 10 ms device-only 결과와 직접 비교하지 않는다.

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
