# L1_SMALL Size Tuning Experiment

TT-NN의 `Conv2D`와 `MaxPool2D`에서 configuration tensor의 저장 위치와 `l1_small_size`가 VGG11 순전파 latency에 미치는 영향을 확인한 실험입니다.

## Configuration

Conv2D:

```python
config_tensors_in_dram = False
```

MaxPool2D:

```python
config_tensor_in_dram = False
```

`False`로 설정하면 해당 연산의 configuration tensor를 DRAM이 아닌 L1_SMALL에 배치합니다.

## Results

| `l1_small_size` | Conv2D config | MaxPool config | Latency |
|---:|---|---|---:|
| 24 KiB | L1_SMALL | L1_SMALL | 약 35 ms |
| 24 KiB | DRAM | L1_SMALL | 약 28 ms |
| 48 KiB | L1_SMALL | L1_SMALL | 약 28 ms |
| 96 KiB | L1_SMALL | L1_SMALL | 약 29 ms |

## Observations

### 24 KiB에서 두 configuration tensor를 모두 L1_SMALL에 배치

```text
L1_SMALL : 24 KiB
Conv2D   : L1_SMALL
MaxPool  : L1_SMALL
Latency  : 약 35 ms
```

이 조건은 이번 실험에서 가장 느렸습니다.

### 24 KiB에서 Conv2D configuration tensor만 DRAM에 배치

```text
L1_SMALL : 24 KiB
Conv2D   : DRAM
MaxPool  : L1_SMALL
Latency  : 약 28 ms
```

같은 24 KiB 예약 크기에서 Conv2D configuration tensor를 DRAM으로 이동하자 약 7 ms 감소했습니다.

이 결과만으로 DRAM이 L1_SMALL보다 빠르다고 결론 내릴 수는 없습니다. 두 종류의 configuration tensor를 24 KiB L1_SMALL에 동시에 배치했을 때 발생한 allocation pressure, placement 변화 또는 다른 실행 계획의 영향을 우선 의심할 수 있습니다.

### 48 KiB에서 두 configuration tensor를 모두 L1_SMALL에 배치

```text
L1_SMALL : 48 KiB
Conv2D   : L1_SMALL
MaxPool  : L1_SMALL
Latency  : 약 28 ms
```

L1_SMALL을 48 KiB로 늘리자 두 configuration tensor를 모두 L1_SMALL에 둔 상태에서도 약 28 ms로 회복했습니다. 이는 24 KiB 조건에서 L1_SMALL 용량 또는 allocation pressure가 영향을 주었을 가능성과 일치합니다.

### 96 KiB에서 두 configuration tensor를 모두 L1_SMALL에 배치

```text
L1_SMALL : 96 KiB
Conv2D   : L1_SMALL
MaxPool  : L1_SMALL
Latency  : 약 29 ms
```

48 KiB보다 예약 영역을 더 늘렸지만 추가 개선은 없었고 약 1 ms 느려졌습니다.

가능한 가설:

- 일반 L1에서 Circular Buffer, activation 및 runtime working buffer가 사용할 공간 감소
- 다른 tiling 또는 buffer placement 선택
- 단순 측정 변동

현재 데이터만으로 세 원인을 구분할 수는 없습니다.

## Interpretation

관찰된 trade-off는 다음과 같이 정리할 수 있습니다.

```text
L1_SMALL이 부족한 조건
    ↓
configuration tensor allocation pressure 가능성
    ↓
latency 증가

L1_SMALL이 충분한 조건
    ↓
configuration tensor를 local memory에 유지
    ↓
latency 회복

L1_SMALL을 과도하게 예약한 조건
    ↓
일반 L1 working space 감소 가능성
    ↓
추가 개선 없음 또는 소폭 악화
```

단, 이는 현재 결과를 설명하는 **가설**이며 profiler와 반복 측정으로 검증해야 합니다.

## Conclusion

전체 조건의 최저 관측값은 약 28 ms이며 다음 두 구성이 같은 수준을 기록했습니다.

```text
24 KiB + Conv2D DRAM + MaxPool L1_SMALL → 약 28 ms
48 KiB + Conv2D L1_SMALL + MaxPool L1_SMALL → 약 28 ms
```

따라서 48 KiB가 유일한 최적값이라고 단정할 수는 없습니다. 다만 **Conv2D와 MaxPool configuration tensor를 모두 L1_SMALL에 배치하는 조건 중에서는 48 KiB가 가장 좋은 관측 결과**였습니다.

현재 all-L1 후보 설정:

```python
l1_small_size = 48 * 1024

Conv2D:
    config_tensors_in_dram = False

MaxPool2D:
    config_tensor_in_dram = False
```

48 KiB와 96 KiB의 1 ms 차이는 측정 변동일 수 있으므로 충분한 warm-up 이후 여러 번 반복해 평균과 p50/p95를 비교해야 합니다.

## 추가 검증

- [ ] 각 조건을 별도 process에서 반복 실행
- [ ] 동일한 software/firmware commit 사용
- [ ] 최소 100개 batch 측정
- [ ] 평균, p50, p95 및 표준편차 기록
- [ ] program cache와 compile warm-up 조건 통일
- [ ] TTNN profiler로 allocator, reshard 및 DRAM transfer 비교
- [ ] 정확도와 출력 수치 일치 확인
- [ ] 일반 L1과 L1_SMALL 사용량 기록
