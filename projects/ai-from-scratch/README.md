# AI From Scratch

## 범위

`VGG → ResNet → YOLO → U-Net → Multi-task`

## 첫 번째 목표: ResNet-18

- residual/basic block 직접 구현
- CIFAR-10 또는 작은 공개 dataset으로 학습
- tensor shape와 parameter 수 검증
- CPU/GPU baseline 측정
- Conv, BatchNorm, ReLU 구간의 profiler 분석

## 예정 구조

```text
src/
tests/
configs/
scripts/
benchmarks/
reports/
```

코드와 결과가 준비될 때까지 빈 구현 파일은 커밋하지 않습니다.
