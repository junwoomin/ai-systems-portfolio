# Tenstorrent Full Stack

## 목표

하나의 대표 모델을 다음 실행 경로 전체에서 추적합니다.

```text
PyTorch
  → TT-XLA / graph capture
  → TT-MLIR lowering
  → TT-NN operators
  → TT-Metalium kernel
  → Tensix cores / SRAM / NoC
```

## 첫 번째 대상

ResNet-18 또는 작은 Transformer를 사용합니다.

## 필수 비교

- PyTorch CPU/GPU baseline
- TT-XLA 실행 결과
- TT-NN port
- custom Metalium operator
- 정확도 오차와 compile time
- inference latency
- SRAM/DRAM traffic
- core utilization

성능 수치는 측정 조건과 함께 기록하며 서로 다른 precision의 이론 성능을 직접 동일시하지 않습니다.
