# ResNet From Scratch

Residual connection과 downsampling path를 직접 구현하고 성능 특성을 분석합니다.

## 구현 범위

- [ ] BasicBlock
- [ ] Bottleneck
- [ ] ResNet-18
- [ ] ResNet-50
- [ ] training/evaluation pipeline
- [ ] reference implementation과 수치·구조 비교

## 분석 항목

- residual connection의 gradient flow
- stage별 tensor shape
- Conv/BatchNorm/ReLU 병목
- parameter, FLOPs, activation memory
- CPU/GPU 및 향후 NPU baseline

## 예정 구조

```text
src/
tests/
configs/
scripts/
benchmarks/
reports/
```
