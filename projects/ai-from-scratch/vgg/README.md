# VGG From Scratch

CNN의 기본 구성과 feature hierarchy를 이해하기 위한 첫 프로젝트입니다.

## 구현 범위

- [ ] VGG-11
- [ ] VGG-16
- [ ] configurable convolution blocks
- [ ] weight initialization
- [ ] training/evaluation pipeline
- [ ] tensor shape 및 parameter 수 검증

## 분석 항목

- convolution stage별 feature-map 크기
- parameter와 FLOPs 분포
- activation memory
- CPU/GPU inference latency

## 예정 구조

```text
src/
tests/
configs/
scripts/
benchmarks/
reports/
```
