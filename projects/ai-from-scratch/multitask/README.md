# Multi-task Perception From Scratch

Shared backbone에서 detection과 semantic segmentation을 동시에 수행하는 모델을 구현합니다.

## 구현 범위

- [ ] shared backbone
- [ ] detection head
- [ ] segmentation head
- [ ] task별 loss
- [ ] loss weighting
- [ ] joint training/evaluation pipeline

## 비교 실험

- single-task 대비 task별 정확도
- hard sharing과 분기 위치
- static/dynamic loss weighting
- 공유 feature의 memory 절감
- sequential 실행 대비 latency

## 예정 구조

```text
src/
tests/
configs/
scripts/
benchmarks/
reports/
```
