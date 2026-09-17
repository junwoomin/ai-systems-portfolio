# YOLO From Scratch

Object detection의 backbone, neck, head 및 후처리 흐름을 분리해 구현합니다.

## 구현 범위

- [ ] backbone
- [ ] multi-scale neck
- [ ] detection head
- [ ] target assignment
- [ ] box/objectness/class loss
- [ ] IoU 및 NMS
- [ ] training/evaluation pipeline

## 평가 항목

- mAP
- precision/recall
- input resolution별 latency
- 전처리·모델·NMS 구간별 실행 시간
- peak memory

## 예정 구조

```text
src/
tests/
configs/
scripts/
benchmarks/
reports/
```
