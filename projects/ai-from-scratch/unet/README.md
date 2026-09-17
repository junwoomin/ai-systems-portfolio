# U-Net From Scratch

Encoder-decoder와 skip connection을 직접 구현해 semantic segmentation을 학습합니다.

## 구현 범위

- [ ] encoder
- [ ] decoder
- [ ] skip connection
- [ ] upsampling 방식 비교
- [ ] segmentation loss
- [ ] training/evaluation pipeline

## 평가 항목

- mIoU
- Dice score
- class별 성능
- 입력 해상도별 latency와 memory
- encoder와 decoder의 연산 비중

## 예정 구조

```text
src/
tests/
configs/
scripts/
benchmarks/
reports/
```
