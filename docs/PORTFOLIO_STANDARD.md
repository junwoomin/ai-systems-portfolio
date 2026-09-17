# Portfolio Evidence Standard

단순히 “구현했다”는 설명은 포트폴리오 증거로 충분하지 않습니다.

## Reproducibility

- 환경 파일과 의존성 버전
- 학습·평가·benchmark 명령
- seed와 dataset split
- checkpoint 또는 공개 가중치 출처

## Correctness

- tensor shape unit test
- reference implementation과 수치 비교
- task metric: accuracy, mAP, mIoU 등

## Performance

| 항목 | 필수 기록 |
|---|---|
| Hardware | CPU/GPU/NPU 모델 |
| Precision | FP32, BF16, FP8, INT8 등 |
| Input | shape, sequence length, batch size |
| Latency | 평균과 p50/p95 |
| Throughput | samples/s 또는 tokens/s |
| Memory | peak memory와 model size |
| Protocol | warm-up 횟수와 반복 횟수 |

## Optimization Log

1. 관측한 병목
2. profiler 근거
3. 변경한 layout, tiling, fusion 또는 scheduling
4. 정확도 영향
5. latency와 memory 변화
6. 실패했거나 되돌린 이유

## README 최소 구성

Overview, Architecture, Repository Structure, Setup, Training, Evaluation, Benchmark, Optimization, Results, Limitations, References
