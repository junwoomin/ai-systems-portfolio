# AI Systems Portfolio

AI 모델 구현에서 시작해 **compiler, runtime, kernel, accelerator optimization**까지 연결하는 학습·개발 포트폴리오입니다.

## 목표

- 대표 딥러닝 모델을 구조부터 직접 구현
- FLOPs, parameter, activation memory, DRAM traffic 분석
- PyTorch baseline과 accelerator backend 성능 비교
- Tenstorrent TT-XLA, TT-MLIR, TT-NN, TT-Metalium 실행 경로 분석
- 자율주행 perception에서 planning/control까지 실제 응용

## 핵심 트랙

| 트랙 | 범위 | 최종 결과 |
|---|---|---|
| [AI From Scratch](projects/ai-from-scratch/README.md) | VGG, ResNet, YOLO, U-Net, Multi-task | 모델 구조와 학습 파이프라인 |
| [LLM From Scratch](projects/llm-from-scratch/README.md) | Attention, Transformer, Llama, KV cache, quantization | 소형 decoder-only LLM |
| [Tenstorrent Full Stack](projects/tenstorrent-fullstack/README.md) | TT-XLA, TT-MLIR, TT-NN, Metalium | PyTorch부터 custom kernel까지 |
| [Autonomous Driving NPU](projects/autonomous-driving-npu/README.md) | YOLO, BEVFormer, TCP, ST-P3 | perception-planning 최적화 |

## 작업 원칙

1. 모든 모델은 정확한 PyTorch baseline부터 만든다.
2. 정확도 검증 없이 성능 최적화를 시작하지 않는다.
3. latency는 warm-up, 반복 횟수, batch size, precision과 함께 기록한다.
4. 최적화 전후의 memory traffic과 latency를 비교한다.
5. 성공 결과뿐 아니라 설계 판단과 실패 원인도 문서화한다.

전체 순서와 완료 조건은 [ROADMAP.md](ROADMAP.md)를 따릅니다.

## Git 규칙

- `feat/*`: 기능 구현
- `perf/*`: 성능 개선
- `bench/*`: benchmark
- `docs/*`: 문서
- `fix/*`: 버그 수정

```text
feat: implement residual block
test: verify ResNet output shapes
bench: add RTX 4090 baseline
perf: reduce DRAM traffic
docs: explain TT-MLIR lowering path
```
