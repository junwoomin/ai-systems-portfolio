# Tenstorrent Study Notes

Tenstorrent P100a의 하드웨어 구조와 소프트웨어 실행 계층을 체계적으로 정리한 학습 문서이다. 각 문서는 공식 문서와 공개 기술 자료를 바탕으로 작성하였으며, 실험에서 확인된 사실과 구조적 해석을 구분하여 기술한다.

## 문서

| 문서 | 범위 |
|---|---|
| [P100a Blackhole 하드웨어 구조와 TTNN 메모리 실행 모델](p100a-hardware-architecture.md) | Tensix, L1 SRAM, RISC-V, FPU·SFPU, NoC, GDDR6, sharding 및 Circular Buffer |
| [PyTorch에서 P100a까지: 컴파일러·런타임·학습 스택 분석](pytorch-to-p100a-software-stack.md) | PyTorch, TorchDynamo, FX, XLA, StableHLO, PJRT, MLIR, TT-MLIR, TTNN, TT-Metal 및 TTML |

## 작성 원칙

- 공식 문서와 공개 저장소를 우선 근거로 사용한다.
- 제품 사양, 런타임 관찰값과 해석을 구분한다.
- 동일한 약어를 가진 TTML과 TT-MLIR을 분명히 구분한다.
- layout, sharding, memory configuration과 물리 메모리 계층을 혼용하지 않는다.
- 성능 수치는 측정 조건과 동기화 범위를 함께 기록한다.
