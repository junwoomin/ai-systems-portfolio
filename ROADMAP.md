# Roadmap

## Phase 0 — Measurement Foundation

- 공통 환경 고정: Python, framework, device, precision
- seed, dataset split, checkpoint 규칙 정의
- parameter, FLOPs, activation memory 측정 도구 작성
- latency/throughput benchmark harness 작성

**완료 조건:** 동일한 명령으로 정확도와 성능을 재현할 수 있어야 합니다.

## Phase 1 — CNN Fundamentals

- Conv2D, pooling, BatchNorm, activation을 NumPy로 검증
- VGG 구현
- ResNet-18 구현 및 residual connection 분석
- EfficientNet은 구조 분석 후 MBConv 핵심 블록 구현

**대표 산출물:** ResNet-18 baseline, shape test, profiler report

## Phase 2 — Detection, Segmentation, Multi-task

- YOLO 계열의 backbone/neck/head 분리 구현
- U-Net 기반 semantic segmentation
- shared backbone을 사용하는 detection + segmentation multi-task
- IoU, NMS, task loss weighting 비교

**대표 산출물:** multi-task inference pipeline과 task별 성능표

## Phase 3 — Transformer and LLM

- scaled dot-product attention과 multi-head attention 구현
- decoder-only Transformer 학습
- RMSNorm, RoPE, SwiGLU, GQA 구현
- KV cache, sampling, batching, quantization 비교

**대표 산출물:** 소형 Llama 계열 모델과 prefill/decode benchmark

## Phase 4 — Tenstorrent Full Stack

- PyTorch baseline을 TT-XLA 경로로 실행
- TT-MLIR IR과 lowering 단계 분석
- TT-NN으로 핵심 operator 재구현
- TT-Metalium custom kernel 작성
- tiling, core mapping, SRAM reuse, NoC traffic 최적화

**대표 산출물:** ResNet 또는 Transformer의 baseline-to-kernel 최적화 보고서

## Phase 5 — Autonomous Driving NPU

- YOLO/segmentation perception baseline
- BEVFormer의 multi-camera-to-BEV 경로 분석
- TCP 또는 ST-P3의 planning/control 경로 재현
- accelerator에서 병목 operator 식별 및 포팅

**대표 산출물:** 정확도·latency·memory를 함께 제시하는 자율주행 NPU 데모
