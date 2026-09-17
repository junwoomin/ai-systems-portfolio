# LLM From Scratch

## 범위

`Attention → Decoder Transformer → Llama blocks → KV cache → Quantization`

## 첫 번째 목표

작은 decoder-only Transformer를 직접 구현하고 다음을 검증합니다.

- causal mask
- attention tensor shape
- autoregressive generation
- prefill과 decode 분리
- KV cache 적용 전후 latency
- greedy, top-k, top-p sampling

최종적으로 RMSNorm, RoPE, SwiGLU, GQA를 포함한 소형 Llama 계열 구조로 확장합니다.
