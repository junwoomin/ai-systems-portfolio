# P100a Blackhole 하드웨어 구조와 TTNN 메모리 실행 모델

## 초록

본 문서는 Tenstorrent P100a의 Blackhole 기반 하드웨어 구조를 TTNN 및 VGG 실행 관점에서 분석한다. P100a의 Tensix 배열, 코어별 L1 SRAM, RISC-V 제어 코어, FPU·SFPU, NoC, GDDR6 계층을 정리하고, `HEIGHT_SHARDED`, `L1_SMALL`, Circular Buffer와 activation 수명 관리가 실제 실행에 미치는 영향을 설명한다. 제품 사양의 120개 Tensix와 TTNN에서 관찰되는 110개 compute-with-storage core의 차이도 공식 자료와 측정 결과를 바탕으로 구분한다.

## 하드웨어 참고 이미지

![P100a accelerator card](https://tenstorrent.com/_next/image?q=75&url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fjpb4ed5r%2Fproduction%2Fbf6bedd60f2b0d9be3380849364110421d7de6b0-3200x3200.png&w=3840)

![Blackhole tile and NoC structure](https://docs.tenstorrent.com/tt-system-firmware/_images/blackhole.webp)

![Blackhole processor](https://tenstorrent.com/_next/image?q=75&url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fjpb4ed5r%2Fproduction%2F05fa1b264ddebf290b20b64a6552926b29c6a407-1920x1280.png&w=640)

![Tensix memory and dataflow structure](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/_images/tensix-memory-diagram.webp)

각 이미지는 P100a 카드, Blackhole의 타일·NoC 구조, 프로세서 구성, Tensix 내부 L1 SRAM과 compute/data-movement 구조를 나타낸다.

## 1. P100a 전체 구조

P100a의 핵심 사양은 다음과 같다.

| 항목 | P100a |
| --------------- | -------------------------- |
| 칩               | Blackhole Tensix Processor |
| Tensix Core     | **120개**                   |
| Tensix당 L1 SRAM | **1.5 MB**                 |
| 총 SRAM          | **180 MB**                 |
| Big RISC-V      | **16개**                    |
| GDDR6           | **28 GB**                  |
| GDDR6 대역폭       | **448 GB/s**               |
| PCIe            | **PCIe 5.0 x16**           |
| 최대 AI Clock     | **1.35 GHz**               |
| 전력              | 최대 **300 W**               |

Tenstorrent가 공식적으로 P100a에 120 Tensix, 180 MB SRAM, 28 GB GDDR6, 448 GB/s를 명시하고 있다. ([Tenstorrent](https://tenstorrent.com/hardware/cards "Cards"))

개념적으로는:

```text
┌────────────────────────────── P100a ──────────────────────────┐
│                                                               │
│   PCIe 5.0 x16                                                │
│       ↑ ↓                                                     │
│      Host CPU / DDR4                                          │
│                                                               │
│ ┌──────────────── Blackhole Chip ──────────────────────────┐  │
│ │                                                          │  │
│ │    GDDR6 28GB                                            │  │
│ │       ↑ ↓                                                │  │
│ │      NoC                                                 │  │
│ │       ↑ ↓                                                │  │
│ │  ┌───────┐ ┌───────┐ ┌───────┐          ...            │  │
│ │  │Tensix │ │Tensix │ │Tensix │                         │  │
│ │  │ Core  │ │ Core  │ │ Core  │                         │  │
│ │  │1.5MB  │ │1.5MB  │ │1.5MB  │                         │  │
│ │  │ L1    │ │ L1    │ │ L1    │                         │  │
│ │  └───────┘ └───────┘ └───────┘                         │  │
│ │                                                          │  │
│ │    + 16 Big RISC-V                                      │  │
│ │    + PCIe / management / memory-controller nodes        │  │
│ │                                                          │  │
│ └──────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. 제품 사양 120개와 TTNN compute grid 110개의 차이

이 차이는 제품 사양과 런타임 가용 자원의 집계 기준에서 발생한다.

Blackhole 기본 구조 자료에는 Tensix가:

```text
전체 Tensix grid       : 14 × 10
compute 사용 가능     : 13 × 10
```

으로 설명되어 있다. ([GitHub](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md "tt-metal/tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md at main · tenstorrent/tt-metal · GitHub"))

P100a 보드 설정은 다음과 같다.

```text
tensix_col_disable_count = 2
dram_disable_count       = 1
eth_disabled             = true
```

이다. ([Tenstorrent](https://docs.tenstorrent.com/tt-system-firmware/boards/tenstorrent/tt_blackhole/doc/_generated/tt_blackhole_configuration.html "TT BlackHole Board Configuration Reference — TT-System-Firmware 19.15.99 documentation"))

공식 자료를 종합하면 P100a의 활성 Tensix 구성은 다음과 같이 해석할 수 있다.

```text
Blackhole 원래
14 × 10 Tensix
      │
      ├─ Tensix column 2개 disable
      ↓
12 × 10
= 120 Tensix
```

또한 일반 Blackhole에서 일부 열이 `compute_with_storage` grid에서 제외되는 구조를 함께 고려하면 다음 구성이 도출된다.

```text
P100a TTNN compute grid
11 × 10
= 110
```

이는 측정된 TTNN 출력과 일치한다.

따라서 **P100a 제품 사양의 120개와 TTNN에서 관찰되는 110개는 서로 다른 집계 기준이며 모순되지 않는다.**

```text
120 = 제품에 존재하는 활성 Tensix 수
110 = 해당 TTNN 연산 grid에서 사용 가능한 compute-with-storage cores
```

로 구분할 수 있다.

---

## 3. Tensix Core의 내부 구조

Tensix Core는 GPU의 SM과 유사한 계층으로 비교할 수 있으나, 내부 실행 구조에는 상당한 차이가 있다.

```text
              Tensix Core 하나
┌───────────────────────────────────────────────┐
│                                               │
│        L1 SRAM ≈ 1.5 MB                       │
│   ┌───────────────────────────────────────┐   │
│   │ Tensor shard                          │   │
│   │ Circular Buffer                       │   │
│   │ Halo                                  │   │
│   │ Intermediate activation               │   │
│   │ Temporary buffers                     │   │
│   └───────────────────────────────────────┘   │
│                                               │
│ Data Movement RISC-V 0       RISC-V 1        │
│       │                          │             │
│       └────────── NoC ──────────┘             │
│                                               │
│          Unpack RISC-V                        │
│                ↓                              │
│             Unpacker                          │
│                ↓                              │
│         SrcA / SrcB registers                 │
│                ↓                              │
│       ┌───────────────────┐                   │
│       │ FPU / Matrix Unit │                   │
│       │ SFPU / Vector Unit│                   │
│       └───────────────────┘                   │
│                ↓                              │
│          Dst registers                        │
│                ↓                              │
│             Packer                            │
│                ↓                              │
│           Pack RISC-V                         │
│                ↓                              │
│             L1 SRAM                           │
│                                               │
└───────────────────────────────────────────────┘
```

Tensix 하나에는 **5개의 작은 RISC-V CPU**, FPU(matrix), SFPU(vector), unpacker/packer, 두 NoC interface, 그리고 1.5 MB의 local SRAM이 있다. ([GitHub](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md "tt-metal/METALIUM_GUIDE.md at main · tenstorrent/tt-metal · GitHub"))

핵심적인 구분은 다음과 같다.

> **RISC-V 코어는 주로 제어와 데이터 이동을 담당하며, 행렬 연산은 FPU와 SFPU가 수행한다.**

RISC-V는 주로 명령 발행과 데이터 이동을 제어한다.

주요 AI 계산은 다음 연산 장치에서 수행된다.

```text
FPU / SFPU
```

가 담당한다.

---

## 4. 5개 RISC-V 코어의 역할

기능을 단순화하면:

```text
RISC-V #0
    Data movement / Reader

RISC-V #1
    Data movement / Writer

RISC-V #2
    Unpack 제어

RISC-V #3
    Math 제어

RISC-V #4
    Pack 제어
```

이다.

Metalium에서 흔히 말하는:

```text
reader kernel
compute kernel
writer kernel
```

이는 위 실행 역할에 대응한다.

예를 들어 Conv라면:

```text
Reader
 │
 │ GDDR6 / 다른 L1에서 데이터를 가져옴
 ↓
L1 Circular Buffer
 │
 ↓
Unpacker
 │
 ↓
FPU
 │
 ↓
Packer
 │
 ↓
L1 Circular Buffer
 │
 ↓
Writer
 │
 └────────→ DRAM 또는 다른 Tensix
```

형태이다.

Tenstorrent는 compute kernel 하나도 내부적으로 unpack/math/pack을 담당하는 별개의 RISC-V binary들로 컴파일한다고 설명한다. ([GitHub](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst "tt-metal/docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst at main · tenstorrent/tt-metal · GitHub"))

---

## 5. FPU와 SFPU의 기능 구분

**FPU(Matrix engine)**는 AI의 대형 연산을 담당한다.

대표적으로:

```text
MatMul
Conv
matrix multiply
일부 elementwise
pooling 관련 계산
```

이다.

**SFPU(Vector engine)**는 벡터/특수 연산을 담당한다.

예를 들면:

```text
exp
sqrt
activation
일부 normalization
vector unary operation
```

등이 해당한다.

두 연산 장치는 일반 CPU의 load/store 방식으로 L1을 직접 접근하지 않는다.

```text
L1
 ↓
Unpacker
 ↓
register
 ↓
FPU / SFPU
 ↓
register
 ↓
Packer
 ↓
L1
```

구조이다. ([GitHub](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst "tt-metal/docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst at main · tenstorrent/tt-metal · GitHub"))

---

## 6. Unpacker와 Packer의 분리 구조

이는 Tenstorrent 데이터플로우 구조의 주요 특징이다.

예를 들어 L1에 BF16 데이터가 있다고 하면:

```text
L1 BF16 tensor
       ↓
    Unpacker
       ↓
compute engine이 사용하기 좋은 내부 register 표현
       ↓
     FPU
       ↓
result register
       ↓
     Packer
       ↓
L1 BF16 tensor
```

이다.

그래서 저장 형식과 연산 형식이 달라도 하드웨어가 변환을 해줍니다.

BLOCKFP8 같은 압축 형식도 이 구조를 활용한다. ([GitHub](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst "tt-metal/docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst at main · tenstorrent/tt-metal · GitHub"))

---

## 7. 명시적으로 관리되는 L1 SRAM

이 지점에서 GPU의 자동 관리 캐시와 명확한 구분이 필요하다.

P100a의 L1은 일반 CPU의:

```text
자동 L1 cache
```

와 같은 자동 관리 캐시로만 해석해서는 안 된다.

오히려:

> **프로그래머/runtime가 명시적으로 관리하는 매우 빠른 scratchpad SRAM**

에 가깝다.

따라서 본 실험에서 사용하는 다음 설정은

```python
ttnn.to_memory_config(...)
```

와 sharding 설정이 중요한다.

예를 들어:

```text
DRAM
 ↓
HEIGHT_SHARDED
 ↓
110개 core의 L1에 분배
```

를 runtime이 명시적으로 수행한다.

---

## 8. `HEIGHT_SHARDED`의 의미

해당 테스트에서:

```text
Compute grid : 11 × 10 = 110

Tensor:
(1, 1, 65560, 64)
```

이고 결과가:

```text
ShardSpec shape=[596,64]
```

였죠.

정확히:

```text
65560 / 110 = 596
```

이다.

즉:

```text
전체 Tensor
NHW = 65560
C   = 64

             ↓ HEIGHT_SHARDED

Core 0   [596,64]
Core 1   [596,64]
Core 2   [596,64]
...
Core109  [596,64]
```

이다.

그래서 **110개 Tensix가 각각 자기 L1에 tensor 일부를 가지고 동시에 계산**한다.

이 구조는 TTNN에서 sharding 설정이 필요한 이유를 설명한다.

---

## 9. Circular Buffer의 L1 점유

실험 과정에서 관찰된 다음 오류는

```text
Statically allocated circular buffers
clash with L1 buffers
```

위 메모리 구조와 직접 관련된다.

각 1.5 MB L1에는 activation 외에도 여러 실행 버퍼가 함께 배치된다.

```text
Tensix L1 ~1.5 MB
┌───────────────────────────┐
│ Runtime / reserved        │
├───────────────────────────┤
│ Circular Buffer input     │
├───────────────────────────┤
│ Circular Buffer weight    │
├───────────────────────────┤
│ Circular Buffer output    │
├───────────────────────────┤
│ Halo buffer               │
├───────────────────────────┤
│ Tensor shard              │
├───────────────────────────┤
│ Temporary buffers         │
├───────────────────────────┤
│ L1_SMALL reservation      │
└───────────────────────────┘
```

그래서 해당 Conv에서:

```text
pool output
+
conv input
+
halo
+
circular buffer
```

가 동시에 존재하면 1.5 MB를 넘어갈 수 있다.

Tenstorrent 역시 reader/compute/writer kernel이 L1의 circular buffer를 통해 데이터를 전달한다고 설명한다. ([GitHub](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md "tt-metal/METALIUM_GUIDE.md at main · tenstorrent/tt-metal · GitHub"))

---

## 10. `deallocate_activation=True`의 효과

처음에는:

```text
약 233 KB 부족
```

이었다가:

```python
deallocate_activation=True
```

한 뒤:

```text
약 12 KB 부족
```

까지 떨어졌죠.

그 이유는 단순한다.

기존:

```text
old input activation ─────┐
new halo                  │
CB                        ├→ 동시에 L1 점유
output                    │
temporary                 │
                          ┘
```

였다가 old activation을 최대한 빨리 해제하면서:

```text
old input FREE
      ↓

halo
CB
output
```

만 유지되는 구조가 된다.

---

## 11. GDDR6 대용량 저장 계층

P100a에서는:

```text
L1 SRAM
약 1.5 MB / Tensix

vs

GDDR6
28 GB
```

이다.

GDDR6는 대용량 저장 공간이지만 연산 장치가 직접 계산을 수행하는 공간은 아니다.

일반적인 데이터 흐름은:

```text
GDDR6
   ↓
NoC
   ↓
Tensix L1
   ↓
Unpacker
   ↓
FPU/SFPU
   ↓
Packer
   ↓
L1
   ↓
NoC
   ↓
GDDR6
```

이다. ([GitHub](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/labs/matmul/lab1/lab1.rst "tt-metal/docs/source/tt-metalium/tt_metal/labs/matmul/lab1/lab1.rst at main · tenstorrent/tt-metal · GitHub"))

---

## 12. 측정 대역폭과 메모리 계층의 관계

해당 결과:

```text
CPU DDR4 → P100a DRAM
7.3 GB/s

P100a DRAM → CPU
1.9 GB/s

L1 → GDDR6
119 GB/s

GDDR6 → sharded L1
45 GB/s
```

를 그림에 놓으면:

```text
            PC
┌──────────────────────────┐
│ CPU                      │
│ DDR4                     │
└────────────┬─────────────┘
             │
             │ PCIe
             │
       ↓ 7.3 GB/s
       ↑ 1.9 GB/s
             │
             ▼
┌──────────────────────────┐
│ P100a GDDR6 28 GB        │
│ theoretical 448 GB/s     │
└────────────┬─────────────┘
             │
             │ NoC + memory layout conversion
             │
       ↑ L1→DRAM 119 GB/s
       ↓ DRAM→L1 45 GB/s
             │
             ▼
┌──────────────────────────┐
│ Tensix L1 SRAM           │
│ ~1.5 MB/core             │
└────────────┬─────────────┘
             │
             ▼
        Unpacker
             │
       ┌─────┴─────┐
       ▼           ▼
      FPU         SFPU
    Matrix        Vector
       │           │
       └─────┬─────┘
             ▼
           Packer
             │
             ▼
           L1
```

이렇게 된다.

---

## 13. P100a의 28 GB GDDR6 구성

이것도 문서를 보면 이유가 드러납니다.

Blackhole 기본 구조는:

```text
DRAM banks = 8
각 bank ≈ 4 GB
```

이다. ([GitHub](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md "tt-metal/tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md at main · tenstorrent/tt-metal · GitHub"))

P100a 설정은:

```text
dram_disable_count = 1
```

이다. ([Tenstorrent](https://docs.tenstorrent.com/tt-system-firmware/boards/tenstorrent/tt_blackhole/doc/_generated/tt_blackhole_configuration.html "TT BlackHole Board Configuration Reference — TT-System-Firmware 19.15.99 documentation"))

그래서:

```text
8 banks
 - 1 disabled
-------------
7 banks

7 × 4 GB
= 28 GB
```

가 된다.

대역폭도 일치한다.

```text
Full 8-bank Blackhole
512 GB/s

P100a
512 × 7/8
= 448 GB/s
```

실제 공식 P100a 스펙도 **448 GB/s**이다. ([Tenstorrent](https://tenstorrent.com/hardware/cards "Cards"))

따라서 P100a는 Blackhole의 일부 자원을 비활성화한 SKU로 해석할 수 있다.

---

## 14. P100a 통합 구조

```text
                       P100a
                         │
              PCIe 5.0 x16
                         │
        ┌─────────────────────────┐
        │     GDDR6 28 GB         │
        │       448 GB/s          │
        │     7 DRAM banks        │
        └────────────┬────────────┘
                     │
                    NoC
                     │
       ┌─────────────┴─────────────┐
       │                           │
   Tensix Core                Tensix Core
   × 120                      ...
       │
       │ TTNN compute grid에서
       │ 해당 장치는 11×10 = 110 사용
       │
┌──────┴─────────────────────────┐
│ Tensix 한 개                   │
│                                │
│ L1 SRAM ~1.5MB                 │
│                                │
│ RISC-V Data0 ─── NoC0          │
│ RISC-V Data1 ─── NoC1          │
│                                │
│ RISC-V Unpack                   │
│       ↓                        │
│    Unpacker                    │
│       ↓                        │
│ ┌───────────────┐              │
│ │ Matrix FPU    │              │
│ │ Vector SFPU   │              │
│ └───────────────┘              │
│       ↓                        │
│     Packer                     │
│       ↓                        │
│ L1 Circular Buffer             │
└────────────────────────────────┘
```

실험에서 사용한 `act_block_h_override`, `HEIGHT_SHARDED`, `L1_SMALL`, `deallocate_activation`, `DRAM_MEMORY_CONFIG`와 Circular Buffer OOM은 모두 **P100a 내부 자원의 배치와 데이터 이동을 조정하는 문제**에 해당한다.

일반적인 PyTorch·CUDA 환경에서는 이러한 세부 사항이 CUDA와 cuDNN 계층에 상당 부분 추상화된다. 반면 TTNN과 TT-Metal은 메모리 배치와 데이터 이동 계층을 비교적 명시적으로 제공하므로 실행 구조를 직접 분석하고 조정할 수 있다.
