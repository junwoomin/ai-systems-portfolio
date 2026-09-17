# PyTorch에서 P100a까지: 컴파일러·런타임·학습 스택 분석

## 초록

본 문서는 PyTorch 모델이 Tenstorrent P100a에서 실행되기까지의 소프트웨어 계층을 체계적으로 분석한다. 텐서와 계산 그래프의 기본 개념에서 출발하여 `torch.compile`, TorchDynamo, FX Graph, PyTorch/XLA, StableHLO, PJRT, MLIR, TT-MLIR, TTNN, TT-Metal과 TTML의 역할을 구분한다. 또한 layout, sharding, L1 SRAM, Circular Buffer, fusion, decomposition과 autograd가 실행 경로에 미치는 영향을 설명하고, TT-XLA 경로와 TTML·TTNN 직접 실행 경로의 성능을 비교할 때 필요한 조건을 제시한다.

## 1. 텐서와 연산

### 텐서는 숫자들을 일정한 모양으로 배열한 것

딥러닝 프로그래밍에서 텐서는 **다차원 수치 배열**로 정의할 수 있다.

```text
숫자 하나              3.0                  스칼라

숫자를 한 줄로 배열    [1.0, 2.0, 3.0]      벡터

숫자를 행과 열로 배열  [[1.0, 2.0],
                       [3.0, 4.0]]          행렬
```

이미지는 보통 행렬보다 차원이 많다.

예시 입력은 다음과 같다.

```python
x.shape == [8, 3, 112, 112]
```

은 NCHW 형식으로 해석하면 다음 뜻이다.

```text
8    : 이미지 개수
3    : 색상 채널 개수
112  : 이미지 높이
112  : 이미지 너비
```

이렇게 배열의 각 축 크기를 나타내는 것이 **shape**이다. 축이 몇 개인지는 **rank**라고 한다. `[8, 3, 112, 112]`는 축이 네 개이므로 rank 4이다. StableHLO 같은 중간 표현도 텐서의 shape와 원소 자료형을 중요한 정보로 취급한다. ([OpenXLA Project](https://openxla.org/stablehlo/spec "StableHLO Specification  |  OpenXLA Project"))

### 텐서는 숫자만 가지고 있지 않다

실제로는 숫자 이외에도 여러 정보가 필요하다.

| 정보 | 의미 | 예시 |
| -------------------- | -------------------- | ------------------ |
| Shape                | 숫자 배열의 모양            | `[8, 128]`         |
| Dtype                | 숫자 하나를 어떤 형식으로 저장하는지 | FP32, BF16         |
| Device               | 어느 장치에서 관리되는지        | CPU, XLA 장치, TT 장치 |
| Layout               | 숫자들을 메모리에 어떤 순서로 놓는지 | ROW\_MAJOR, TILE   |
| Memory configuration | 어떤 메모리 공간에 어떻게 분산하는지 | DRAM, L1, sharding |

TTNN에서는 **layout과 memory configuration을 별도로 관리한다.** 따라서 TILE layout 오류와 sharding 오류는 서로 다른 계층의 제약에서 발생한다. ([Tenstorrent](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tensor.html "Tensor — TT-NN\&trade; documentation"))

### 연산은 텐서를 받아 새로운 텐서를 만드는 계산

예를 들어:

```python
z = x @ w
```

는 두 텐서에 행렬곱을 적용한다.

```text
x: [8, 64]
w: [64, 128]

x @ w: [8, 128]
```

행렬곱의 원소 하나는 다음과 같이 계산한다.

```math
z_{i,j}=\sum_{k=0}^{63}x_{i,k}w_{k,j}
```

즉, `x`의 한 행과 `w`의 한 열에서 대응하는 숫자들을 곱하고 모두 더한다.

그다음:

```python
z = z + b
```

에서 `b.shape == [128]`이면 같은 편향 벡터를 여덟 개 행에 각각 더한다.

마지막:

```python
y = torch.relu(z)
```

는 각 원소에 다음 계산을 한다.

```math
\operatorname{ReLU}(z)=\max(z,0)
```

여기까지는 **수학적으로 무엇을 계산할지**만 정하였다.

아직 “몇 개 코어를 사용할지”, “L1에 무엇을 올릴지”는 정하지 않았다.

---

## 2. Host와 Device 실행 모델

다음 코드를 실행했다고 생각해 봅시다.

```python
y = torch.matmul(x, w)
```

P100a가 이 Python 문장을 읽고 해석하는 것은 아니다.

Python 프로그램은 일반적으로 host CPU에서 실행된다. CPU 쪽 프로그램이 가속기에 작업을 요청하고, 가속기는 전달받은 장치용 프로그램을 실행한다. TT-Metal 예제에서도 호스트 코드가 입력 전송과 장치 workload 실행을 요청하는 구조를 사용한다. ([Tenstorrent](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html "Matmul (Single Core) — TT-Metalium\&trade; documentation"))

여기서 용어가 두 개 나옵니다.

**Host**는 가속기를 제어하며 운영체제와 Python 프로그램이 실행되는 CPU 측 시스템을 의미한다.

**Device**는 주요 텐서 계산을 수행하는 가속기를 의미하며, 본 문서에서는 P100a가 이에 해당한다.

```text
Host: PC의 CPU
─────────────────────────────
Python 실행
모델과 데이터 준비
컴파일러 실행
장치에 작업 요청
결과 확인

             ↓ 작업 요청

Device: P100a
─────────────────────────────
장치용 프로그램 실행
행렬곱, 활성화 함수 등 계산
장치 메모리에 결과 저장
```

따라서 **“P100a에서 학습한다”는 말은 Python 프로그램 전체가 P100a에서 실행된다는 뜻이 아니다.**

주요 텐서 계산을 P100a에서 수행한다는 뜻이다.

### Kernel의 정의

이 문맥의 **kernel**은 특정 계산을 수행하는 하드웨어용 프로그램이다.

운영체제의 Linux kernel과는 다른 의미이다.

행렬곱을 예로 들면, 장치에는 대략 다음 일을 수행할 프로그램이 필요하다.

```text
입력 데이터 읽기
→ 곱셈과 누적합 수행
→ 출력 데이터 저장
```

Tenstorrent에서는 이런 일을 reader, compute, writer 같은 여러 kernel로 나눌 수 있다. 따라서 **고수준 연산 하나가 kernel 하나와 항상 대응하는 것은 아니다.** ([Tenstorrent](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html "Matmul (Single Core) — TT-Metalium\&trade; documentation"))

---

## 3. PyTorch Eager Execution

다음 코드를 살펴본다.

```python
a = x @ w
z = a + b
y = torch.relu(z)
```

일반적인 PyTorch의 eager 방식에서는 코드가 해당 연산에 도달할 때 연산 실행을 요청한다.

```text
첫 번째 줄에 도달 → 행렬곱 실행 요청
두 번째 줄에 도달 → 덧셈 실행 요청
세 번째 줄에 도달 → ReLU 실행 요청
```

CPU 텐서면 CPU용 구현을, CUDA 텐서면 CUDA용 구현을 사용하는 식이다. PyTorch/XLA 문서도 CPU·CUDA의 eager 실행과 XLA의 lazy 실행을 구분한다. ([PyTorch Docs](https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html "PyTorch on XLA Devices — PyTorch/XLA master documentation"))

여기서 오해하면 안 되는 부분이 있다.

**PyTorch가 숫자 하나하나를 Python** **`for`****문으로 계산하는 것은 아니다.**

Python은 보통 이미 구현된 연산 라이브러리를 호출하는 입구 역할을 한다. 그래서 `torch.compile`을 쓰지 않아도 행렬곱 자체는 최적화된 구현으로 실행될 수 있다.

그래프 컴파일이 필요한 이유는 연산 간 최적화 기회를 확보하기 위해서이다.

**연산 하나를 빠르게 실행하는 것과 여러 연산을 묶어서 효율적으로 실행하는 것은 다르기 때문이다.**

예를 들어 행렬곱, 덧셈, ReLU를 각각 따로 처리하면 중간 결과를 여러 번 읽고 쓸 수 있다. 여러 연산을 함께 보면 이런 낭비를 줄일 기회가 생깁니다. 이것이 그래프 기반 최적화의 중요한 목적이다. ([PyTorch Docs](https://docs.pytorch.org/blog/accelerated-pytorch-inference/ "Accelerated PyTorch inference with torch.compile on AWS Graviton processors | PyTorch"))

또한 eager에서 “바로 실행 요청”한다는 것은 **CPU가 매번 계산 완료까지 기다린다**는 뜻은 아니다. GPU나 가속기는 비동기로 작업을 수행할 수 있다.

---

## 4. 계산 그래프

계산 그래프는 **어떤 연산이 있고, 어떤 결과가 다음 연산의 입력으로 들어가는지 표현한 것**이다.

앞의 코드를 그래프로 표현하면 다음과 같다.

```text
x ─────┐
       │
       ▼
     Matmul ◀──── w
       │
       ▼
      Add ◀────── b
       │
       ▼
      ReLU
       │
       ▼
       y
```

계산 그래프에서 연산은 **node**, 연결 관계는 **edge**로 표현된다.

중요한 것은 화살표가 **데이터 의존성**을 나타낸다는 점이다.

`Add`는 `Matmul` 결과가 필요하다. 따라서 행렬곱 결과가 준비되기 전에 덧셈 결과를 완성할 수 없다.

반면 서로 독립적인 두 연산은 실행 순서를 바꾸거나 병렬로 처리할 여지가 있다. MLIR도 연산과 값의 연결 관계를 중심으로 프로그램을 표현한다. ([MLIR](https://mlir.llvm.org/docs/LangRef/ "MLIR Language Reference - MLIR"))

### 그래프가 있다는 것과 계산을 끝냈다는 것은 다르다

예를 들어:

```text
입력 x와 w를 곱한다.
그 결과에 b를 더한다.
음수를 0으로 바꾼다.
```

라는 계획만 만들어 놓을 수 있다.

이 시점에는 실제 결과 숫자가 아직 없을 수 있다.

즉:

```text
그래프 생성 = 계산 계획을 표현

그래프 실행 = 실제 입력 숫자로 계산
```

이다.

이 구분이 뒤에서 나오는 **tracing**, **lazy execution**, **compilation**을 이해하는 출발점이다.

---

## 5. 컴파일러와 중간 표현

### 컴파일러의 역할

일반적인 C 언어 컴파일 과정은 다음과 같다.

```text
C 코드
→ 컴파일
→ 실행 파일
```

컴파일러는 단순히 단어를 일대일로 치환하는 번역기가 아니다.

프로그램의 의미를 분석하고, 필요한 변환과 최적화를 수행한 다음, 실행 가능한 형태를 만든다.

딥러닝 컴파일러도 비슷한다.

```text
행렬곱을 수행하라
    ↓
입력 크기와 자료형 확인
    ↓
하드웨어에서 실행할 연산과 메모리 배치 결정
    ↓
실행 가능한 프로그램 생성
```

MLIR은 이런 단계적 분석과 변환을 구현하기 위한 컴파일러 기반 기술이다. ([MLIR](https://mlir.llvm.org/docs/LangRef/ "MLIR Language Reference - MLIR"))

### IR: 컴파일러가 다루기 위한 중간 표현

IR은 **Intermediate Representation**, 즉 중간 표현이다.

사람은 다음 코드가 편한다.

```python
y = torch.relu(x @ w + b)
```

컴파일러는 이를 다음처럼 명시적으로 나누어 다루는 편이 적절하다.

```text
v0 = Matmul(x, w)
v1 = Add(v0, b)
v2 = ReLU(v1)
Return(v2)
```

왜냐하면 각 연산의 입력, 출력, 자료형, 의존 관계를 따로 분석할 수 있기 때문이다.

IR은 보통 내부적으로 자료구조 형태로 존재한다. 디버깅할 때 사람이 읽을 수 있도록 텍스트로 출력하기도 한다. **IR이 반드시 텍스트 파일이라는 뜻은 아니다.** MLIR은 메모리상의 표현, 사람이 읽는 텍스트 표현, 저장·전송용 직렬화 표현을 구분한다. ([MLIR](https://mlir.llvm.org/docs/LangRef/ "MLIR Language Reference - MLIR"))

### 중간 표현의 장점

프레임워크가 세 개이고 하드웨어가 세 종류라고 가정한다.

중간 표현 없이 모두 직접 연결하면 최대 아홉 개의 변환 조합을 관리해야 한다.

```text
프레임워크 A → 하드웨어 1, 2, 3
프레임워크 B → 하드웨어 1, 2, 3
프레임워크 C → 하드웨어 1, 2, 3
```

공통 중간 표현을 두면:

```text
프레임워크 A ─┐
프레임워크 B ─┼→ 공통 IR → 하드웨어별 컴파일러
프레임워크 C ─┘
```

처럼 역할을 나눌 수 있다.

StableHLO가 바로 이런 상호운용성을 목적으로 하는 표현이다. ([OpenXLA Project](https://openxla.org/ "OpenXLA Project"))

---

## 6. `torch.compile()`의 역할

`torch.compile()`은 **PyTorch 프로그램을 그래프로 포착하고, 선택한 backend가 최적화하도록 연결하는 API**이다.

```python
compiled_model = torch.compile(
    model,
    backend="tt",
)
```

여기서 `backend`는 포착된 그래프를 전달받아 처리하는 구현을 의미한다.

다음 두 경로는 구분해야 한다.

```python
torch.compile(model)
```

과:

```python
torch.compile(model, backend="tt")
```

는 같은 컴파일 경로가 아니다.

PyTorch의 기본 backend는 TorchInductor이고, `"tt"`는 Tenstorrent 쪽 backend를 명시적으로 선택한 것이다. 따라서 **`torch.compile`****을 쓴다고 무조건 XLA나 StableHLO를 거치는 것은 아니다.** ([PyTorch Docs](https://docs.pytorch.org/docs/stable/generated/torch.compile "torch.compile — PyTorch 2.14 documentation"))

### TorchDynamo

`torch.compile`의 앞부분에서 Python 실행을 분석하고 PyTorch 연산을 포착하는 역할을 하는 것이 **TorchDynamo**이다.

예를 들어 다음 함수를 분석한다.

```python
def forward(x, w, b):
    a = x @ w
    z = a + b
    return torch.relu(z)
```

Dynamo는 실행 과정에서 나타나는 연산들을 그래프로 표현한다.

이때 Python 프로그램 전체를 무조건 하나의 그래프로 만드는 것은 아니다. 포착할 수 있는 구간을 그래프로 만들고, 나머지 Python 동작은 별도로 남길 수 있다. ([PyTorch Docs](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html "Dynamo Core Concepts — PyTorch main documentation"))

### Graph break

예를 들어 모델 중간에서 텐서 값을 Python 숫자로 꺼내 복잡한 Python 분기를 한다고 생각해 봅시다.

```python
if x.sum().item() > 0:
    ...
```

이런 동작은 설정과 지원 범위에 따라 그래프 포착을 어렵게 만들 수 있다.

그 결과:

```text
큰 그래프 하나
```

대신:

```text
그래프 A
→ 일반 Python 실행
→ 그래프 B
```

처럼 나뉠 수 있다.

이를 **graph break**라고 한다. 그래프가 나뉘면 여러 연산을 함께 최적화할 기회가 줄어들 수 있다. 다만 구체적으로 어떤 코드가 graph break를 만드는지는 버전과 설정에 따라 다릅니다. ([PyTorch Docs](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html "Working with Graph Breaks — PyTorch main documentation"))

### Guard

컴파일 결과가 특정 조건에서만 유효할 수 있으므로, 그 조건을 검사하는 장치가 필요하다.

예를 들어:

```text
입력 차원은 2개인가?
자료형은 BF16인가?
이번 입력에도 기존 실행 계획을 사용할 수 있는가?
```

같은 조건이다.

이 조건들을 **guard**라고 한다. 조건이 달라 기존 결과를 쓸 수 없으면 재컴파일이 일어날 수 있다. 단, dynamic shape 지원을 사용하면 일부 크기 변화는 재컴파일 없이 처리할 수도 있다. ([PyTorch Docs](https://docs.pytorch.org/docs/stable/generated/torch.compile "torch.compile — PyTorch 2.14 documentation"))

---

## 7. FX Graph

**FX는 PyTorch 연산 그래프를 표현하고 변환하는 데 사용하는 도구 체계**이다.

Dynamo가 포착한 그래프를 FX 형태로 backend에 넘길 수 있다. PyTorch/XLA의 Dynamo 연동 문서도 FX 그래프를 받아 기존 lazy tensor 기술로 처리하는 구조를 설명한다. ([PyTorch Docs](https://docs.pytorch.org/xla/master/torch_compile.html "TorchDynamo(torch.compile) integration in PyTorch XLA — PyTorch/XLA master documentation"))

개념적으로는 다음과 같다.

```text
입력 x
입력 w
입력 b

matmul = x @ w
add    = matmul + b
relu   = ReLU(add)

출력 relu
```

### FX Graph 확인 예제

아래 코드는 P100a 없이 CPU에서 실행할 수 있다.

여기서는 Dynamo 대신 `symbolic_trace()`를 사용해 FX 그래프 모양을 관찰한다. **FX를 이해하기 위한 예제이지, TT-XLA 전체 경로를 실행하는 코드는 아니다.** `symbolic_trace`가 만드는 `GraphModule`은 그래프를 담고 있으면서 실행도 가능한 객체이다. ([PyTorch Docs](https://docs.pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html "(beta) Building a Simple CPU Performance Profiler with FX — PyTorch Tutorials 2.14.0+cu130 documentation"))

```python
import torch
from torch.fx import symbolic_trace


class TinyModel(torch.nn.Module):
    def forward(self, x, w, b):
        return torch.relu(x @ w + b)


model = TinyModel()
graph_model = symbolic_trace(model)

print("=== 계산 그래프 ===")
print(graph_model.graph)

x = torch.tensor([
    [1.0,  2.0, -1.0],
    [0.0, -1.0,  3.0],
])

w = torch.tensor([
    [ 2.0, -1.0],
    [ 1.0,  3.0],
    [-2.0,  1.0],
])

b = torch.tensor([1.0, -2.0])

print("=== 실행 결과 ===")
print(graph_model(x, w, b))
```

위 예제를 실행하면 계산 그래프에는 대략 다음 내용이 나타난다.

```text
x, w, b      : 입력 자리
matmul       : operator.matmul(x, w)
add          : operator.add(matmul, b)
relu         : torch.relu(add)
return relu
```

결과는:

```text
tensor([[7., 2.],
        [0., 0.]])
```

이다.

여기서 핵심은 **그래프를 출력하는 것과 그래프에 실제 숫자를 넣어 실행하는 것이 별도 동작**이라는 점이다.

---

## 8. PyTorch/XLA

`torch-xla`, 즉 PyTorch/XLA는 **PyTorch와 XLA 계열 실행 환경을 연결하는 소프트웨어**이다.

사용자는 여전히 PyTorch 문법을 씁니다.

```python
y = torch.relu(x @ w + b)
```

하지만 `x`, `w`, `b`가 XLA 장치의 텐서라면 내부 처리 방식이 일반 CPU 텐서와 다릅니다.

### Lazy execution: 연산을 모아 두었다가 실행

전통적인 PyTorch/XLA의 lazy 방식에서는 연산을 만날 때마다 바로 장치 연산을 요청하기보다, **어떤 계산이 필요한지 그래프에 기록**한다.

```python
a = x @ w
z = a + b
y = torch.relu(z)
```

를 실행하면서 개념적으로:

```text
a는 x와 w의 행렬곱 결과다.
z는 a와 b의 덧셈 결과다.
y는 z에 ReLU를 적용한 결과다.
```

라고 기록한다.

그리고 실행 경계나 결과가 필요한 지점에서 그래프를 컴파일하고 실행한다. PyTorch/XLA 문서는 이런 동작을 lazy tensor 방식으로 설명한다. ([PyTorch Docs](https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html "PyTorch on XLA Devices — PyTorch/XLA master documentation"))

예를 들어 lazy 경로에서:

```python
torch_xla.sync()
```

는 모아 둔 작업을 실행하는 경계로 사용된다. CPU에서 실제 값을 읽어야 하는 동작도 실행이나 동기화를 유발할 수 있다. ([PyTorch Docs](https://docs.pytorch.org/xla/release/r2.7/learn/eager.html "Eager Mode + Compile API — PyTorch/XLA master documentation"))

### `torch.compile`과의 차이

아니다.

**`torch.compile`****은 그래프 포착과 backend 최적화를 연결하는 API이고,** **`torch-xla`****는 PyTorch를 XLA 계열 장치 실행 체계에 연결하는 구현이다.**

따라서 아래 두 형태를 구분해야 한다.

```text
경로 A: PyTorch/XLA의 lazy 방식

PyTorch 코드 + XLA 텐서
→ lazy graph 기록
→ 컴파일
→ 장치 실행
```

```text
경로 B: torch.compile과 결합한 방식

PyTorch 코드
→ torch.compile
→ Dynamo / FX
→ 선택한 backend와 PyTorch/XLA 처리
→ 컴파일
→ 장치 실행
```

**PyTorch/XLA를 쓰기 위해 항상** **`torch.compile()`****을 호출해야 하는 것은 아니다.**

앞선 설명에서 둘을 `/`로 묶어 하나의 필수 단계처럼 표현한 부분은 부정확하였다. ([PyTorch Docs](https://docs.pytorch.org/xla/master/torch_compile.html "TorchDynamo(torch.compile) integration in PyTorch XLA — PyTorch/XLA master documentation"))

---

## 9. StableHLO

StableHLO는 **머신러닝 계산을 프레임워크에 덜 종속적인 형태로 표현하는 연산 집합이자 중간 표현**이다.

이름의 HLO는 High-Level Operations를 뜻한다.

여기서 high-level이라는 말은:

```text
레지스터 3번에 값을 넣어라.
몇 번째 코어에서 명령어를 실행하라.
```

보다:

```text
이 두 텐서를 행렬곱하라.
이 축을 따라 합계를 구하라.
이 텐서의 모양을 바꿔라.
```

처럼 **텐서 계산의 의미에 가까운 수준**이라는 뜻이다. ([OpenXLA Project](https://openxla.org/stablehlo/spec "StableHLO Specification  |  OpenXLA Project"))

### 예제의 StableHLO 표현

실제 문법의 세부 속성을 생략한 설명용 표현이다.

```text
%0 = stablehlo.dot_general(x, w)

%1 = stablehlo.broadcast_in_dim(b)
     # [128]을 [8, 128]에 맞추는 의미

%2 = stablehlo.add(%0, %1)

%3 = stablehlo.maximum(%2, 같은 shape의 0)

return %3
```

각 연산의 뜻은 다음과 같다.

`dot_general`은 행렬곱보다 일반적인 형태의 곱셈·합산 연산이다. 일반 행렬곱에서는 한쪽의 열 축과 다른 쪽의 행 축을 대응시켜 곱하고 합산한다.

`broadcast_in_dim`은 작은 텐서를 더 큰 shape의 연산에 맞추는 의미를 명시한다.

`add`는 대응하는 원소를 더한다.

`maximum`은 대응하는 두 값 중 큰 값을 선택한다. 0과 비교하면 ReLU를 표현할 수 있다.

이런 방식으로 **PyTorch 함수 호출을 표준화된 텐서 연산으로 표현**한다. ([OpenXLA Project](https://openxla.org/stablehlo/spec "StableHLO Specification  |  OpenXLA Project"))

### `%0`과 `tensor<8x128xbf16>`의 의미

IR을 보면 이런 표현이 나옵니다.

```text
%0
%1
tensor<8x128xbf16>
```

`%0`은 중간 계산 결과에 붙인 이름이다. “첫 번째 메모리 주소”라는 뜻은 아니다.

`tensor<8x128xbf16>`은:

```text
shape: [8, 128]
원소 자료형: BF16
```

이라는 타입 정보이다.

이런 명시적인 정보가 있어야 컴파일러가 행렬곱의 크기가 맞는지, 다음 연산이 가능한지 확인할 수 있다.

### StableHLO 호환성의 의미

학습 loss가 안정적이라는 뜻이 아니다.

**서로 다른 소프트웨어 버전 사이에서 계산 표현을 주고받을 때의 호환성**과 관련된 이름이다. StableHLO는 직렬화된 계산 표현의 호환성 정책을 제공한다. ([OpenXLA Project](https://openxla.org/stablehlo/compatibility "StableHLO Compatibility  |  OpenXLA Project"))

### StableHLO 단계의 하드웨어 배치 범위

일반적으로 아직 아니다.

StableHLO는 계산의 의미를 표현하는 데 중점을 둡니다. P100a의 구체적인 타일 배치, 코어별 L1 할당, circular buffer 크기 같은 실행 계획은 이후 Tenstorrent용 컴파일 과정에서 구체화된다.

TT-XLA는 PyTorch/XLA 등을 통해 얻은 StableHLO를 TT-MLIR에 전달하는 구조이다. ([Tenstorrent](https://docs.tenstorrent.com/tt-xla/getting_started.html "Getting Started — TT-XLA documentation"))

---

## 10. PJRT 장치 인터페이스

PJRT는 StableHLO나 FX처럼 **계산을 표현하는 언어가 아니다.**

프레임워크가 여러 종류의 가속기 backend를 공통된 방식으로 사용할 수 있도록 하는 인터페이스이다.

인터페이스는 **서로 다른 소프트웨어 구성요소가 기능을 요청하고 응답하는 규약**이다. PJRT는 장치별 구현을 plugin으로 연결할 수 있게 한다. ([OpenXLA Project](https://openxla.org/xla/pjrt "PJRT - Uniform Device API  |  OpenXLA Project"))

프레임워크 쪽에서는 다음과 같은 요청이 필요하다.

```text
사용 가능한 장치가 무엇인가?
이 입력을 장치 메모리로 보내라.
이 프로그램을 컴파일하라.
컴파일된 프로그램을 실행하라.
결과가 준비되었는가?
결과를 가져와라.
```

PJRT에는 장치, 메모리 버퍼, 컴파일된 실행 객체, 실행과 완료 상태 등을 다루는 개념이 있다. **컴파일 요청과 실행 요청 모두와 관련된다.** ([OpenXLA Project](https://openxla.org/xla/pjrt/cpp_api_overview "PJRT C++ Device API Overview  |  OpenXLA Project"))

Tenstorrent에서는 대략:

```text
PyTorch/XLA
     ↕
PJRT 인터페이스
     ↕
Tenstorrent PJRT plugin
     ↕
Tenstorrent 컴파일러 / 런타임
```

으로 연결된다.

TT-XLA 설치 과정의 `pjrt_plugin_tt`가 이 연결을 담당한다. ([Tenstorrent](https://docs.tenstorrent.com/tt-xla/getting_started.html "Getting Started — TT-XLA documentation"))

따라서 PJRT를:

```text
FX → PJRT IR → StableHLO
```

같은 새로운 중간 언어 단계로 이해하면 안 된다.

**StableHLO는 전달하는 계산 표현이고, PJRT는 컴파일·장치 실행 등을 요청하는 연결 규약**이다.

---

## 11. MLIR과 Dialect

### MLIR 컴파일러 인프라

MLIR은 여러 추상화 수준의 프로그램 표현과 변환을 지원하는 컴파일러 기반 기술이다.

핵심은 **한 번에 기계어로 내려가지 않고, 의미를 유지하면서 점점 구체적인 표현으로 바꿀 수 있다는 것**이다. ([MLIR](https://mlir.llvm.org/docs/LangRef/ "MLIR Language Reference - MLIR"))

행렬곱을 예로 들면:

```text
높은 수준:
행렬 A와 B를 곱하라.

조금 낮은 수준:
출력 행과 열을 순회하며 곱셈과 합산을 하라.

더 낮은 수준:
출력을 여러 블록으로 나누어라.

하드웨어에 가까운 수준:
특정 코어에 블록을 배정하고,
L1에 입력 타일을 준비한 다음 계산하라.
```

### Dialect

Dialect는 MLIR 안에서 사용하는 **특정 목적의 연산·타입·속성 체계**이다.

일상 언어의 “방언”처럼 번역하지만, 기술적으로는 “이 영역에서 사용할 연산들의 어휘와 규칙”에 가깝다.

예를 들어:

```text
stablehlo.add
ttir.add
ttnn.add
```

는 접두사가 다릅니다.

각각 서로 다른 dialect에 속한 연산이다.

같은 덧셈 계열이어도 어느 추상화 수준에서 어떤 추가 정보를 표현하는지가 다를 수 있다.

MLIR은 이런 다양한 dialect를 같은 기반 위에서 표현하고 변환할 수 있게 한다. ([MLIR](https://mlir.llvm.org/docs/LangRef/ "MLIR Language Reference - MLIR"))

### Lowering

**Lowering은 추상적인 표현을 더 구체적인 표현으로 바꾸는 것**이다.

예를 들어:

```text
행렬곱
```

을 다음과 같은 반복 계산으로 표현할 수 있다.

```python
for i in range(M):
    for j in range(N):
        total = 0

        for k in range(K):
            total += A[i, k] * B[k, j]

        C[i, j] = total
```

계산의 의미는 같지만, 두 번째 표현은 실행 절차가 더 자세한다.

이처럼 추상화 수준을 낮추는 것이 lowering이다. MLIR의 공식 학습 자료도 고수준 연산을 더 낮은 dialect로 부분적으로 변환하는 과정을 설명한다. ([MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/ "Chapter 5: Partial Lowering to Lower-Level Dialects for Optimization - MLIR"))

### Pass

Pass는 컴파일러가 프로그램을 분석하거나 바꾸는 **하나의 처리 단계**이다.

예를 들면:

```text
사용되지 않는 결과를 제거하는 단계
불필요한 변환을 줄이는 단계
연산을 합치는 단계
메모리 배치를 정하는 단계
```

이다.

컴파일러는 보통 이런 pass들을 여러 개 연결해 실행한다.

**Lowering 자체가 반드시 최적화를 뜻하는 것은 아니다.** 실행 가능한 형태로 바꾸는 변환과 성능을 개선하는 최적화는 구분해야 한다. ([MLIR](https://mlir.llvm.org/docs/LangRef/ "MLIR Language Reference - MLIR"))

---

## 12. TT-MLIR의 역할

TT-MLIR은 **MLIR 기반의 Tenstorrent용 컴파일러 인프라**이다.

지금 이야기하는 일반적인 TTNN backend 경로에서는 다음과 같은 변환이 중요한다.

```text
StableHLO
    ↓
TTIR
    ↓
TTNN dialect
    ↓
실행용 프로그램
```

다만 TT-MLIR에는 다른 저수준 경로도 있으므로, 이것이 모든 경우의 유일한 경로라는 뜻은 아니다. ([Tenstorrent](https://docs.tenstorrent.com/tt-mlir/overview.html "Introduction — TT-MLIR documentation"))

### TTIR: Tenstorrent 컴파일러의 고수준 텐서 표현

TTIR에서는 여전히 텐서 연산의 의미를 중심으로 계산을 다룹니다.

예를 들어:

```text
ttir.matmul
ttir.add
ttir.relu
```

같은 형태이다.

StableHLO가 외부 프레임워크와 공통으로 사용하는 표현이라면, TTIR은 Tenstorrent 컴파일러 내부에서 계산을 다루는 표현이라고 이해하면 된다.

이 단계의 변환을 문서에서 **legalization**이라고 부르기도 한다.

여기서 legal은 법률과 무관한다. **다음 단계가 허용하고 처리할 수 있는 연산 형태로 바꾼다**는 의미이다. ([Tenstorrent](https://docs.tenstorrent.com/tt-xla/fusing_and_composite_ops.html "Fusing and Composite Ops — TT-XLA documentation"))

### TTNN dialect: TTNN 라이브러리와 가까운 표현

TTNN dialect는 실제 TTNN API를 가깝게 모델링한다.

이 단계에서는 단순히:

```text
행렬을 곱한다.
```

를 넘어:

```text
어떤 tensor layout을 사용하는가?
어느 memory space에 있는가?
어떤 연산 설정을 사용할 것인가?
```

같은 정보가 구체적으로 부여된다. 공식 예제에서도 TTNN dialect 타입에 tile, DRAM, interleaved 등의 정보가 나타난다. ([Tenstorrent](https://docs.tenstorrent.com/tt-mlir/overview.html "Introduction — TT-MLIR documentation"))

### 메모리 배치 최적화

그렇다.

현재 TTNN optimizer 문서는 연산별 layout·설정 선택과 L1 예산을 넘지 않도록 일부 텐서를 DRAM으로 보내는 처리를 설명한다. 이런 판단은 성능과 실행 가능성을 모두 좌우한다. ([Tenstorrent](https://docs.tenstorrent.com/tt-mlir/specs/ttnn-optimizer.html "TTNN Optimizer — TT-MLIR documentation"))

다만 **모든 세부 결정을 컴파일러 하나가 끝내는 것은 아니다.**

컴파일러가 설정을 결정하는 부분도 있고, TTNN 라이브러리가 전달된 설정이나 기본값을 바탕으로 구현을 선택하는 부분도 있다.

### 컴파일 결과의 형태

아니다.

TT-MLIR은 컴파일 결과를 직렬화된 프로그램 형태로 만들 수 있고, 런타임은 이를 읽어 실행한다. `ttrt`는 이런 FlatBuffer 프로그램을 검사하고 실행하는 도구이다. ([Tenstorrent](https://docs.tenstorrent.com/tt-mlir/ttrt.html "ttrt — TT-MLIR documentation"))

별도의 code generation 기능으로 TTNN Python/C++ 코드를 생성할 수도 있지만, **일반 실행이 항상 “Python 파일 생성 → 그 파일 실행”인 것은 아니다.** ([Tenstorrent](https://docs.tenstorrent.com/tt-xla/getting_started_codegen.html "Code Generation Guide — TT-XLA documentation"))

---

## 13. Layout, Sharding과 L1 설정

이 절은 실제 오류 사례를 메모리 배치 관점에서 분석한다.

### 같은 행렬도 저장하는 순서는 다를 수 있다

설명용으로 다음 행렬을 가정한다.

```text
 1  2  3  4
 5  6  7  8
 9 10 11 12
13 14 15 16
```

행 순서로 저장하면:

```text
1, 2, 3, 4, 5, 6, 7, 8, ...
```

이다. 이런 개념이 ROW\_MAJOR이다.

설명을 위해 2×2 블록 단위로 묶는다고 하면:

```text
첫 블록:  1, 2, 5, 6
둘째 블록: 3, 4, 7, 8
...
```

처럼 저장 순서가 달라집니다.

실제 TTNN은 대표적으로 32×32 tile을 사용하며, 세부 저장 형식은 위의 작은 예제보다 복잡한다. 중요한 것은 **수학적 행렬이 같아도 하드웨어가 읽기 좋은 배치는 다를 수 있다**는 점이다. ([Tenstorrent](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tensor.html "Tensor — TT-NN\&trade; documentation"))

관찰된 다음 오류는

```text
Inputs to matmul must be tilized
```

는 “행렬곱 수학이 틀렸다”가 아니라, **해당 구현이 요구하는 물리적 데이터 배치를 맞추지 않았다**는 뜻이다. TTNN matmul 문서에도 입력 TILE layout 요구가 명시되어 있다. ([Tenstorrent](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.matmul.html "ttnn.matmul — TT-NN\&trade; documentation"))

### Sharding은 저장 순서와 또 다르다

Sharding은 **하나의 텐서를 여러 메모리 영역이나 코어에 나누어 배치하는 것**이다.

예를 들어 64×64 행렬을 해당 코어에 나누면:

```text
┌────────────┬────────────┐
│ Core 0     │ Core 1     │
│ 32×32 부분 │ 32×32 부분 │
├────────────┼────────────┤
│ Core 2     │ Core 3     │
│ 32×32 부분 │ 32×32 부분 │
└────────────┴────────────┘
```

처럼 생각할 수 있다.

반면 TILE layout은 각 데이터 묶음이 어떤 형식으로 저장되는가에 관한 것이다.

따라서:

```text
TILE / ROW_MAJOR
→ 데이터를 어떤 형식으로 배열하는가?

SHARDED / INTERLEAVED
→ 데이터를 메모리 공간에 어떻게 분산하는가?

L1 / DRAM
→ 어떤 메모리 공간에 두는가?
```

는 서로 다른 질문이다. ([Tenstorrent](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tensor.html "Tensor — TT-NN\&trade; documentation"))

### DRAM과 L1의 역할 구분

실행 자체는 가능한 경우가 많지만, 연산 사이에 데이터를 계속 DRAM으로 내보냈다가 다시 읽으면 데이터 이동이 많아집니다.

가능한 중간 결과를 L1에 유지하면 이런 이동을 줄일 수 있다. 하지만 L1 용량은 제한되므로 모두 넣을 수는 없다. 컴파일러나 직접 작성한 TTNN 설정이 이 균형을 다룹니다. ([Tenstorrent](https://docs.tenstorrent.com/tt-mlir/specs/ttnn-optimizer.html "TTNN Optimizer — TT-MLIR documentation"))

Tensix의 L1은 **CPU의 자동 관리 캐시와 동일하게 해석해서는 안 된다.** TT-Metal에서는 kernel 사이에 데이터를 전달할 버퍼 등을 소프트웨어가 명시적으로 구성한다. ([Tenstorrent](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html "Matmul (Single Core) — TT-Metalium\&trade; documentation"))

---

## 14. Fusion과 Decomposition

### Fusion

우리 예제는:

```python
y = torch.relu(x @ w + b)
```

이다.

각 연산이 중간 결과를 외부 메모리에 쓴다고 가정하면:

```text
Matmul
→ 중간 결과 쓰기
→ Add가 다시 읽기
→ 중간 결과 쓰기
→ ReLU가 다시 읽기
→ 최종 결과 쓰기
```

가 된다.

가능한 구현에서는 편향 더하기와 ReLU를 행렬곱의 후처리에 결합해:

```text
Matmul
→ 내부에서 bias 추가
→ 내부에서 ReLU
→ 최종 결과 쓰기
```

처럼 처리할 수 있다.

이것이 fusion의 기본 아이디어이다.

다만 **어떤 연산을 실제로 합칠 수 있는지는 하드웨어, 자료형, shape, 구현에 따라 다릅니다.** TT-XLA는 FX 수준의 패턴 변환과 StableHLO composite 등을 이용해 최적화된 TTNN 연산으로 연결할 기회를 보존한다. ([Tenstorrent](https://docs.tenstorrent.com/tt-xla/fusing_and_composite_ops.html "Fusing and Composite Ops — TT-XLA documentation"))

### Decomposition

반대로 복잡한 연산을 기본 연산들로 나누기도 한다.

예를 들어 정규화 연산은 개념적으로:

```text
평균 계산
→ 평균 빼기
→ 분산 계산
→ 정규화
→ scale과 bias 적용
```

으로 나눌 수 있다.

컴파일러가 고수준 연산을 직접 처리하지 못하더라도 기본 연산들을 처리할 수 있으면 실행 경로를 만들 수 있다.

문제는 하드웨어에 이미 좋은 정규화 구현이 있을 때이다. 너무 잘게 나누면 그 최적화된 구현을 사용하기 어려워질 수 있다.

그래서 composite 같은 장치를 통해 **“이 기본 연산 묶음은 원래 하나의 정규화 연산이다”라는 의미를 보존**하기도 한다. ([Tenstorrent](https://docs.tenstorrent.com/tt-xla/fusing_and_composite_ops.html "Fusing and Composite Ops — TT-XLA documentation"))

즉 컴파일러는 무조건 합치기만 하거나 나누기만 하는 것이 아니라, **다음 단계가 잘 처리할 수 있도록 표현을 조정**한다.

---

## 15. TTNN과 TT-Metal의 역할

### TTNN 텐서 연산 라이브러리

직접 TTNN을 사용하면 다음처럼 작성한다.

```python
y = ttnn.matmul(x, w)
```

여기서는 이미 Tenstorrent용 연산 API를 선택하였다.

따라서 일반적인 직접 호출 경로에서는:

```text
PyTorch 코드 포착
→ StableHLO 변환
→ TTIR 변환
```

같은 프런트엔드 변환이 필요하지 않는다.

하지만 TTNN이 하드웨어 명령어 그 자체는 아니다. `ttnn.matmul()` 아래에는 해당 연산을 실행할 프로그램과 kernel을 구성하는 구현이 있다. TTNN은 program configuration, compute configuration, memory configuration 등의 설정을 받는다. ([Tenstorrent](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.matmul.html "ttnn.matmul — TT-NN\&trade; documentation"))

또한 모든 TTNN 호출에서 코어를 직접 지정해야 하는 것은 아니다. 연산 구현이 기본값이나 자동 선택을 제공하고, 필요하면 사용자가 이를 조정하는 구조이다.

### TT-Metal 실행 계층

TT-Metal 수준에서는 실제 데이터 흐름을 관리한다.

단순화하면:

```text
DRAM의 입력
    ↓
Reader kernel
    ↓
L1의 입력 Circular Buffer
    ↓
Compute kernel
    ↓
L1의 출력 Circular Buffer
    ↓
Writer kernel
    ↓
DRAM 또는 다음 연산
```

이다. ([Tenstorrent](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html "Matmul (Single Core) — TT-Metalium\&trade; documentation"))

### Circular Buffer

Circular Buffer는 생산자와 소비자 사이에서 데이터를 전달하는 재사용 가능한 버퍼이다.

Reader가 데이터를 넣고, compute가 그것을 소비한다.

```text
Reader: 다음 입력 타일 준비
Compute: 준비된 입력 타일 계산
Writer: 계산된 출력 타일 저장
```

버퍼 공간을 적절히 나누면 입력을 가져오는 동안 이전 데이터를 계산할 수 있다. 이를 **overlap**, 즉 작업 겹치기라고 한다.

Double buffering은 그 방법 중 하나이다. 한쪽 데이터를 처리하는 동안 다른 쪽에 다음 데이터를 준비한다. 대신 버퍼가 더 필요하다. ([Tenstorrent](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html "Matmul (Single Core) — TT-Metalium\&trade; documentation"))

### L1 및 sharding 오류 사례

관찰된 오류는 다음과 같다.

```text
Statically allocated circular buffers ...
beyond max L1 size ...
```

는 **선택한 실행 구성이 해당 코어의 L1에 들어가지 않는다**는 오류였다.

이는 전체 보드 DRAM 용량이 부족하다는 뜻과 다릅니다.

그리고:

```text
Conv2d supports ... Sharded Layouts
but got ... INTERLEAVED
```

는 해당 실행 경로에서 요구한 sharding 설정과 전달된 설정이 맞지 않았다는 뜻이다.

**이 로그만으로 “MaxPool은 항상 INTERLEAVED를 출력한다”거나 “모든 Conv는 INTERLEAVED 입력을 받을 수 없다”고 일반화하면 안 된다.** 앞서 그 원인을 단정한 설명은 과도하였다.

---

## 16. 학습과 Autograd

앞선 절에서는 주로 forward 계산을 다루었다.

하지만 학습에는 다음 과정이 필요하다.

```text
Forward
→ Loss 계산
→ Backward
→ Weight update
```

### Backward는 forward를 거꾸로 실행하는 것이 아니다

간단한 예를 살펴본다.

```math
y=wx+b
```

```math
L=\frac{1}{2}(y-t)^2
```

여기서 `t`는 정답이다.

학습에서는 `w`와 `b`를 조금 바꾸었을 때 loss가 어떻게 변하는지 알아야 한다.

미분하면:

```math
\frac{\partial L}{\partial y}=y-t
```

```math
\frac{\partial L}{\partial w}=(y-t)x
```

```math
\frac{\partial L}{\partial b}=y-t
```

이다.

이런 미분값이 **gradient**이다.

예를 들어:

```text
x = 2
w = 3
b = 1
t = 5
```

라면:

```text
y = 3×2 + 1 = 7
오차 = 7 - 5 = 2

w의 gradient = 2×2 = 4
b의 gradient = 2
```

학습률이 0.1인 단순 경사하강법으로 갱신하면:

```text
새 w = 3 - 0.1×4 = 2.6
새 b = 1 - 0.1×2 = 0.8
```

이다.

### Autograd가 하는 일

큰 모델에는 이런 계산이 수천 개 연결되어 있다.

Autograd는 각 연산에 정의된 미분 규칙과 연쇄법칙을 이용해 gradient를 계산한다. 사용자가 전체 미분식을 매번 손으로 작성하지 않도록 하는 기능이다. PyTorch의 autograd도 계산 그래프와 연쇄법칙을 이용한다. ([PyTorch Docs](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html "Automatic Differentiation with torch.autograd — PyTorch Tutorials 2.13.0+cu130 documentation"))

여기서 두 일을 구분해야 한다.

```text
Autograd:
어떤 gradient 계산이 필요한가?

Compiler:
그 계산을 하드웨어에서 어떻게 실행할 것인가?
```

**자동미분과 하드웨어 컴파일은 같은 작업이 아니다.**

### 행렬곱의 backward도 다시 행렬곱이다

```math
Z=XW
```

이고 `G=\partial L/\partial Z`라고 하면:

```math
\frac{\partial L}{\partial X}=GW^\mathsf{T}
```

```math
\frac{\partial L}{\partial W}=X^\mathsf{T}G
```

이다.

그래서 backward도 하드웨어 입장에서는 또 다른 텐서 연산들의 집합이다.

학습용 컴파일 경로에서는 backward 계산도 그래프로 다룰 수 있다. PyTorch의 compiled training에는 AOTAutograd/AOTDispatcher처럼 backward 그래프를 포착하는 구성요소도 있다. 구체적인 사용 방식은 backend에 따라 달라집니다. ([PyTorch Docs](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_backward.html "torch.compile has different autograd semantics — PyTorch main documentation"))

---

## 17. TTML은 이 중 어디에 들어가는가

TTML은 **TTNN 기반 연산에 자동미분과 학습 기능을 연결하는 tt-train의 학습 프레임워크 계층**이다.

일반적인 구조는 다음과 같다.

```text
TTML
 ├─ 학습용 Tensor / Parameter
 ├─ 연산 사이의 미분 관계
 ├─ Backward 실행
 └─ Optimizer

          ↓

TTNN 연산 / 하드웨어용 구현

          ↓

TT-Metal

          ↓

P100a
```

TTML은 연산의 forward뿐 아니라 대응하는 backward와 optimizer를 연결해 실제 학습 루프를 구성한다. ([Tenstorrent](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/ct1-understanding-training/ "Understanding Custom Training — Tenstorrent Lessons"))

### TTNN 연산과 backward 지원

**“TTNN에는 backward가 전혀 없다”는 표현도 정확하지 않는다.**

TTNN에는 `ttnn.relu_bw` 같은 개별 backward 연산이 존재한다. ([Tenstorrent](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.relu_bw.html "ttnn.relu_bw — TT-NN\&trade; documentation"))

다만 다음 두 가지는 다릅니다.

```text
ReLU의 gradient를 계산하는 함수가 있다.
```

와:

```text
모델 전체의 연산 관계를 기록하고,
loss에서 모든 parameter까지 자동으로 gradient를 전달한다.
```

두 번째가 autograd 프레임워크의 역할이다.

따라서 raw TTNN 연산을 아무렇게나 연결했다고 해서 전체 학습 그래프가 자동으로 만들어지는 것은 아니다. TTML에서 지원하는 미분 가능한 연산을 사용하거나, 필요한 backward 연결을 구현해야 한다. ([Tenstorrent](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/ct1-understanding-training/ "Understanding Custom Training — Tenstorrent Lessons"))

### TTML과 TT-MLIR의 구분

```text
TTML / tt-train
→ 학습 프레임워크

TT-MLIR
→ Tenstorrent용 컴파일러 인프라
```

**TTML은 TT-MLIR의 줄임말이 아니다.**

---

## 18. 두 실행 경로의 비교

### PyTorch + TT-XLA

사용자는 PyTorch 모델과 학습 코드를 작성한다.

```text
PyTorch 모델 / PyTorch 자동미분
               │
               ├─ 선택한 경로에 따라 Dynamo / FX
               │
               ▼
         PyTorch/XLA 처리
               │
               ▼
           StableHLO
               │
        TT PJRT plugin을 통해
               ▼
            TT-MLIR
               │
        TTIR → TTNN dialect
               │
               ▼
         실행용 프로그램
               │
               ▼
      TT 런타임 / TTNN / TT-Metal
               │
               ▼
              P100a
```

이 경로에서는 **PyTorch에서 표현한 계산을 Tenstorrent에 맞는 계산으로 변환하는 과정**이 있다. ([Tenstorrent](https://docs.tenstorrent.com/tt-xla/getting_started.html "Getting Started — TT-XLA documentation"))

### TTML + TTNN

사용자는 Tenstorrent용 학습 연산을 사용해 모델을 구성한다.

```text
TTML 모델 / TTML 자동미분
               │
               ▼
      지원되는 forward/backward 구현
               │
               ▼
          TTNN / TT-Metal
               │
               ▼
              P100a
```

이 경로에서는 이미 Tenstorrent용 학습 구현을 사용하므로, PyTorch 계산을 StableHLO로 바꾸는 프런트엔드 과정이 필요하지 않는다. ([Tenstorrent](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/ct1-understanding-training/ "Understanding Custom Training — Tenstorrent Lessons"))

그러나 **TTML을 쓴다고 모든 미분을 직접 구현해야 하는 것은 아니다.** 지원되는 연산의 backward는 프레임워크 구현을 사용한다.

반대로 **TTML이 있다고 모든 TTNN 연산을 자동으로 학습할 수 있는 것도 아니다.** 필요한 연산의 autograd 지원 여부는 별도로 확인해야 한다.

---

## 19. 컴파일과 실행 캐시

앞서 “중간 단계가 많아서 XLA가 느리다”고 단순하게 설명한 부분은 정확하지 않았다.

실제로는 **첫 실행과 반복 실행을 구분**해야 한다.

### 첫 실행

조건에 따라 다음 작업들이 발생할 수 있다.

```text
그래프 포착
→ 중간 표현 생성
→ 최적화와 변환
→ 실행 프로그램 준비
→ 필요한 kernel 컴파일
→ 장치 실행
```

### 같은 조건의 후속 실행

이미 만든 결과를 재사용할 수 있다.

```text
기존 컴파일 결과 확인
→ 새 입력으로 실행
```

PyTorch/XLA는 그래프가 바뀌지 않으면 컴파일 결과를 재사용할 수 있고, TT-XLA 성능 문서도 초기 컴파일·kernel 준비 비용과 후속 실행을 구분한다. ([PyTorch Docs](https://docs.pytorch.org/xla/release/r2.4/index.html "PyTorch/XLA documentation — PyTorch/XLA master documentation"))

### Weight 갱신과 재컴파일

일반적으로 **weight의 숫자가 바뀐다는 사실만으로 재컴파일할 필요는 없다.**

예를 들어:

```text
[64, 128] BF16 weight
```

의 숫자는 바뀌어도:

```text
입력 shape
weight shape
자료형
계산 구조
```

가 같다면 같은 실행 계획에 새로운 weight 값을 넣을 수 있다.

반면 batch shape, 계산 경로, 특정 정적 설정 등이 바뀌어 기존 계획이 유효하지 않으면 새로운 컴파일이 필요할 수 있다. ([PyTorch Docs](https://docs.pytorch.org/docs/stable/generated/torch.compile "torch.compile — PyTorch 2.14 documentation"))

### TTNN 직접 실행도 컴파일이 있다

TTNN을 직접 쓴다고 컴파일이 완전히 사라지는 것도 아니다.

TTNN 연산에 필요한 kernel을 처음 준비하고 이후 재사용하는 과정이 있다. TTNN 행렬곱 학습 자료도 초기 kernel 컴파일과 program cache를 설명한다. ([Tenstorrent](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tutorials/ttnn_tutorials/002.html "Matrix Multiplication — TT-NN  documentation"))

실험 과정에서 나타나는 컴파일은 다음과 같이 구분된다.

| 종류 | 준비 대상 |
| ----------------------- | ----------------------------- |
| `build_metal.sh`의 소스 빌드 | TTNN·TTML 같은 라이브러리와 실행 구성요소   |
| 모델 그래프 컴파일              | 특정 모델 계산을 장치 실행 프로그램으로 변환     |
| Kernel JIT 컴파일          | 필요한 장치용 kernel 구현을 실행 가능하게 준비 |

즉 **라이브러리를 설치할 때 한 빌드와, 모델을 처음 실행할 때의 컴파일을 같은 것으로 보면 안 된다.** ([GitHub](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/README.md "tt-metal/tt-train/README.md at main · tenstorrent/tt-metal · GitHub"))

---

## 20. TTML 경로의 성능 차이에 대한 가설

가능한 설명은 다음과 같다.

직접 사용하는 TTML 구현이 해당 모델에 더 적합한 연산 조합을 사용했을 수 있다. TT-XLA 쪽의 메모리 배치나 fusion 결과가 덜 효율적이었을 수도 있다. 초기 컴파일 시간을 포함했거나 재컴파일이 반복됐을 가능성도 있다.

이들은 **확인해야 할 가설이지, 지금까지의 정보만으로 확정한 원인은 아니다.** 컴파일러의 메모리 최적화, 초기 준비 비용, host-device 통신 비용은 각각 성능에 영향을 줄 수 있다. ([Tenstorrent](https://docs.tenstorrent.com/tt-mlir/specs/ttnn-optimizer.html "TTNN Optimizer — TT-MLIR documentation"))

특히 다음 두 비교는 구분해야 한다.

```text
같은 VGG11 학습
TT-XLA vs TTML
→ 소프트웨어 경로 비교가 가능
```

```text
TT-XLA의 VGG11 학습
vs
TTML의 NanoGPT 학습
→ 서로 다른 모델의 비교
```

두 번째라면 빠르다는 관찰만으로 TTML 경로 자체가 더 빠르다고 결론 낼 수 없다.

### 완료까지 기다렸는지도 중요하다

가속기 실행은 비동기일 수 있다.

```text
CPU: 작업을 요청하고 다음 코드로 진행
P100a: 요청받은 계산을 아직 수행 중
```

이 상태에서 CPU 시간만 재면 실제 계산보다 짧게 측정할 수 있다. PJRT도 비동기 실행과 결과 준비 상태를 다루며, TTNN에는 장치 완료를 기다리는 `synchronize_device`가 있다. ([OpenXLA Project](https://openxla.org/xla/pjrt/cpp_api_overview "PJRT C++ Device API Overview  |  OpenXLA Project"))

또 1,000회를 연속 제출하고 마지막에 한 번 기다려 얻은:

```text
총시간 / 1000
```

은 **연속 처리했을 때의 평균 처리 시간**이다.

매번 한 번 실행하고 완료를 기다리는 **단일 요청 지연시간**과는 다를 수 있다.

결국 성능을 판단할 때는 “중간 단계가 몇 개인가”보다 **최종적으로 어떤 연산·데이터 이동·동기화가 발생했는가**가 중요한다.

---

## 21. 소프트웨어 계층별 역할 정리

다음 표는 각 구성요소를 역할에 따라 정리한다.

| 용어 | 역할 |
| --------------- | ------------------------------------- |
| PyTorch         | 텐서 계산과 모델·학습을 작성하는 프레임워크              |
| `torch.compile` | 그래프 포착과 선택한 backend의 최적화를 연결하는 API    |
| TorchDynamo     | Python 실행에서 PyTorch 계산을 포착하는 구성요소     |
| FX Graph        | PyTorch 연산과 연결 관계를 표현하는 그래프           |
| PyTorch/XLA     | PyTorch를 XLA 계열 실행 환경에 연결하는 구현        |
| StableHLO       | 프레임워크와 컴파일러 사이에서 사용하는 텐서 계산 표현        |
| PJRT            | 장치·버퍼·컴파일·실행을 요청하는 공통 인터페이스           |
| MLIR            | 여러 수준의 IR과 변환을 구현하는 컴파일러 기반 기술        |
| TT-MLIR         | Tenstorrent용 컴파일러 인프라                 |
| TTIR            | TT-MLIR의 고수준 텐서 연산 표현                 |
| TTNN dialect    | TTNN API에 가까운 컴파일러 내부 표현              |
| TTNN            | Tenstorrent용 텐서·연산 라이브러리              |
| TTML / tt-train | Tenstorrent용 자동미분·학습 프레임워크            |
| TT-Metal        | 장치 프로그램, kernel, 메모리·데이터 이동 등을 다루는 계층 |

이 역할 구분은 PyTorch, OpenXLA, MLIR, Tenstorrent의 공식 설명에 대응한다. ([PyTorch Docs](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html "torch.compiler — PyTorch main documentation"))

전체 실행 경로의 핵심 관계는 다음과 같다.

**PyTorch 코드**에서는:

> “`Y=\operatorname{ReLU}(XW+b)`를 계산하라.”

라고 기술한다.

**그래프와 IR**에서는:

> “행렬곱 결과를 덧셈에 전달하고, 그 결과를 ReLU에 전달한다. 각 텐서의 shape와 자료형은 이렇다.”

라고 표현한다.

**컴파일러와 연산 구현**에서는:

> “어떤 연산을 합칠지, 데이터를 어떻게 배치할지, 어떤 실행 구성을 사용할지 결정한다.”

가 된다.

**런타임과 TT-Metal**에서는:

> “실제 버퍼를 준비하고, 데이터를 옮기고, kernel을 실행하고, 완료를 관리한다.”

가 된다.

**TTNN을 직접 쓴다는 것은 이 전체 시스템을 없애는 것이 아니라, 계산을 작성하는 출발점을 Tenstorrent 연산 라이브러리 수준으로 옮기는 것이다. TTML은 그 출발점에서 학습에 필요한 자동미분과 optimizer를 제공하는 계층이다.**
