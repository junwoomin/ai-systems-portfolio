# Tenstorrent P100a — TTML / tt-train Build Setup

P100a에서 TTNN 기반 학습을 수행하기 위해 `tt-metal`의 `tt-train / TTML`을 소스 빌드한 과정과 오류 해결 기록입니다.

## 환경

```text
OS        : Ubuntu
Device    : Tenstorrent P100a
Arch      : Blackhole
Python    : 3.12
Conda env : tt
```

기존 Conda 환경을 그대로 사용했습니다.

```bash
conda activate tt
```

---

## 1. tt-metal 다운로드

```bash
cd ~

git clone https://github.com/tenstorrent/tt-metal.git \
    --recurse-submodules

cd ~/tt-metal
```

이미 clone되어 있다면 다음과 같이 갱신합니다.

```bash
cd ~/tt-metal

git pull

git submodule update \
    --init \
    --recursive
```

---

## 2. CMake 버전 문제 해결

처음 빌드할 때 다음 오류가 발생했습니다.

```text
CMake 3.24 or higher is required.
You are running version 3.22.1
```

Conda 환경 내부의 CMake와 Ninja를 설치했습니다.

```bash
conda activate tt
conda install -c conda-forge "cmake>=3.24" ninja -y
```

확인:

```bash
which cmake
cmake --version
```

Conda 환경의 CMake가 잡혀야 합니다.

```text
/home/woo/anaconda3/envs/tt/bin/cmake
```

---

## 3. Clang 20 설치

다음 빌드에서는 아래 오류가 발생했습니다.

```text
The CMAKE_C_COMPILER:

    clang-20

was not found in the PATH.

The CMAKE_CXX_COMPILER:

    clang++-20

was not found in the PATH.
```

`tt-metal` dependency 설치 스크립트를 실행했습니다.

```bash
cd ~/tt-metal
sudo ./install_dependencies.sh
```

설치 후 확인:

```bash
which clang-20
which clang++-20

clang-20 --version
clang++-20 --version
```

정상적으로 설치되면 다음과 같은 경로가 출력됩니다.

```text
/usr/bin/clang-20
/usr/bin/clang++-20
```

---

## 4. 이전 CMake build 정리

CMake 또는 compiler 구성을 변경했으므로 이전 build cache를 제거했습니다.

> 아래 명령은 기존 `build_Release` 결과를 삭제합니다.

```bash
cd ~/tt-metal
rm -rf build_Release
```

---

## 5. TTML / tt-train 빌드

TTNN뿐 아니라 training stack까지 빌드하기 위해 다음 명령을 사용했습니다.

```bash
cd ~/tt-metal
./build_metal.sh -b Release --build-tt-train
```

빌드 설정에서 Conda Python이 정상적으로 선택되었습니다.

```text
Python3_EXECUTABLE=/home/woo/anaconda3/envs/tt/bin/python3
Python3_INCLUDE_DIR=/home/woo/anaconda3/envs/tt/include/python3.12
Python3_LIBRARY=/home/woo/anaconda3/envs/tt/lib/libpython3.12.so
```

따라서 `tt` Conda 환경의 Python 3.12를 기준으로 Python binding이 빌드됩니다.

---

## 6. emsdk 다운로드 오류 해결

빌드 중 다음 오류가 한 번 발생했습니다.

```text
xz: (stdin): Unexpected end of input
tar: Unexpected EOF in archive
tar: Error is not recoverable
emsdk install failed
```

깨진 파일은 다음 CPM cache 내부의 Emscripten archive였습니다.

```text
~/tt-metal/.cpmcache/emsdk/...
```

다운로드가 중단되어 archive가 손상된 경우, emsdk cache만 제거합니다.

> 아래 명령은 다운로드된 emsdk cache를 삭제하며 다음 빌드에서 다시 내려받습니다.

```bash
cd ~/tt-metal
rm -rf .cpmcache/emsdk
```

필요하면 build directory도 함께 정리합니다.

```bash
rm -rf build_Release
```

그 후 다시 빌드합니다.

```bash
./build_metal.sh -b Release --build-tt-train
```

이 환경에서는 동일한 빌드 명령을 다시 실행한 뒤 정상적으로 진행됐습니다.

---

## 최종 명령 순서

처음 설치할 때의 전체 순서입니다.

```bash
# Conda
conda activate tt

# CMake / Ninja
conda install -c conda-forge "cmake>=3.24" ninja -y

# tt-metal
cd ~
git clone https://github.com/tenstorrent/tt-metal.git \
    --recurse-submodules
cd ~/tt-metal

# System dependencies / clang-20
sudo ./install_dependencies.sh

# 확인
cmake --version
clang-20 --version
clang++-20 --version
python --version

# Build cache 초기화
rm -rf build_Release

# TTNN + TTML / tt-train build
./build_metal.sh -b Release --build-tt-train
```

---

## 문제 해결 요약

### CMake가 너무 오래된 경우

```text
CMake 3.24 or higher is required
```

```bash
conda install -c conda-forge "cmake>=3.24" ninja -y
```

### clang-20을 찾지 못하는 경우

```text
clang-20 was not found in PATH
```

```bash
sudo ./install_dependencies.sh

which clang-20
which clang++-20
```

### emsdk 압축 파일이 깨진 경우

```text
xz: Unexpected end of input
tar: Unexpected EOF
```

```bash
cd ~/tt-metal
rm -rf .cpmcache/emsdk
./build_metal.sh -b Release --build-tt-train
```

---

## 전체 스택

이번 빌드의 목적은 기존 TT-XLA 경로가 아니라 TTML을 이용해 P100a에서 training stack을 직접 다루는 것입니다.

```text
Application / Model
        ↓
TTML / tt-train
        ↓
Autograd / Optimizer
        ↓
TTNN
        ↓
TT-Metal
        ↓
P100a (Blackhole)
```

TT-XLA 경로와는 구분합니다.

```text
PyTorch
   ↓
Torch-XLA
   ↓
StableHLO
   ↓
TT-MLIR
   ↓
TTNN / TT-Metal
   ↓
P100a
```

## 다음 단계

- [ ] TTML 최소 training example 검증
- [ ] forward/backward 및 optimizer 동작 확인
- [ ] loss 감소 여부 확인
- [ ] VGG11 training으로 확장
- [ ] batch latency, throughput, memory 및 정확도 기록
