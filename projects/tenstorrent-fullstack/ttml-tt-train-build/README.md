# Tenstorrent P100a — TTML / tt-train Setup

P100a에서 TTNN 기반 학습을 진행하기 위해 별도의 `ttml` Conda 환경을 만들고, `tt-metal`의 TTML / tt-train을 소스 빌드한 과정입니다.

## 1. Conda 환경 생성

```bash
conda create -n ttml python=3.12 -y
conda activate ttml

python --version
which python

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

정상 출력:

```text
Python 3.12.x
/home/woo/anaconda3/envs/ttml/bin/python
```

---

## 2. tt-metal clone

처음 설치하는 경우:

```bash
cd ~

git clone https://github.com/tenstorrent/tt-metal.git \
    --recurse-submodules

cd ~/tt-metal
```

이미 clone했다면:

```bash
cd ~/tt-metal

git pull

git submodule update \
    --init \
    --recursive
```

`tt-train`도 `tt-metal` 소스 트리 안에서 함께 관리됩니다.

---

## 3. 시스템 의존성 설치

처음 한 번만 실행합니다.

```bash
cd ~/tt-metal

sudo ./install_dependencies.sh
```

이 스크립트가 CMake와 Clang 등 `tt-metal` 빌드에 필요한 시스템 의존성을 준비합니다.

확인:

```bash
cmake --version

clang-20 --version
clang++-20 --version
```

최소 CMake 요구 버전을 충족하지 못하거나 `clang-20`이 없다면 이 단계에서 해결해야 합니다.

---

## 4. Conda 환경에 Python 의존성 설치

```bash
conda activate ttml
cd ~/tt-metal

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

`tt-metal`의 `pyproject.toml`에서 Python 패키지는 `ttnn`으로 정의되어 있습니다. 이 명령으로 현재 source tree의 TTNN Python 패키지를 사용합니다.

확인:

```bash
python -c "import ttnn; print(ttnn.__file__)"
```

최종적으로 다음 계열의 경로가 출력돼야 합니다.

```text
/home/woo/tt-metal/ttnn/ttnn/__init__.py
```

> `pjrt_plugin_tt/...`가 출력되면 안 됩니다.

---

## 5. TTML / tt-train 빌드

반드시 `ttml` 환경에서 실행합니다.

```bash
conda activate ttml

cd ~/tt-metal

rm -rf build_Release

./build_metal.sh \
    -b Release \
    --build-tt-train
```

> `rm -rf build_Release`는 기존 Release build 결과를 삭제합니다.

완료되면 현재 브랜치에서는 다음 경로를 확인합니다.

```bash
ls ~/tt-metal/build_Release/ttml/
```

다음 파일이 있어야 합니다.

```text
_ttml.abi3.so
```

---

## 6. TTML Python 경로 등록

먼저 현재 Conda environment의 `site-packages` 경로를 구합니다.

```bash
SITE_PACKAGES=$(python -c \
'import site; print(site.getsitepackages()[0])')

echo "$SITE_PACKAGES"
```

예상 경로:

```text
/home/woo/anaconda3/envs/ttml/lib/python3.12/site-packages
```

TTML Python source와 빌드된 `_ttml` extension 경로를 `.pth` 파일로 등록합니다.

```bash
echo "$HOME/tt-metal/tt-train/sources/ttml" \
    > "$SITE_PACKAGES/ttml.pth"

echo "$HOME/tt-metal/build_Release/ttml" \
    > "$SITE_PACKAGES/_ttml.pth"
```

공식 `create_venv.sh`도 TTML Python source와 빌드된 `_ttml` extension을 `.pth`로 Python 환경에 추가하는 방식을 사용합니다.

확인:

```bash
cat "$SITE_PACKAGES/ttml.pth"
cat "$SITE_PACKAGES/_ttml.pth"
```

출력:

```text
/home/woo/tt-metal/tt-train/sources/ttml
/home/woo/tt-metal/build_Release/ttml
```

---

## 7. 런타임 환경 변수

```bash
export TT_METAL_HOME="$HOME/tt-metal"

export TT_METAL_RUNTIME_ROOT="$HOME/tt-metal"

export LD_LIBRARY_PATH="$HOME/tt-metal/build_Release/lib:${LD_LIBRARY_PATH:-}"
```

이 `LD_LIBRARY_PATH`가 없으면 다음과 같은 오류가 발생할 수 있습니다.

```text
libtracy.so.0.14.1: cannot open shared object file
```

---

## 8. import 검증

```bash
python - <<'PY'
import sys
import ttnn
import _ttml
import ttml

print("Python :", sys.executable)
print("TTNN   :", ttnn.__file__)
print("_TTML  :", _ttml.__file__)
print("TTML   :", ttml.__file__)
print()
print("TTNN + TTML OK")
PY
```

목표 출력:

```text
Python : /home/woo/anaconda3/envs/ttml/bin/python
TTNN   : /home/woo/tt-metal/ttnn/ttnn/__init__.py
_TTML  : /home/woo/tt-metal/build_Release/ttml/_ttml.abi3.so
TTML   : /home/woo/tt-metal/tt-train/sources/ttml/ttml/__init__.py

TTNN + TTML OK
```

---

## 9. `ttml` 명령 하나로 환경 활성화

`~/.bashrc`를 엽니다.

```bash
nano ~/.bashrc
```

맨 아래에 다음 alias를 추가합니다.

```bash
alias ttml='conda activate ttml && SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])") && echo "$HOME/tt-metal/tt-train/sources/ttml" > "$SITE_PACKAGES/ttml.pth" && echo "$HOME/tt-metal/build_Release/ttml" > "$SITE_PACKAGES/_ttml.pth" && unset PYTHONPATH && export TT_METAL_HOME="$HOME/tt-metal" && export TT_METAL_RUNTIME_ROOT="$HOME/tt-metal" && export LD_LIBRARY_PATH="$HOME/tt-metal/build_Release/lib:${LD_LIBRARY_PATH:-}"'
```

저장 후 설정을 다시 불러옵니다.

```bash
source ~/.bashrc
```

이제 어느 터미널에서든 다음 명령으로 환경을 활성화합니다.

```bash
ttml
```

```text
(base)
  ↓
ttml
  ↓
(ttml)
```

동시에 다음 항목이 설정됩니다.

- TTML source path
- `_ttml.so` path
- `TT_METAL_HOME`
- `TT_METAL_RUNTIME_ROOT`
- `LD_LIBRARY_PATH`

확인:

```bash
ttml

python -c \
'import ttnn, ttml, _ttml; print("TTML READY")'
```

## 최종 구조

```text
~/tt-metal/
│
├── ttnn/
│   └── ttnn/
│       └── Python TTNN
│
├── tt-train/
│   └── sources/
│       └── ttml/
│           └── Python TTML
│
└── build_Release/
    ├── lib/
    │   ├── libtt_metal.so
    │   ├── libtracy.so...
    │   └── ...
    │
    └── ttml/
        └── _ttml.abi3.so
```

## 다음 단계

- [ ] TTML 최소 training example 실행
- [ ] forward/backward 및 optimizer 검증
- [ ] loss 감소 확인
- [ ] VGG11 training으로 확장
- [ ] training latency, throughput, memory 및 정확도 기록
