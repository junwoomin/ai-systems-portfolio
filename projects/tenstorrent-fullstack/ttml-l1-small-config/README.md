# P100a TTML L1_SMALL Configuration Patch

## 목적

TTML의 기본 device 생성 경로가 `L1_SMALL = 0`으로 장치를 열어, convolution configuration tensor를 L1_SMALL에 배치할 수 없었습니다. 이 때문에 학습 코드에서 `config_tensors_in_dram=True`를 사용했고, DRAM 접근이 latency 병목일 가능성을 확인하기 위해 L1_SMALL 크기를 Python에서 지정할 수 있도록 TTML 경로를 수정했습니다.

> 실제 수정 대상은 **TTML 소스 5개 파일과 학습 코드 1개**입니다. 처음 정리한 “4개 소스 파일”과 달리 `nb_autograd.cpp`까지 포함하면 TTML 파일은 총 5개입니다.

## 변경 흐름

```text
train_ttml.py
  l1_small_size=24576
        ↓
nanobind
  nb_autograd.cpp
        ↓
AutoContext
  auto_context.hpp
  auto_context.cpp
        ↓
MeshDevice
  mesh_device.hpp
  mesh_device.cpp
        ↓
ttnn::distributed::open_mesh_device(
    shape,
    24576,
    ...
)
```

## 수정 파일

| 순서 | 파일 | 역할 |
|---:|---|---|
| 1 | `tt-train/sources/ttml/core/mesh_device.hpp` | `MeshDevice` API에 크기 인자 추가 |
| 2 | `tt-train/sources/ttml/core/mesh_device.cpp` | 실제 TTNN device 생성 함수로 값 전달 |
| 3 | `tt-train/sources/ttml/autograd/auto_context.hpp` | `AutoContext.open_device()` API 확장 |
| 4 | `tt-train/sources/ttml/autograd/auto_context.cpp` | `MeshDevice`까지 값 전달 |
| 5 | `tt-train/sources/ttml/nanobind/nb_autograd.cpp` | Python의 `l1_small_size=` 인자 노출 |
| 6 | `train_ttml.py` | P100a 실험값 24 KiB 지정 |

---

## 1. mesh_device.hpp

파일:

```text
tt-train/sources/ttml/core/mesh_device.hpp
```

`MeshDevice` 생성자에 `l1_small_size`를 추가합니다. 기존 호출과의 호환성을 유지하기 위해 기본값은 반드시 `0`으로 둡니다.

```cpp
explicit MeshDevice(
    const tt::tt_metal::distributed::MeshShape& shape,
    const std::vector<int>& device_ids,
    std::size_t l1_small_size = 0
);
```

기본값이 있으므로 기존 코드는 그대로 동작합니다.

```cpp
MeshDevice device(shape, {});
```

---

## 2. mesh_device.cpp

파일:

```text
tt-train/sources/ttml/core/mesh_device.cpp
```

기존에 고정된 `DEFAULT_L1_SMALL_SIZE` 대신 생성자에서 전달받은 `l1_small_size`를 사용합니다.

```cpp
MeshDevice::MeshDevice(
    const tt::tt_metal::distributed::MeshShape& shape,
    const std::vector<int>& device_ids,
    std::size_t l1_small_size) :
    m_mesh_device(ttnn::distributed::open_mesh_device(
        shape,
        l1_small_size,
        DEFAULT_TRACE_REGION_SIZE,
        /* num_command_queues=*/1,
        tt::tt_metal::DispatchCoreConfig{},
        /* offset=*/std::nullopt,
        /* physical_device_ids=*/device_ids)) {
    assert(m_mesh_device);
}
```

핵심 변경:

```diff
- DEFAULT_L1_SMALL_SIZE
+ l1_small_size
```

---

## 3. auto_context.hpp

파일:

```text
tt-train/sources/ttml/autograd/auto_context.hpp
```

`AutoContext::open_device()`에도 `l1_small_size`를 추가합니다. 기존 TTML 호출을 깨뜨리지 않도록 기본값 `0`을 둡니다.

```cpp
void open_device(
    const tt::tt_metal::distributed::MeshShape& mesh_shape =
        tt::tt_metal::distributed::MeshShape(1, 1),
    const std::vector<int>& device_ids = std::vector<int>{},
    std::size_t l1_small_size = 0);
```

---

## 4. auto_context.cpp

파일:

```text
tt-train/sources/ttml/autograd/auto_context.cpp
```

함수 인자를 추가하고 `MeshDevice` 생성자까지 전달합니다.

```cpp
void AutoContext::open_device(
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    const std::vector<int>& device_ids,
    std::size_t l1_small_size) {

    if (m_device) {
        throw std::runtime_error(
            "open_device was called after the device was created.");
    }

    m_mesh_shape = mesh_shape;

    m_device = std::make_unique<core::MeshDevice>(
        m_mesh_shape,
        device_ids,
        l1_small_size
    );
}
```

---

## 5. nb_autograd.cpp

파일:

```text
tt-train/sources/ttml/nanobind/nb_autograd.cpp
```

Python에서 `l1_small_size=`를 사용할 수 있도록 nanobind lambda에 인자를 추가합니다.

```cpp
[](AutoContext& self,
   nb::object mesh_shape_obj,
   nb::object device_ids_obj,
   std::size_t l1_small_size) {
```

변환한 값을 `AutoContext`로 전달합니다.

```cpp
self.open_device(mesh_shape, device_ids, l1_small_size);
```

Python 인자 정의에도 기본값을 추가합니다.

```cpp
nb::arg("mesh_shape") = nb::none(),
nb::arg("device_ids") = nb::none(),
nb::arg("l1_small_size") = 0,
"Open a mesh device");
```

---

## 6. train_ttml.py

P100a 실험값으로 24 KiB를 지정합니다.

```python
L1_SMALL_SIZE = 24 * 1024
```

device를 열 때 값을 전달합니다.

```python
context = ttml.autograd.AutoContext.get_instance()
context.set_seed(SEED)

context.open_device(
    device_ids=[DEVICE_ID],
    l1_small_size=L1_SMALL_SIZE,
)
```

### Conv2d 설정 확인

L1_SMALL을 예약해도 `config_tensors_in_dram=True`를 유지하면 해당 configuration tensor는 계속 DRAM에 배치됩니다. DRAM과 L1_SMALL을 비교하려는 실험에서는 조건을 명시적으로 분리해야 합니다.

DRAM 기준:

```python
conv_config = ttnn.Conv2dConfig(
    config_tensors_in_dram=True,
    # ...
)
```

L1_SMALL 기준:

```python
conv_config = ttnn.Conv2dConfig(
    config_tensors_in_dram=False,
    # ...
)
```

즉, **`l1_small_size=24 * 1024`와 `config_tensors_in_dram=False`가 함께 적용돼야 configuration tensor의 L1_SMALL 사용을 검증할 수 있습니다.**

---

## 기본값이 필요한 이유

두 public C++ header에 기본값을 넣지 않으면 기존 호출부가 새 인자를 제공하지 못해 빌드 오류가 발생합니다.

```cpp
std::size_t l1_small_size = 0
```

기본값이 필요한 위치:

- `mesh_device.hpp`
- `auto_context.hpp`

이 설정으로 기존 TTML 코드의 동작은 유지하면서 새 Python API에서만 선택적으로 L1_SMALL 크기를 지정할 수 있습니다.

---

## 재빌드

C++ source와 nanobind binding을 변경했으므로 TTML을 다시 빌드해야 합니다.

```bash
ttml
cd ~/tt-metal

./build_metal.sh \
    -b Release \
    --build-tt-train
```

완전한 clean build가 필요한 경우에만 기존 결과를 제거합니다.

```bash
rm -rf build_Release

./build_metal.sh \
    -b Release \
    --build-tt-train
```

> `rm -rf build_Release`는 기존 Release build 결과를 삭제합니다.

---

## Python API 검증

```bash
python - <<'PY'
import inspect
import ttml

context = ttml.autograd.AutoContext.get_instance()

print(ttml.__file__)
print(context.open_device)
print("TTML binding loaded")
PY
```

실제 device 검증:

```python
import ttml

context = ttml.autograd.AutoContext.get_instance()
context.open_device(
    device_ids=[0],
    l1_small_size=24 * 1024,
)

print("P100a opened with L1_SMALL = 24 KiB")
```

---

## 실험 설계

다른 조건은 고정하고 다음 두 경우를 비교합니다.

| 조건 | `l1_small_size` | `config_tensors_in_dram` |
|---|---:|---:|
| DRAM baseline | 0 | `True` |
| L1_SMALL experiment | 24 KiB | `False` |

필수 기록:

- model 및 input shape
- batch size
- dtype
- warm-up batch 수
- 측정 batch 수
- batch latency 평균, p50, p95
- throughput
- training loss 및 accuracy
- L1_SMALL OOM 발생 여부
- device temperature 및 power

## 결론

이 패치는 Python 학습 코드에서 지정한 L1_SMALL 크기를 TTML, AutoContext, MeshDevice를 거쳐 TTNN의 실제 mesh device 생성 단계까지 전달합니다. 기본값 `0`을 유지해 기존 코드 호환성을 보존하면서, P100a에서 DRAM configuration tensor와 L1_SMALL configuration tensor의 성능 차이를 직접 비교할 수 있게 합니다.
