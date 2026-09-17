#!/usr/bin/env python3
"""
P100a L1 SRAM <-> device DRAM effective bandwidth benchmark.

Measures the real TTNN data-movement path with synchronization:
  L1 sharded -> device DRAM
  device DRAM -> L1 sharded

The reported GB/s is effective payload bandwidth:
    logical tensor bytes / elapsed wall-clock time

It therefore includes TTNN allocation / memory-config conversion overhead
and is more useful for model latency analysis than a theoretical peak number.

Example:
    python l1_dram_bandwidth.py --size-mb 8 --repeat 30 --l1-small-kb 16
"""

import argparse
import math
import statistics
import time

import torch
import ttnn


BF16_BYTES = 2


def synchronize(device):
    ttnn.synchronize_device(device)


def tensor_logical_nbytes(tensor, bytes_per_element=BF16_BYTES):
    shape = tuple(int(v) for v in tensor.shape)
    return math.prod(shape) * bytes_per_element


def make_height_sharded_l1_config(device, logical_shape):
    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(x=int(grid.x), y=int(grid.y))

    return ttnn.create_sharded_memory_config(
        shape=logical_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def benchmark_l1_dram(device, size_mb=8, channels=64, repeat=20, warmup=3):
    # [1, 1, NHW, C], BF16. NHW를 맞춰 원하는 payload 크기를 만든다.
    target_bytes = int(size_mb * 1024 * 1024)

    # HEIGHT_SHARDED의 기존 create_sharded_memory_config()는
    # shard height(NHW)가 사용 core 수로 정확히 나누어져야 한다.
    grid = device.compute_with_storage_grid_size()
    total_cores = int(grid.x) * int(grid.y)

    target_nhw = max(1, target_bytes // (channels * BF16_BYTES))
    nhw = ((target_nhw + total_cores - 1) // total_cores) * total_cores

    logical_shape = (1, 1, nhw, channels)

    host = torch.randn(logical_shape, dtype=torch.bfloat16)
    host_tt = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    dram_source = ttnn.to_device(
        host_tt,
        device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    synchronize(device)

    l1_config = make_height_sharded_l1_config(device, logical_shape)

    l1_source = ttnn.to_memory_config(
        dram_source,
        l1_config,
    )
    synchronize(device)

    payload_bytes = tensor_logical_nbytes(l1_source)

    print("=" * 72)
    print("P100a L1 SRAM <-> DEVICE DRAM")
    print("=" * 72)
    print(f"Compute grid        : {int(device.compute_with_storage_grid_size().x)} x {int(device.compute_with_storage_grid_size().y)}")
    print(f"Logical shape       : {logical_shape}")
    print(f"Payload             : {payload_bytes / 1024**2:.3f} MiB")
    print(f"L1 memory config    : {l1_source.memory_config()}")
    print(f"DRAM memory config  : {dram_source.memory_config()}")
    print()

    # Warm-up
    for _ in range(warmup):
        tmp_dram = ttnn.to_memory_config(l1_source, ttnn.DRAM_MEMORY_CONFIG)
        synchronize(device)
        tmp_l1 = ttnn.to_memory_config(tmp_dram, l1_config)
        synchronize(device)
        ttnn.deallocate(tmp_dram)
        ttnn.deallocate(tmp_l1)

    l1_to_dram_times = []
    for _ in range(repeat):
        synchronize(device)
        t0 = time.perf_counter()
        tmp = ttnn.to_memory_config(l1_source, ttnn.DRAM_MEMORY_CONFIG)
        synchronize(device)
        dt = time.perf_counter() - t0
        l1_to_dram_times.append(dt)
        ttnn.deallocate(tmp)

    dram_to_l1_times = []
    for _ in range(repeat):
        synchronize(device)
        t0 = time.perf_counter()
        tmp = ttnn.to_memory_config(dram_source, l1_config)
        synchronize(device)
        dt = time.perf_counter() - t0
        dram_to_l1_times.append(dt)
        ttnn.deallocate(tmp)

    avg_l1_dram = statistics.mean(l1_to_dram_times)
    med_l1_dram = statistics.median(l1_to_dram_times)
    avg_dram_l1 = statistics.mean(dram_to_l1_times)
    med_dram_l1 = statistics.median(dram_to_l1_times)

    print(f"L1  -> DRAM average : {avg_l1_dram * 1e3:9.4f} ms | {payload_bytes / avg_l1_dram / 1e9:9.3f} GB/s")
    print(f"L1  -> DRAM median  : {med_l1_dram * 1e3:9.4f} ms | {payload_bytes / med_l1_dram / 1e9:9.3f} GB/s")
    print(f"DRAM -> L1 average  : {avg_dram_l1 * 1e3:9.4f} ms | {payload_bytes / avg_dram_l1 / 1e9:9.3f} GB/s")
    print(f"DRAM -> L1 median   : {med_dram_l1 * 1e3:9.4f} ms | {payload_bytes / med_dram_l1 / 1e9:9.3f} GB/s")
    print("=" * 72)

    ttnn.deallocate(l1_source)
    ttnn.deallocate(dram_source)

    return {
        "payload_bytes": payload_bytes,
        "l1_to_dram_avg_s": avg_l1_dram,
        "dram_to_l1_avg_s": avg_dram_l1,
        "l1_to_dram_gbps": payload_bytes / avg_l1_dram / 1e9,
        "dram_to_l1_gbps": payload_bytes / avg_dram_l1 / 1e9,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--size-mb", type=float, default=8.0)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--l1-small-kb", type=int, default=16)
    args = parser.parse_args()

    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=args.l1_small_kb * 1024,
    )
    try:
        benchmark_l1_dram(
            device,
            size_mb=args.size_mb,
            channels=args.channels,
            repeat=args.repeat,
            warmup=args.warmup,
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
