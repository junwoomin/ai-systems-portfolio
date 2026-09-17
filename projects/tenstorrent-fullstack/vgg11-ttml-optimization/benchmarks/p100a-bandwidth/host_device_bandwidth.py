#!/usr/bin/env python3
"""
CPU host memory <-> P100a device DRAM effective bandwidth benchmark.

Measures:
  CPU DDR4/RAM -> P100a device DRAM  (Host to Device / H2D)
  P100a device DRAM -> CPU DDR4/RAM  (Device to Host / D2H)

This path normally crosses PCIe, so this is not the same as L1 <-> device DRAM.

Example:
    python host_device_bandwidth.py --size-mb 256 --repeat 20
"""

import argparse
import statistics
import time

import torch
import ttnn


def synchronize(device):
    ttnn.synchronize_device(device)


def benchmark_host_device(device, size_mb=256, repeat=10, warmup=2):
    num_elements = int(size_mb * 1024 * 1024) // 2  # BF16 = 2 bytes

    cpu = torch.randn(num_elements, dtype=torch.bfloat16)
    payload_bytes = cpu.numel() * cpu.element_size()

    host_tt = ttnn.from_torch(
        cpu,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    print("=" * 72)
    print("CPU HOST RAM <-> P100a DEVICE DRAM")
    print("=" * 72)
    print(f"Payload             : {payload_bytes / 1024**2:.3f} MiB")
    print()

    # Warm-up H2D / D2H
    for _ in range(warmup):
        dev = ttnn.to_device(
            host_tt,
            device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        synchronize(device)
        host_back = ttnn.from_device(dev)
        synchronize(device)
        del host_back
        ttnn.deallocate(dev)

    h2d_times = []
    for _ in range(repeat):
        synchronize(device)
        t0 = time.perf_counter()
        dev = ttnn.to_device(
            host_tt,
            device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        synchronize(device)
        dt = time.perf_counter() - t0
        h2d_times.append(dt)
        ttnn.deallocate(dev)

    dev_source = ttnn.to_device(
        host_tt,
        device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    synchronize(device)

    d2h_times = []
    for _ in range(repeat):
        synchronize(device)
        t0 = time.perf_counter()
        host_back = ttnn.from_device(dev_source)
        synchronize(device)
        dt = time.perf_counter() - t0
        d2h_times.append(dt)
        del host_back

    ttnn.deallocate(dev_source)

    avg_h2d = statistics.mean(h2d_times)
    med_h2d = statistics.median(h2d_times)
    avg_d2h = statistics.mean(d2h_times)
    med_d2h = statistics.median(d2h_times)

    print(f"Host -> DRAM average : {avg_h2d * 1e3:9.4f} ms | {payload_bytes / avg_h2d / 1e9:9.3f} GB/s")
    print(f"Host -> DRAM median  : {med_h2d * 1e3:9.4f} ms | {payload_bytes / med_h2d / 1e9:9.3f} GB/s")
    print(f"DRAM -> Host average : {avg_d2h * 1e3:9.4f} ms | {payload_bytes / avg_d2h / 1e9:9.3f} GB/s")
    print(f"DRAM -> Host median  : {med_d2h * 1e3:9.4f} ms | {payload_bytes / med_d2h / 1e9:9.3f} GB/s")
    print("=" * 72)

    return {
        "payload_bytes": payload_bytes,
        "h2d_avg_s": avg_h2d,
        "d2h_avg_s": avg_d2h,
        "h2d_gbps": payload_bytes / avg_h2d / 1e9,
        "d2h_gbps": payload_bytes / avg_d2h / 1e9,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--size-mb", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--l1-small-kb", type=int, default=16)
    args = parser.parse_args()

    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=args.l1_small_kb * 1024,
    )
    try:
        benchmark_host_device(
            device,
            size_mb=args.size_mb,
            repeat=args.repeat,
            warmup=args.warmup,
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
