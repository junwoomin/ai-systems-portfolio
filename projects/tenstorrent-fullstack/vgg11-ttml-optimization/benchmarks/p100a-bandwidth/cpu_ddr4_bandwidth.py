#!/usr/bin/env python3
"""
CPU DDR4 / system RAM memcpy bandwidth benchmark using PyTorch.

Reports two interpretations:
  payload GB/s      = copied bytes / time
  read+write GB/s   = 2 * copied bytes / time

The second number better reflects total DRAM traffic for memcpy because the
operation reads the source and writes the destination.

Example:
    python cpu_ddr4_bandwidth.py --size-mb 512 --repeat 20
"""

import argparse
import statistics
import time

import torch


def read_meminfo_total_gib():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    return kib / 1024 / 1024
    except Exception:
        pass
    return None


def benchmark_cpu_ddr4(size_mb=512, repeat=20, warmup=3):
    size_bytes = int(size_mb * 1024 * 1024)

    src = torch.empty(size_bytes, dtype=torch.uint8)
    dst = torch.empty_like(src)

    # Touch pages so first-use page faults are not part of the timed loop.
    src.fill_(1)
    dst.fill_(0)

    print("=" * 72)
    print("CPU DDR4 / SYSTEM RAM MEMCPY")
    print("=" * 72)

    total_gib = read_meminfo_total_gib()
    if total_gib is not None:
        print(f"System RAM total    : {total_gib:.2f} GiB")
    print(f"Payload             : {size_bytes / 1024**2:.3f} MiB")
    print(f"PyTorch CPU threads : {torch.get_num_threads()}")
    print()

    for _ in range(warmup):
        dst.copy_(src)

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        dst.copy_(src)
        dt = time.perf_counter() - t0
        times.append(dt)

    avg = statistics.mean(times)
    med = statistics.median(times)

    avg_payload_gbps = size_bytes / avg / 1e9
    med_payload_gbps = size_bytes / med / 1e9

    print(f"Average copy time   : {avg * 1e3:9.4f} ms")
    print(f"Median copy time    : {med * 1e3:9.4f} ms")
    print(f"Payload bandwidth   : {avg_payload_gbps:9.3f} GB/s average")
    print(f"Payload bandwidth   : {med_payload_gbps:9.3f} GB/s median")
    print(f"Read+write traffic  : {avg_payload_gbps * 2:9.3f} GB/s average")
    print(f"Read+write traffic  : {med_payload_gbps * 2:9.3f} GB/s median")
    print("=" * 72)

    return {
        "payload_bytes": size_bytes,
        "avg_s": avg,
        "payload_gbps": avg_payload_gbps,
        "read_write_gbps": avg_payload_gbps * 2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    benchmark_cpu_ddr4(
        size_mb=args.size_mb,
        repeat=args.repeat,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
