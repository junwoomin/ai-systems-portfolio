#!/usr/bin/env python3
"""
Run all bandwidth tests:
  1) CPU DDR4 memcpy
  2) CPU host RAM -> P100a device DRAM
  3) P100a device DRAM -> CPU host RAM
  4) P100a L1 SRAM -> device DRAM
  5) P100a device DRAM -> L1 SRAM

Example:
    python all_bandwidth.py
"""

import argparse

import ttnn

from cpu_ddr4_bandwidth import benchmark_cpu_ddr4
from host_device_bandwidth import benchmark_host_device
from l1_dram_bandwidth import benchmark_l1_dram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--l1-small-kb", type=int, default=16)

    parser.add_argument("--cpu-size-mb", type=int, default=512)
    parser.add_argument("--pcie-size-mb", type=int, default=256)
    parser.add_argument("--l1-size-mb", type=float, default=8.0)

    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    print()
    cpu_result = benchmark_cpu_ddr4(
        size_mb=args.cpu_size_mb,
        repeat=args.repeat,
        warmup=args.warmup,
    )

    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=args.l1_small_kb * 1024,
    )

    try:
        print()
        host_result = benchmark_host_device(
            device,
            size_mb=args.pcie_size_mb,
            repeat=args.repeat,
            warmup=args.warmup,
        )

        print()
        l1_result = benchmark_l1_dram(
            device,
            size_mb=args.l1_size_mb,
            repeat=args.repeat,
            warmup=args.warmup,
        )
    finally:
        ttnn.close_device(device)

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"CPU DDR4 memcpy payload     : {cpu_result['payload_gbps']:.3f} GB/s")
    print(f"CPU DDR4 memcpy R+W traffic : {cpu_result['read_write_gbps']:.3f} GB/s")
    print(f"Host -> P100a DRAM          : {host_result['h2d_gbps']:.3f} GB/s")
    print(f"P100a DRAM -> Host          : {host_result['d2h_gbps']:.3f} GB/s")
    print(f"P100a L1 -> DRAM            : {l1_result['l1_to_dram_gbps']:.3f} GB/s")
    print(f"P100a DRAM -> L1            : {l1_result['dram_to_l1_gbps']:.3f} GB/s")
    print("=" * 72)


if __name__ == "__main__":
    main()
