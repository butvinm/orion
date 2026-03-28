#!/usr/bin/env python3
"""Compile trained C3AE model to .orion format for the Go evaluator.

Usage:
    python generate_model.py [--weights weights.pth] [--output model.orion]
"""

import argparse
import os
import time
import tracemalloc

import torch
from model import C3AE
from orion_compiler import CKKSParams, Compiler

# CKKS parameters for C3AE.
# LogN=15 with single-CT packing (12288 values fit in 16384 slots).
# 15 computation levels — enough for the full network without bootstrap.
# LogQP=801 < 881 threshold for 128-bit security at logN=15 (HE Standard,
# uniform ternary secret).
PARAMS = CKKSParams(
    logn=15,
    logq=[51, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
    logp=[50, 50, 50],
    log_default_scale=40,
    ring_type="standard",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights.pth")
    parser.add_argument("--output", type=str, default="model.orion")
    parser.add_argument("--stride", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()

    net = C3AE(img_size=64, first_stride=args.stride)
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model loaded: {n_params:,} parameters (stride={args.stride})")

    # Compile
    tracemalloc.start()
    compiler = Compiler(net, PARAMS)

    print("Fitting...")
    t0 = time.time()
    torch.manual_seed(42)
    fit_input = torch.randn(1, 3, 64, 64)
    compiler.fit(fit_input)
    fit_time = time.time() - t0
    fit_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    print(f"Fit: {fit_time:.2f}s, peak memory: {fit_mem:.1f} MB")

    print("Compiling...")
    t0 = time.time()
    compiler.compile_to_file(args.output)
    compile_time = time.time() - t0
    compile_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()
    print(f"Compile: {compile_time:.2f}s, peak memory: {compile_mem:.1f} MB")

    file_size = os.path.getsize(args.output)
    print(f"\nModel written to {args.output} ({file_size:,} bytes)")


if __name__ == "__main__":
    main()
