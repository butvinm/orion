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

from orion_compiler import Compiler, CKKSParams
from model import C3AE

# CKKS parameters for C3AE.
# LogN=14 with multi-CT packing (12288 values split across 2 CTs of 8192 slots).
# Bootstrap for the deep network (6 Quad activations).
PARAMS = CKKSParams(
    logn=14,
    logq=[55, 40, 40, 40, 40, 40, 40, 40, 40, 40],
    logp=[61, 61, 61],
    log_default_scale=40,
    h=192,
    ring_type="standard",
    boot_logp=[61, 61, 61, 61, 61, 61, 61, 61],
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
