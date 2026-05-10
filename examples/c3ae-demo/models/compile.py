#!/usr/bin/env python3
"""Compile a trained C3AE FHE model to ``.orion`` for the Go evaluator.

Adapted from ``examples/c3ae-demo/generate_model.py``. The CKKS configuration
is selected by name from :mod:`models.params` so the same script can produce
both ``logn15`` and ``logn16`` artifacts.

Usage (from ``examples/c3ae-demo/``)::

    python -m models.compile --config logn15 \
        --weights out/weights_fhe.pth \
        --output out/logn15/model.orion

The script measures wall-clock time and two memory figures across ``fit`` +
``compile_to_file``, then writes a sibling ``compile.json`` next to the
produced ``model.orion``::

    {
        "compile_s": <float>,
        "compile_peak_python_mb": <float>,   # tracemalloc — Python heap only
        "compile_peak_rss_mb": <float>,      # getrusage(RUSAGE_SELF).ru_maxrss
        "model_bytes": <int>,
    }

**Why two memory fields?** ``tracemalloc`` only sees Python-side allocations.
The dominant compile-time memory is held by Lattigo via CGO (Go runtime + C
allocations), which ``tracemalloc`` is *blind to*. ``getrusage`` returns the
true process peak RSS (in kB on Linux) including all CGO/Go memory, so it is
the right number to report alongside the existing demo's "compilation memory"
figure. The Python-tracked number is kept as a secondary diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
import tracemalloc

import torch
from orion_compiler import Compiler

from models.c3ae_fhe import C3AE
from models.params import PARAMS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile the C3AE FHE (Quad) variant to a .orion file.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="fhe",
        choices=["fhe"],
        help=(
            "Model variant. Only 'fhe' (Quad) compiles under CKKS — "
            "'relu' is not FHE-compatible. Kept as an explicit flag for "
            "future-proofing but defaults to 'fhe'."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=sorted(PARAMS.keys()),
        help="CKKS configuration name from models.params.PARAMS.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="out/weights_fhe.pth",
        help="Path to the trained Quad weights (.pth).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .orion path (default: out/<config>/model.orion).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        choices=[1, 2],
        help="First-conv stride forwarded to the model constructor.",
    )
    args = parser.parse_args()

    if args.variant != "fhe":
        # Defensive — argparse already enforces this, but keep an explicit
        # message in case the choices ever expand.
        raise SystemExit(
            f"--variant {args.variant!r} is not compileable; only 'fhe' is supported."
        )

    output = args.output or os.path.join("out", args.config, "model.orion")
    output_dir = os.path.dirname(output) or "."
    os.makedirs(output_dir, exist_ok=True)

    ckks_params = PARAMS[args.config]

    net = C3AE(img_size=64, first_stride=args.stride)
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model loaded: {n_params:,} parameters (stride={args.stride}, config={args.config})")

    compiler = Compiler(net, ckks_params)

    # Establish a baseline RSS so our reported "peak" excludes the cost of
    # importing torch / orion_compiler / loading weights — we only attribute
    # memory growth caused by fit+compile to this run.
    rss_baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    tracemalloc.start()
    t0 = time.time()

    print("Fitting...")
    torch.manual_seed(42)
    fit_input = torch.randn(1, 3, 64, 64)
    compiler.fit(fit_input)

    print(f"Compiling to {output} ...")
    compiler.compile_to_file(output)

    compile_s = time.time() - t0
    peak_python_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    compile_peak_python_mb = peak_python_bytes / (1024 * 1024)

    # ru_maxrss is monotonic over the process lifetime, so we report the
    # absolute peak (which dominates the small import-time delta — the
    # Lattigo allocations during compile_to_file are several GB).
    rss_peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux ru_maxrss is in kB; convert to MB.
    compile_peak_rss_mb = rss_peak_kb / 1024.0
    rss_baseline_mb = rss_baseline_kb / 1024.0

    model_bytes = os.path.getsize(output)

    metrics = {
        "compile_s": compile_s,
        # CGO/Go memory is invisible to tracemalloc — the dominant compile
        # memory at logn=15 is several GB on the Lattigo side. Keep both
        # fields so the operator can see the gap and so build_results.py
        # can prefer the true RSS while still showing the Python figure.
        "compile_peak_python_mb": compile_peak_python_mb,
        "compile_peak_rss_mb": compile_peak_rss_mb,
        "compile_baseline_rss_mb": rss_baseline_mb,
        "model_bytes": model_bytes,
    }
    metrics_path = os.path.join(output_dir, "compile.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    print(
        f"Compile: {compile_s:.2f}s, "
        f"peak RSS: {compile_peak_rss_mb:.1f} MB (baseline {rss_baseline_mb:.1f} MB), "
        f"peak python-tracked: {compile_peak_python_mb:.1f} MB"
    )
    print(f"Model written to {output} ({model_bytes:,} bytes)")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
