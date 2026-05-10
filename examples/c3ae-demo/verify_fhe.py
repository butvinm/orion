#!/usr/bin/env python3
"""Compare FHE-decrypted probabilities against cleartext PyTorch inference.

Reads:
  results/<cfg>/decrypt_<idx>.json    written by `bench decrypt`
  out/inputs/sample_<idx>.bin         the same float64 blob fed into encrypt
  out/weights_fhe.pth                 the trained Quad weights

For each sample with a matching ``decrypt_<idx>.json`` file, the script:
  1. Loads the same input blob, reshapes to (1, 3, 64, 64),
  2. Runs the cleartext FHE-variant network in eval mode,
  3. Computes the cleartext sigmoid probability,
  4. Compares it to the FHE-decrypted ``prob`` field,
  5. Writes one row per sample to ``results/<cfg>/cleartext_vs_fhe.csv``,
  6. Exits with code ``2`` if any sample's ``abs_diff >= --tol`` (default 0.05).

This is intentionally **separate** from ``run_fhe.sh`` — running the script
is a soft post-check the operator can invoke once results exist, rather than
an in-line gate that aborts a multi-hour FHE run on a marginal numerical
difference. The Post-Completion runbook documents this script as the place
to verify the "decrypt_mae < 0.05" criterion.

Usage (from ``examples/c3ae-demo/experiments/``):

    python verify_fhe.py --config logn15
    python verify_fhe.py --config logn16 --tol 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from models.c3ae_fhe import C3AE


def load_decrypt_json(path: Path) -> float | None:
    """Return ``prob`` from a ``decrypt_<idx>.json``, or None if unreadable."""
    if not path.is_file():
        return None
    try:
        with path.open("r") as f:
            return float(json.load(f).get("prob"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def cleartext_prob(net: torch.nn.Module, sample_bin: Path) -> float:
    """Run ``net`` on the float64 blob at ``sample_bin`` and return ``sigmoid(logit)``."""
    arr = np.fromfile(sample_bin, dtype=np.float64)
    if arr.size != 12288:
        raise ValueError(f"{sample_bin} has {arr.size} values; expected 12288")
    x = torch.from_numpy(arr.astype(np.float32)).reshape(1, 3, 64, 64)
    with torch.no_grad():
        logit = net(x).item()
    return 1.0 / (1.0 + float(np.exp(-logit)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="FHE config name (e.g. logn15, logn16).")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Experiments root directory.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to weights_fhe.pth (default: <root>/out/weights_fhe.pth).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.05,
        help="Per-sample max allowable |fhe_prob - cleartext_prob|. Exit 2 if any exceeds.",
    )
    args = parser.parse_args()

    weights = args.weights or args.root / "out" / "weights_fhe.pth"
    if not weights.is_file():
        print(f"[verify_fhe] ERROR: weights not found at {weights}", file=sys.stderr)
        sys.exit(1)

    inputs_dir = args.root / "out" / "inputs"
    results_dir = args.root / "results" / args.config

    if not results_dir.is_dir():
        print(f"[verify_fhe] ERROR: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Load the cleartext FHE-variant network with the same weights used to
    # compile the .orion model. Eval mode + no-grad keeps the comparison
    # apples-to-apples with what the FHE pipeline computes.
    net = C3AE(img_size=64, first_stride=2)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()

    rows: list[dict[str, str]] = []
    max_diff = 0.0
    fail = False

    # Discover the samples that actually have a decrypt_<idx>.json.
    decrypt_files = sorted(results_dir.glob("decrypt_*.json"))
    if not decrypt_files:
        print(
            f"[verify_fhe] WARN: no decrypt_*.json files in {results_dir}; "
            f"run `bench decrypt` first.",
            file=sys.stderr,
        )

    for dpath in decrypt_files:
        try:
            idx = int(dpath.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        sample_bin = inputs_dir / f"sample_{idx}.bin"
        fhe_prob = load_decrypt_json(dpath)
        if fhe_prob is None:
            print(f"[verify_fhe] sample {idx}: unreadable {dpath}", file=sys.stderr)
            continue
        if not sample_bin.is_file():
            print(f"[verify_fhe] sample {idx}: missing {sample_bin}", file=sys.stderr)
            continue
        clear_p = cleartext_prob(net, sample_bin)
        diff = abs(fhe_prob - clear_p)
        max_diff = max(max_diff, diff)
        passed = diff < args.tol
        if not passed:
            fail = True
        rows.append(
            {
                "sample_idx": str(idx),
                "fhe_prob": f"{fhe_prob:.6f}",
                "cleartext_prob": f"{clear_p:.6f}",
                "abs_diff": f"{diff:.6f}",
                "passed": "true" if passed else "false",
            }
        )
        print(
            f"  sample {idx:>4d}: fhe={fhe_prob:.4f} clear={clear_p:.4f} "
            f"diff={diff:.4f} {'OK' if passed else 'FAIL'}"
        )

    out_csv = results_dir / "cleartext_vs_fhe.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_idx", "fhe_prob", "cleartext_prob", "abs_diff", "passed"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[verify_fhe] wrote {out_csv} ({len(rows)} row(s)), max_diff={max_diff:.4f}")

    if fail:
        print(
            f"[verify_fhe] FAIL: at least one sample exceeded tolerance {args.tol}",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
