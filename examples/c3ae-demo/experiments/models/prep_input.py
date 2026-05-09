#!/usr/bin/env python3
"""Prepare a UTKFace test sample as a raw float64 blob for the FHE bench.

Reproduces the 70/15/15 train/val/test split with the same ``manual_seed(42)``
as ``models/train.py`` (and the original ``examples/c3ae-demo/train.py``) so
the test indices match exactly across runs and across cleartext eval and the
FHE bench.

Usage (from ``examples/c3ae-demo/experiments/``):

    # Single sample by test-set index
    python -m models.prep_input --idx 0

    # First 3 boundary-band samples (16 <= age <= 20) in test-iteration order
    python -m models.prep_input --boundary-band

Outputs:
    out/inputs/sample_<idx>.bin   raw little-endian float64, 12288 values
                                  (3 * 64 * 64), normalized to [-1, 1]
    out/inputs/ground_truth.csv   header: idx,age,is_adult
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from models.utkface import build_test_split

BOUNDARY_BAND_TARGET = 3


def select_indices(test_set: Subset, args: argparse.Namespace) -> list[int]:
    """Return the list of test-set positions to dump.

    For ``--idx N`` returns ``[N]``. For ``--boundary-band`` returns the first
    ``BOUNDARY_BAND_TARGET`` positions whose age is in ``[16, 20]`` in
    iteration order.

    Behavior on insufficient samples in ``--boundary-band``:
      - 0 samples found  → raise (the original strict failure mode)
      - 1 or 2 samples   → emit a clear ``[warn]`` to stderr and return the
        partial list. Downstream tooling expects exactly ``BOUNDARY_BAND_TARGET``
        samples, so a partial result is operator-visible — but we don't refuse
        outright because tiny synthetic datasets (e.g. CI fixtures) may
        legitimately have fewer than 3 samples in the band.
    """
    if args.idx is not None:
        if not (0 <= args.idx < len(test_set)):
            raise ValueError(f"--idx {args.idx} out of range for test set of size {len(test_set)}")
        return [args.idx]

    # boundary band: first BOUNDARY_BAND_TARGET samples with 16 <= age <= 20
    picked: list[int] = []
    for i in range(len(test_set)):
        _, _, age = test_set[i]
        if 16 <= int(age) <= 20:
            picked.append(i)
            if len(picked) >= BOUNDARY_BAND_TARGET:
                break
    if not picked:
        raise RuntimeError(
            "No samples with 16 <= age <= 20 found in test set; cannot satisfy --boundary-band."
        )
    if len(picked) < BOUNDARY_BAND_TARGET:
        print(
            f"[warn] --boundary-band requested {BOUNDARY_BAND_TARGET} samples but only "
            f"{len(picked)} found in [16, 20] (test set size {len(test_set)}); "
            f"writing the partial result. Downstream may expect "
            f"exactly {BOUNDARY_BAND_TARGET} samples — verify before running the FHE pipeline.",
            file=sys.stderr,
        )
    return picked


def dump_sample(test_set, idx: int, out_dir: Path) -> tuple[int, int]:
    """Dump test-set position ``idx`` as a raw float64 blob.

    Returns ``(age, is_adult)`` for ground-truth CSV bookkeeping.

    The image tensor is shape ``(3, 64, 64)`` already normalized to ``[-1, 1]``.
    We flatten to ``12288`` values and write little-endian ``float64`` (8 bytes
    each = 98304 bytes total).

    float64 is chosen (over float32) because Lattigo's CKKS encoder accepts
    ``[]float64`` natively. Writing float64 here avoids an extra cast in the
    Go bench when reading the file.
    """
    img, target, age = test_set[idx]
    arr = img.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if arr.shape != (12288,):
        raise RuntimeError(f"unexpected sample shape {arr.shape}, want (12288,)")

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"sample_{idx}.bin"
    # ``tobytes`` on a contiguous little-endian float64 array writes exactly
    # 8 * 12288 = 98304 bytes. ``numpy`` is little-endian on x86_64 Linux.
    arr.astype("<f8").tofile(bin_path)

    is_adult = int(float(target.item()) >= 0.5)
    return int(age), is_adult


def write_ground_truth(rows: list[tuple[int, int, int]], out_dir: Path) -> Path:
    """Write/merge ``ground_truth.csv``.

    Idempotency policy: read any existing rows, merge with the rows produced
    in this invocation (keyed by ``idx``, current invocation wins on collision),
    write the deduplicated, idx-sorted result back. This way re-running with
    the same flags is a no-op, and re-running with new indices accumulates.
    """
    csv_path = out_dir / "ground_truth.csv"
    merged: dict[int, tuple[int, int]] = {}

    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    merged[int(r["idx"])] = (int(r["age"]), int(r["is_adult"]))
                except (KeyError, ValueError):
                    continue

    for idx, age, is_adult in rows:
        merged[idx] = (age, is_adult)

    out_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "age", "is_adult"])
        for idx in sorted(merged):
            age, is_adult = merged[idx]
            writer.writerow([idx, age, is_adult])
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--idx",
        type=int,
        default=None,
        help="Dump a single test-set sample at this position.",
    )
    mode.add_argument(
        "--boundary-band",
        action="store_true",
        help="Dump the first 3 test samples with 16 <= age <= 20.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/UTKFace"),
        help="UTKFace image directory (jpg files).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/inputs"),
        help="Output directory for sample_*.bin and ground_truth.csv.",
    )
    args = parser.parse_args()

    test_set = build_test_split(args.data_dir)
    indices = select_indices(test_set, args)

    rows: list[tuple[int, int, int]] = []
    for idx in indices:
        age, is_adult = dump_sample(test_set, idx, args.out_dir)
        rows.append((idx, age, is_adult))
        print(f"  wrote {args.out_dir / f'sample_{idx}.bin'}  age={age} is_adult={is_adult}")

    csv_path = write_ground_truth(rows, args.out_dir)
    print(f"  wrote {csv_path}  ({len(rows)} new/updated row(s))")


if __name__ == "__main__":
    main()
