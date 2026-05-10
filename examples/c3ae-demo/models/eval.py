#!/usr/bin/env python3
"""Cleartext FPR/FNR/Accuracy evaluation for both C3AE variants.

Loads ``out/weights_relu.pth`` (true-ReLU) and ``out/weights_fhe.pth`` (Quad)
and evaluates each variant on the UTKFace test split (reproduced via the same
``manual_seed(42)`` as ``models/train.py`` and ``models/prep_input.py``).

Two scopes are reported per variant:

- ``overall``  — full test set
- ``boundary`` — samples with ``16 <= age <= 20`` only (the hard band where
  the binary 18+ classifier can plausibly disagree with itself)

Decision rule: ``sigmoid(logit) >= 0.5`` -> adult.

Output: ``results/cleartext.csv`` with header
``variant,scope,n,fpr,fnr,accuracy`` and one row per (variant x scope)
combination that was actually evaluated. Missing weight files are SKIPPED
with a printed warning (not an error) so partial runs still produce a CSV.

Usage (from ``examples/c3ae-demo/``):

    python -m models.eval --data-dir ./data/UTKFace --output results/cleartext.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.c3ae import C3AE as C3AE_ReLU
from models.c3ae_fhe import C3AE as C3AE_Quad
from models.metrics import compute_metrics
from models.utkface import build_test_split


def gather_predictions(
    model: torch.nn.Module,
    test_set: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model in eval mode over the test set, returning (probs, targets, ages)."""
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_probs: list[float] = []
    all_targets: list[float] = []
    all_ages: list[int] = []

    with torch.no_grad():
        for images, targets, ages in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            tgts = targets.cpu().numpy().reshape(-1)
            all_probs.extend(probs.tolist())
            all_targets.extend(tgts.tolist())
            # ``ages`` from the dataset is a Python int; the DataLoader collates
            # to a tensor of ints. Convert defensively to a list of ints.
            if isinstance(ages, torch.Tensor):
                all_ages.extend(int(a) for a in ages.tolist())
            else:
                all_ages.extend(int(a) for a in ages)

    return np.array(all_probs), np.array(all_targets), np.array(all_ages)


VARIANTS: dict[str, type] = {
    "relu": C3AE_ReLU,
    "fhe": C3AE_Quad,
}


def evaluate_variant(
    variant: str,
    weights_path: Path,
    test_set: torch.utils.data.Dataset,
    device: torch.device,
) -> list[tuple[str, str, dict[str, float | int]]]:
    """Evaluate a single variant. Returns list of (variant, scope, metrics).

    If the weights file is missing, prints a warning and returns ``[]`` so the
    caller can keep going with the other variants.
    """
    if not weights_path.exists():
        print(f"[skip] weights not found for variant={variant!r}: {weights_path}")
        return []

    cls = VARIANTS[variant]
    model = cls(img_size=64, first_stride=2).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    probs, targets, ages = gather_predictions(model, test_set, device)

    overall = compute_metrics(probs, targets)
    boundary_mask = (ages >= 16) & (ages <= 20)
    boundary = compute_metrics(probs[boundary_mask], targets[boundary_mask])

    return [
        (variant, "overall", overall),
        (variant, "boundary", boundary),
    ]


def write_csv(rows: list[tuple[str, str, dict[str, float | int]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "scope", "n", "fpr", "fnr", "accuracy"])
        for variant, scope, m in rows:
            writer.writerow(
                [
                    variant,
                    scope,
                    int(m["n"]),
                    f"{m['fpr']:.6f}",
                    f"{m['fnr']:.6f}",
                    f"{m['accuracy']:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/UTKFace"),
        help="UTKFace image directory (jpg files).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cleartext.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("out"),
        help="Directory containing weights_<variant>.pth files.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = build_test_split(args.data_dir)
    print(f"Test set: {len(test_set)} samples on device={device}")

    rows: list[tuple[str, str, dict[str, float | int]]] = []
    for variant in ("relu", "fhe"):
        weights = args.weights_dir / f"weights_{variant}.pth"
        rows.extend(evaluate_variant(variant, weights, test_set, device))

    if not rows:
        print(
            "[warn] no variants evaluated — both weights files are missing. "
            f"Looked under {args.weights_dir}/. Writing empty CSV anyway."
        )

    write_csv(rows, args.output)
    print(f"Wrote {args.output} ({len(rows)} row(s))")
    for variant, scope, m in rows:
        print(
            f"  {variant:>4s} {scope:>8s}: n={m['n']:>5d}  "
            f"fpr={m['fpr']:.4f}  fnr={m['fnr']:.4f}  acc={m['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
