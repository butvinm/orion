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

Usage (from ``examples/c3ae-demo/experiments/``):

    python -m models.eval --data-dir ./data/UTKFace --output results/cleartext.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from models.c3ae import C3AE as C3AE_ReLU
from models.c3ae_fhe import C3AE as C3AE_Quad

AGE_MAX = 100


# NOTE: UTKFaceDataset is re-defined here (rather than imported from
# ``models.train`` or ``models.prep_input``) so this evaluator is independent
# of those modules' imports. It mirrors the dataset used during training and
# preprocessing exactly so the test split is identical.
class UTKFaceDataset(Dataset):
    """UTKFace image+label loader. Mirrors ``models.train.UTKFaceDataset`` exactly.

    File-name format: ``<age>_<gender>_<race>_<timestamp>.jpg``.
    """

    def __init__(self, data_dir, img_size=64, age_threshold=18):
        self.img_size = img_size
        self.samples = []

        for img_path in Path(data_dir).glob("*.jpg*"):
            try:
                age = min(max(int(img_path.name.split("_")[0]), 0), AGE_MAX)
                is_adult = 1.0 if age >= age_threshold else 0.0
                self.samples.append((img_path, age, is_adult))
            except (ValueError, IndexError):
                continue

        if not self.samples:
            raise ValueError(f"No samples found in {data_dir}")

        ages = [s[1] for s in self.samples]
        minors = sum(1 for s in self.samples if s[2] == 0.0)
        adults = len(self.samples) - minors
        print(
            f"[Dataset] {len(self.samples)} samples: "
            f"{minors} minors ({minors / len(self.samples) * 100:.0f}%), "
            f"{adults} adults, ages {min(ages)}-{max(ages)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age, is_adult = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, torch.tensor([is_adult], dtype=torch.float32), age


def build_test_split(data_dir: Path):
    """Reproduce the 70/15/15 split with ``manual_seed(42)`` and return the test Subset."""
    dataset = UTKFaceDataset(data_dir, img_size=64)
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    return test_set


def compute_metrics(probs: np.ndarray, targets: np.ndarray) -> dict:
    """Compute FPR / FNR / accuracy from sigmoid probabilities and binary targets.

    Decision rule: ``probs >= 0.5`` -> predicted adult.

    Empty-input convention: when ``targets`` has length 0 (e.g. an empty
    boundary band), returns zeros for all metrics. We pick zero rather than
    NaN so the CSV downstream is plain numeric and easy to consume; the
    accompanying ``n`` field disambiguates "0 because perfect" vs "0 because
    empty". The same convention is applied per-class: a scope with no minors
    reports ``fpr=0.0``, a scope with no adults reports ``fnr=0.0``.

    Args:
        probs:   1-D array of sigmoid outputs in ``[0, 1]``.
        targets: 1-D array of ground-truth labels in ``{0.0, 1.0}``.

    Returns:
        ``{"n": int, "fpr": float, "fnr": float, "accuracy": float}``.
    """
    probs = np.asarray(probs).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    n = int(targets.shape[0])

    if n == 0:
        return {"n": 0, "fpr": 0.0, "fnr": 0.0, "accuracy": 0.0}

    pred_adult = probs >= 0.5
    true_adult = targets >= 0.5
    minors_mask = ~true_adult

    fpr = float(pred_adult[minors_mask].mean()) if minors_mask.sum() > 0 else 0.0
    fnr = float((~pred_adult[true_adult]).mean()) if true_adult.sum() > 0 else 0.0
    accuracy = float((pred_adult == true_adult).mean())

    return {"n": n, "fpr": fpr, "fnr": fnr, "accuracy": accuracy}


def gather_predictions(model, test_set, device, batch_size=64):
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
    test_set,
    device,
) -> list[tuple[str, str, dict]]:
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


def write_csv(rows: list[tuple[str, str, dict]], out_path: Path) -> None:
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

    rows: list[tuple[str, str, dict]] = []
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
