"""Shared UTKFace dataset + test-split helper for the C3AE experiments.

This module owns the canonical implementation of ``UTKFaceDataset`` and the
70/15/15 train/val/test split used by ``models/train.py``, ``models/eval.py``,
and ``models/prep_input.py``. Centralizing the implementation prevents drift
between the three call sites — historically the same class was duplicated in
each script and had to be kept in sync by hand.

Reproducibility note: ``Path.glob("*.jpg*")`` is wrapped in ``sorted(...)`` so
the dataset's iteration order is purely lexicographic. ``manual_seed(42)``
alone is **not** sufficient for cross-script reproducibility unless the input
order is also stable; filesystem ``readdir`` order varies between filesystems
and after add/remove/rename operations.

This module intentionally has no ``orion_compiler`` dependency so it can be
imported from client-side / minimal environments (e.g. ``prep_input``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split

AGE_MAX = 100


class UTKFaceDataset(Dataset):
    """UTKFace image+label loader.

    Filename format: ``<age>_<gender>_<race>_<timestamp>.jpg``.

    Returns ``(image_tensor, target_tensor, age_int)`` per item, where:
        - ``image_tensor`` is a ``float32`` tensor of shape ``(3, img_size, img_size)``
          normalized to ``[-1, 1]``.
        - ``target_tensor`` is a ``float32`` tensor ``[is_adult]`` (0.0 or 1.0).
        - ``age_int`` is the parsed age clipped to ``[0, AGE_MAX]``.
    """

    def __init__(self, data_dir, img_size: int = 64, age_threshold: int = 18):
        self.img_size = img_size
        self.samples: list[tuple[Path, int, float]] = []

        # sorted() is load-bearing: filesystem-order is unstable across
        # mounts, after add/remove/rename, and across `git clean`. The split
        # downstream is seeded, but the seed only randomizes a *list* — the
        # list itself must be stable.
        for img_path in sorted(Path(data_dir).glob("*.jpg*")):
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, age, is_adult = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_arr = (img_arr - 0.5) / 0.5  # Normalize to [-1, 1]
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1)
        return img_t, torch.tensor([is_adult], dtype=torch.float32), age


def build_test_split(data_dir: Path, img_size: int = 64) -> Subset:
    """Reproduce the canonical 70/15/15 split with ``manual_seed(42)``.

    Returns the ``test`` subset (15% of samples) for downstream evaluation /
    sample preparation. The split sizes use ``int(0.70 * N)`` and
    ``int(0.15 * N)``, with the remainder going to the test subset, matching
    the original ``examples/c3ae-demo/train.py`` arithmetic exactly.
    """
    dataset = UTKFaceDataset(data_dir, img_size=img_size)
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    return test_set
