#!/usr/bin/env python3
"""Train C3AE for binary age verification (18+).

Uses asymmetric BCE loss to penalize false positives (minors classified as adult).

Usage:
    python train.py --data-dir ./data/UTKFace --epochs 60
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from model import C3AE

AGE_MAX = 100


class UTKFaceDataset(Dataset):
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


def asymmetric_loss(pred, target, fpr_weight):
    """Asymmetric BCE: penalizes false positives more heavily."""
    pred = pred.clamp(1e-7, 1 - 1e-7)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    weights = torch.where(target < 0.5, fpr_weight, 1.0)
    return (bce * weights).mean()


def evaluate(model, loader, device, fpr_weight):
    model.eval()
    all_probs, all_targets = [], []
    total_loss, n = 0, 0

    with torch.no_grad():
        for images, targets, _ in loader:
            images, targets = images.to(device), targets.to(device)
            probs = torch.sigmoid(model(images))
            loss = asymmetric_loss(probs, targets, fpr_weight)
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
            all_probs.extend(probs.cpu().squeeze().tolist())
            all_targets.extend(targets.cpu().squeeze().tolist())

    probs = np.array(all_probs)
    targets = np.array(all_targets)
    pred_adult = probs >= 0.5
    true_adult = targets >= 0.5
    minors_mask = ~true_adult

    fpr = pred_adult[minors_mask].mean() if minors_mask.sum() > 0 else 0
    fnr = (~pred_adult[true_adult]).mean() if true_adult.sum() > 0 else 0
    accuracy = (pred_adult == true_adult).mean()

    return {"loss": total_loss / n, "accuracy": accuracy, "fpr": fpr, "fnr": fnr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=2, choices=[1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--fpr-weight", type=float, default=40.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--data-dir", type=Path, default=Path("./data/UTKFace"))
    parser.add_argument("--output", type=Path, default=Path("./weights.pth"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training C3AE (stride={args.stride}) on {device}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, FPR weight: {args.fpr_weight}")

    # Data
    dataset = UTKFaceDataset(args.data_dir, img_size=64)
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Model
    model = C3AE(img_size=64, first_stride=args.stride).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_acc, best_state = -1.0, None
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss, n_train = 0, 0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            probs = torch.sigmoid(model(images))
            loss = asymmetric_loss(probs, targets, args.fpr_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            n_train += images.size(0)
        scheduler.step()

        val = evaluate(model, val_loader, device, args.fpr_weight)
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{args.epochs} | "
                f"Loss: {train_loss / n_train:.4f} | "
                f"FPR: {val['fpr'] * 100:.1f}% | FNR: {val['fnr'] * 100:.1f}% | "
                f"Acc: {val['accuracy'] * 100:.1f}%"
            )

    total_time = time.time() - t0
    print(f"\nTraining done in {total_time:.1f}s ({total_time / args.epochs:.1f}s/epoch)")

    # Save
    torch.save(best_state, args.output)
    print(f"Saved best model to {args.output}")

    # Test
    model.load_state_dict(best_state)
    model = model.to(device)
    test = evaluate(model, test_loader, device, args.fpr_weight)
    print(f"\nTest: FPR={test['fpr'] * 100:.1f}%, FNR={test['fnr'] * 100:.1f}%, "
          f"Acc={test['accuracy'] * 100:.1f}%")


if __name__ == "__main__":
    main()
