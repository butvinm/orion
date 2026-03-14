"""Unified training script for all Orion examples.

Usage:
    python examples/models/train.py mlp
    python examples/models/train.py alexnet --epochs 200
"""

import argparse
import importlib
import os
import sys

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
AVAILABLE = ["mlp", "lenet", "lola", "alexnet", "vgg", "resnet"]

TRAIN_DEFAULTS = {
    "mnist": {"func": "train_on_mnist", "epochs": 10},
    "cifar": {"func": "train_on_cifar", "epochs": 200},
}


def main():
    parser = argparse.ArgumentParser(description="Train Orion example models")
    parser.add_argument("example", choices=AVAILABLE, help="Example model to train")
    parser.add_argument("--epochs", type=int, default=None, help="Override default epochs")
    args = parser.parse_args()

    sys.path.insert(0, EXAMPLES_DIR)
    mod = importlib.import_module(args.example)
    sys.path.pop(0)

    net = mod.Model()
    dataset = mod.CONFIG["dataset"]
    defaults = TRAIN_DEFAULTS[dataset]

    epochs = args.epochs or defaults["epochs"]
    save_path = os.path.join(EXAMPLES_DIR, f"{args.example}_weights.pt")

    from orion_compiler.core.utils import train_on_mnist, train_on_cifar

    train_func = train_on_mnist if dataset == "mnist" else train_on_cifar
    train_func(net, data_dir=os.path.join(EXAMPLES_DIR, "data"), epochs=epochs, save_path=save_path)


if __name__ == "__main__":
    main()
