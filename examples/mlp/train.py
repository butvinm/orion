"""Train MLP on MNIST and save weights.

Usage: cd examples/mlp && python train.py
"""

from model import MLP
from orion_compiler.core.utils import train_on_mnist


def main():
    net = MLP()
    train_on_mnist(net, data_dir="./data", epochs=10, save_path="weights.pt")


if __name__ == "__main__":
    main()
