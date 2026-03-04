"""Train LoLA on MNIST and save weights.

Usage: cd examples/lola && python train.py
"""

from model import LoLA
from orion_compiler.core.utils import train_on_mnist


def main():
    net = LoLA()
    train_on_mnist(net, data_dir="./data", epochs=10, save_path="weights.pt")


if __name__ == "__main__":
    main()
