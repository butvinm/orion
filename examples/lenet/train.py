"""Train LeNet on MNIST and save weights.

Usage: cd examples/lenet && python train.py
"""

from model import LeNet
from orion_compiler.core.utils import train_on_mnist


def main():
    net = LeNet()
    train_on_mnist(net, data_dir="./data", epochs=10, save_path="weights.pt")


if __name__ == "__main__":
    main()
