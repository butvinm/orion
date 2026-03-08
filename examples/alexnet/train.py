"""Train AlexNet on CIFAR-10 and save weights.

Usage: cd examples/alexnet && python train.py
"""

from model import AlexNet
from orion_compiler.core.utils import train_on_cifar


def main():
    net = AlexNet()
    train_on_cifar(net, data_dir="./data", epochs=200, save_path="weights.pt")


if __name__ == "__main__":
    main()
