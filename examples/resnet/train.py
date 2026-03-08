"""Train ResNet20 on CIFAR-10 and save weights.

Usage: cd examples/resnet && python train.py
"""

from model import ResNet20
from orion_compiler.core.utils import train_on_cifar


def main():
    net = ResNet20()
    train_on_cifar(net, data_dir="./data", epochs=200, save_path="weights.pt")


if __name__ == "__main__":
    main()
