"""Train VGG16 on CIFAR-10 and save weights.

Usage: cd examples/vgg && python train.py
"""

from model import VGG
from orion_compiler.core.utils import train_on_cifar


def main():
    net = VGG("VGG16")
    train_on_cifar(net, data_dir="./data", epochs=200, save_path="weights.pt")


if __name__ == "__main__":
    main()
