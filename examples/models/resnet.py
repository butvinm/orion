"""ResNet20 model for CIFAR-10 classification.

Architecture: ResNet20 with BasicBlock, [3,3,3] blocks, [16,32,64] channels.
Uses ReLU activation approximated via minimax sign polynomials (degrees=[15,15,27]).
Residual connections via on.Add. Input: 3x32x32 (CIFAR-10).

Requires bootstrap (~38 operations). Full FHE E2E needs 64+ GB RAM.
"""

import torch.nn as nn
import orion_compiler.nn as on

CONFIG = {
    "input_shape": (1, 3, 32, 32),
    "dataset": "cifar",
    "ckks_params": dict(
        logn=16,
        logq=[55] + [40] * 10,
        logp=[61, 61, 61],
        log_default_scale=40,
        h=192,
        ring_type="standard",
        boot_logp=[61] * 8,
    ),
}


class BasicBlock(on.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = on.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = on.BatchNorm2d(out_channels)
        self.act1 = on.ReLU(degrees=[15, 15, 27])

        self.conv2 = on.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = on.BatchNorm2d(out_channels)
        self.act2 = on.ReLU(degrees=[15, 15, 27])

        self.add = on.Add()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                on.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                on.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add(out, self.shortcut(x))
        return self.act2(out)


class ResNet20(on.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 16

        self.conv1 = on.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = on.BatchNorm2d(16)
        self.act = on.ReLU(degrees=[15, 15, 27])

        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)

        self.avgpool = on.AdaptiveAvgPool2d((1, 1))
        self.flatten = on.Flatten()
        self.linear = on.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        return self.linear(out)


Model = ResNet20
