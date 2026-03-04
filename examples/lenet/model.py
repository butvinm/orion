"""LeNet model for MNIST classification using FHE-compatible layers.

Architecture: Conv2d(1,32,5,s=2,p=2) → BN2d → Quad → Conv2d(32,64,5,s=2,p=2) → BN2d → Quad
           → Flatten → Linear(3136,512) → BN1d → Quad → Linear(512,10)
"""

import orion_compiler.nn as on


class LeNet(on.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = on.Conv2d(1, 32, kernel_size=5, padding=2, stride=2)
        self.bn1 = on.BatchNorm2d(32)
        self.act1 = on.Quad()

        self.conv2 = on.Conv2d(32, 64, kernel_size=5, padding=2, stride=2)
        self.bn2 = on.BatchNorm2d(64)
        self.act2 = on.Quad()

        self.flatten = on.Flatten()
        self.fc1 = on.Linear(7 * 7 * 64, 512)
        self.bn3 = on.BatchNorm1d(512)
        self.act3 = on.Quad()

        self.fc2 = on.Linear(512, num_classes)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.bn3(self.fc1(x)))
        return self.fc2(x)
