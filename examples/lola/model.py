"""LoLA-inspired model for MNIST classification using FHE-compatible layers.

Architecture: Conv2d(1,32,5,s=2,p=2) → BN2d → Quad → Flatten → Linear(6272,100) → BN1d → Quad → Linear(100,10)

Based on LoLA ("Low-Latency") — a single-conv architecture. The original LoLA used
Conv2d(1,5,k=2,s=2,p=0) but small channel counts trigger a packing bug in the
compiler (systematic error regardless of CKKS precision). Adapted to 32 channels
with k=5 padding for FHE compatibility.
"""

import orion_compiler.nn as on


class LoLA(on.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = on.Conv2d(1, 32, kernel_size=5, padding=2, stride=2)
        self.bn1 = on.BatchNorm2d(32)
        self.act1 = on.Quad()

        self.flatten = on.Flatten()
        self.fc1 = on.Linear(32 * 14 * 14, 100)
        self.bn2 = on.BatchNorm1d(100)
        self.act2 = on.Quad()

        self.fc2 = on.Linear(100, num_classes)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.flatten(x)
        x = self.act2(self.bn2(self.fc1(x)))
        return self.fc2(x)
