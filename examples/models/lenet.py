"""LeNet model for MNIST classification.

Architecture: Conv2d(1,32,5,s=2,p=2) → BN2d → Quad → Conv2d(32,64,5,s=2,p=2) → BN2d → Quad
           → Flatten → Linear(3136,512) → BN1d → Quad → Linear(512,10)
"""

import orion_compiler.nn as on

CONFIG = {
    "input_shape": (1, 1, 28, 28),
    "dataset": "mnist",
    # LogQP=269 ≤ 438 (128-bit secure at logN=14, HE Standard).
    # 7 computation levels = exact depth of this network.
    # Needs logN=14 (not 13) because 8 Q primes at scale=26 gives LogQP=269 > 218.
    "ckks_params": dict(
        logn=14,
        logq=[29, 26, 26, 26, 26, 26, 26, 26],
        logp=[29, 29],
        log_default_scale=26,
        ring_type="conjugate_invariant",
    ),
}


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


Model = LeNet
