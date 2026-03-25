"""MLP model for MNIST classification.

Architecture: Flatten → Linear(784,128) → BN → Quad → Linear(128,128) → BN → Quad → Linear(128,10)
"""

import orion_compiler.nn as on

CONFIG = {
    "input_shape": (1, 1, 28, 28),
    "dataset": "mnist",
    "ckks_params": dict(
        logn=13,
        logq=[29, 26, 26, 26, 26, 26, 26, 26, 26, 26],
        logp=[29, 29],
        log_default_scale=26,
        h=8192,
        ring_type="conjugate_invariant",
    ),
}


class MLP(on.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = on.Flatten()

        self.fc1 = on.Linear(784, 128)
        self.bn1 = on.BatchNorm1d(128)
        self.act1 = on.Quad()

        self.fc2 = on.Linear(128, 128)
        self.bn2 = on.BatchNorm1d(128)
        self.act2 = on.Quad()

        self.fc3 = on.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        return self.fc3(x)


Model = MLP
