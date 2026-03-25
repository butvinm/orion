"""AlexNet model for CIFAR-10 classification.

Architecture: 5 Conv2d blocks with SiLU activations, AvgPool2d downsampling,
AdaptiveAvgPool2d, then 3 FC layers. Input: 3x32x32 (CIFAR-10).

Requires bootstrap (3 operations). Full FHE E2E needs 64+ GB RAM.
"""

import orion_compiler.nn as on

CONFIG = {
    "input_shape": (1, 3, 32, 32),
    "dataset": "cifar",
    "ckks_params": dict(
        logn=15,
        logq=[55] + [40] * 20,
        logp=[61, 61, 61],
        log_default_scale=40,
        h=192,
        ring_type="standard",
        boot_logp=[61] * 6,
    ),
}


class AlexNet(on.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = on.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = on.BatchNorm2d(64)
        self.act1 = on.SiLU(degree=127)

        self.pool1 = on.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = on.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.bn2 = on.BatchNorm2d(192)
        self.act2 = on.SiLU(degree=127)

        self.pool2 = on.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = on.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.bn3 = on.BatchNorm2d(384)
        self.act3 = on.SiLU(degree=127)

        self.conv4 = on.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = on.BatchNorm2d(256)
        self.act4 = on.SiLU(degree=127)

        self.conv5 = on.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = on.BatchNorm2d(256)
        self.act5 = on.SiLU(degree=127)

        self.adapt_pool = on.AdaptiveAvgPool2d((2, 2))
        self.flatten = on.Flatten()

        self.fc1 = on.Linear(1024, 4096)
        self.bn6 = on.BatchNorm1d(4096)
        self.act6 = on.SiLU(degree=127)

        self.fc2 = on.Linear(4096, 4096)
        self.bn7 = on.BatchNorm1d(4096)
        self.act7 = on.SiLU(degree=127)

        self.fc3 = on.Linear(4096, num_classes)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.bn5(self.conv5(x)))
        x = self.adapt_pool(x)
        x = self.flatten(x)
        x = self.act6(self.bn6(self.fc1(x)))
        x = self.act7(self.bn7(self.fc2(x)))
        return self.fc3(x)


Model = AlexNet
