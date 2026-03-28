"""VGG16 model for CIFAR-10 classification.

Architecture: 13 Conv2d blocks with ReLU activations (minimax sign approximation,
degrees=[15,15,27]), 5 AvgPool2d downsampling layers, then a single FC layer.
Input: 3x32x32 (CIFAR-10).

Requires bootstrap (~13 operations). Full FHE E2E needs >128 GB RAM.
"""

import orion_compiler.nn as on
import torch.nn as nn

CONFIG = {
    "input_shape": (1, 3, 32, 32),
    "dataset": "cifar",
    # Bootstrap LogQP=1642 ≤ 1770 (128-bit secure at logN=16, HE Standard).
    # Reduced from 21 to 11 Q primes so bootstrap LogQP fits threshold.
    "ckks_params": dict(
        logn=16,
        logq=[55] + [40] * 10,
        logp=[61, 61, 61],
        log_default_scale=40,
        ring_type="standard",
        boot_logp=[61] * 6,
    ),
}

VGG_CONFIGS = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(on.Module):
    def __init__(self, vgg_name="VGG16", num_classes=10):
        super().__init__()
        self.features = self._make_layers(VGG_CONFIGS[vgg_name])
        self.flatten = on.Flatten()
        self.classifier = on.Linear(512, num_classes)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers.append(on.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.extend(
                    [
                        on.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        on.BatchNorm2d(x),
                        on.ReLU(degrees=[15, 15, 27]),
                    ]
                )
                in_channels = x
        layers.append(on.AvgPool2d(kernel_size=1, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


Model = VGG
