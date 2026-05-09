"""C3AE Age Verification Model — True ReLU variant (cleartext only).

Uses standard torch.nn primitives with ReLU activations. This variant is
NOT compiled to .orion (no Quad approximation), so it serves as a cleartext
quality baseline against the FHE-compatible Quad variant in c3ae_fhe.py.

Architecture mirrors examples/c3ae-demo/model.py:13-86 exactly, with each
orion_compiler.nn layer swapped for its torch.nn counterpart and Quad
swapped for ReLU(inplace=False).

Input:  64x64x3 RGB face image
Output: Single logit (0=minor, 1=adult)
"""

import torch.nn as nn


class C3AE(nn.Module):
    """C3AE-style CNN for binary age classification (18+ verification).

    Args:
        img_size: Input image size (default: 64)
        first_stride: Stride for first conv (1 or 2, default: 2)
                     stride=2 reduces spatial dims early for ~4.5x speedup
    """

    def __init__(self, img_size=64, first_stride=2):
        super().__init__()

        # Block 1: 3->32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=first_stride, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=False)
        self.pool1 = nn.AvgPool2d(2)

        # Block 2: 32->32 channels
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU(inplace=False)
        self.pool2 = nn.AvgPool2d(2)

        # Block 3: 32->32 channels
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.act3 = nn.ReLU(inplace=False)
        self.has_pool3 = first_stride == 1
        if self.has_pool3:
            self.pool3 = nn.AvgPool2d(2)

        # Block 4: 32->32 channels
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.act4 = nn.ReLU(inplace=False)

        # Block 5: 1x1 conv (channel mixing)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=1, bias=True)
        self.act5 = nn.ReLU(inplace=False)

        # Compute flatten size
        # stride=2: 64->31->15->13->6->4->2->2, flat=128
        # stride=1: 64->62->31->29->14->12->6->4->4, flat=512
        s = img_size
        s = (s - 3) // first_stride + 1  # conv1
        s = s // 2  # pool1
        s = s - 2  # conv2
        s = s // 2  # pool2
        s = s - 2  # conv3
        if first_stride == 1:
            s = s // 2  # pool3
        s = s - 2  # conv4
        flat_size = 32 * s * s

        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_size, 12)
        self.act6 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(12, 1)

        self.first_stride = first_stride

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.act3(self.bn3(self.conv3(x)))
        if self.has_pool3:
            x = self.pool3(x)
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.conv5(x))
        x = self.flatten(x)
        x = self.act6(self.fc1(x))
        return self.fc2(x)
