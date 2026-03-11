from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .linear import Conv2d


class AvgPool2d(Conv2d):
    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        bsgs_ratio: int = 2,
        level: int | None = None,
    ) -> None:
        super().__init__(
            1,
            1,
            kernel_size,
            stride or kernel_size,
            padding,
            dilation=1,
            groups=1,
            bias=False,
            bsgs_ratio=bsgs_ratio,
            level=level,
        )

    def extra_repr(self) -> str:
        return (
            f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride} "
            + f"level = {self.level})"
        )

    def update_params(self) -> None:
        kH, kW = self.kernel_size
        self.in_channels = self.out_channels = self.groups = self.input_shape[1]  # type: ignore[index,assignment]
        self.on_weight = torch.ones(self.out_channels, 1, kH, kW) / (kH * kW)
        self.on_bias = torch.zeros(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)


class AdaptiveAvgPool2d(AvgPool2d):
    def __init__(
        self,
        output_size: int | tuple[int, int],
        bsgs_ratio: int = 2,
        level: int | None = None,
    ) -> None:
        super().__init__(kernel_size=1, stride=1, padding=0, bsgs_ratio=bsgs_ratio, level=level)

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def extra_repr(self) -> str:
        return f"AdaptiveAvgPool2d(output_size={self.output_size}) " + f"level={self.level})"

    def update_params(self) -> None:
        Hi, Wi = self.input_shape[2:]  # type: ignore[index]
        Ho, Wo = self.output_size

        self.stride = (Hi // Ho, Wi // Wo)
        kH = Hi - (Ho - 1) * self.stride[0]
        kW = Wi - (Wo - 1) * self.stride[1]
        self.kernel_size = (kH, kW)
        super().update_params()

    def compute_fhe_output_gap(self, **kwargs: Any) -> int:
        input_gap = kwargs["input_gap"]
        input_shape = kwargs["input_shape"]
        output_shape = kwargs["output_shape"]
        return input_gap * (input_shape[2] // output_shape[2])  # type: ignore[no-any-return]

    def compute_fhe_output_shape(self, **kwargs: Any) -> tuple[int, ...]:
        input_shape = kwargs["input_shape"]
        output_shape = kwargs["clear_output_shape"]
        input_gap = kwargs["input_gap"]

        Hi, Wi = input_shape[2:]
        No, Co, Ho, Wo = output_shape

        output_gap = self.compute_fhe_output_gap(
            input_gap=input_gap, input_shape=input_shape, output_shape=output_shape
        )

        on_Co = math.ceil(Co / (output_gap**2))
        on_Ho = max(Hi, Ho * output_gap)
        on_Wo = max(Wi, Wo * output_gap)

        return torch.Size((No, on_Co, on_Ho, on_Wo))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Ho, Wo = self.output_size
        if x.shape[2] % Ho != 0 or x.shape[3] % Wo != 0:
            raise ValueError(
                f"Output spatial dimensions {self.output_size} are not "
                + f"a multiple of the input spatial dimensions {x.shape[2:]}."
            )
        return F.adaptive_avg_pool2d(x, self.output_size)
