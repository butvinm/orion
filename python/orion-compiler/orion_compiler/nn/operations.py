from __future__ import annotations

import math
from typing import Any

import torch

from .module import Module


class Add(Module):
    def __init__(self) -> None:
        super().__init__()
        self.set_depth(0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + y


class Mult(Module):
    def __init__(self) -> None:
        super().__init__()
        self.set_depth(1)

    def forward(self, x: torch.Tensor, y: torch.Tensor | float) -> torch.Tensor:  # type: ignore[override]
        return x * y


class Bootstrap(Module):
    def __init__(self, input_min: float, input_max: float, input_level: int) -> None:
        super().__init__()
        self.input_min = input_min
        self.input_max = input_max
        self.input_level = input_level
        self.prescale: float = 1
        self.postscale: float = 1
        self.constant: float = 0

    def extra_repr(self) -> str:
        return f"input_level={self.input_level}"

    def fit(self, context: Any) -> None:
        margin = context.margin
        center = (self.input_min + self.input_max) / 2
        half_range = (self.input_max - self.input_min) / 2
        self.low = (center - (margin * half_range)).item()
        self.high = (center + (margin * half_range)).item()

        if self.high - self.low > 2:
            self.postscale = math.ceil((self.high - self.low) / 2)
            self.prescale = 1 / self.postscale

        self.constant = -(self.low + self.high) / 2

    def compile(self, context: Any) -> None:
        elements = self.fhe_input_shape.numel()  # type: ignore[operator]
        curr_slots = 2 ** math.ceil(math.log2(elements))

        prescale_vec = torch.zeros(curr_slots)
        prescale_vec[:elements] = self.prescale

        ql = context.encoder.get_moduli_chain()[self.input_level]
        self.prescale_ptxt = context.encoder.encode(prescale_vec, level=self.input_level, scale=ql)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
