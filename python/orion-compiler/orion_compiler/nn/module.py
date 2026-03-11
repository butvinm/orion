from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Module(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.level: int | None = None
        self.depth: int | None = None
        self.fused: bool = False

        # Set by tracer (StatsTracker.sync_module_attributes)
        self.input_min: float = float("inf")
        self.input_max: float = float("-inf")
        self.output_min: float = float("inf")
        self.output_max: float = float("-inf")
        self.input_shape: torch.Size | None = None
        self.output_shape: torch.Size | None = None
        self.fhe_input_shape: torch.Size | None = None
        self.fhe_output_shape: torch.Size | None = None
        self.input_gap: int = 1
        self.output_gap: int = 1

        # Set by tracer (StatsTracker.sync_module_attributes)
        self.name: str = ""

        # Set by Compiler during compilation
        self.scheme: Any = None  # Compiler instance (avoid circular import)

    def _set_attribute_for_all(self, attr: str, value: object) -> None:
        for m in self.modules():
            setattr(m, attr, value)

    def extra_repr(self) -> str:
        torch_repr = super().extra_repr()
        orion_repr = (", " if torch_repr else "") + f"level={self.level}"
        return torch_repr + orion_repr

    def set_depth(self, depth: int) -> None:
        self.depth = depth

    def set_level(self, level: int | None) -> None:
        self.level = level

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"The 'forward' method is not implemented in {type(self).__name__}. "
            "All Orion modules must override this method with a custom "
            "implementation."
        )
