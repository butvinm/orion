from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Module(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.level: int | None = None
        self.depth: int | None = None
        self.fused: bool = False

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
