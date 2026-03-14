import torch

from .module import Module


class Flatten(Module):
    def __init__(self) -> None:
        super().__init__()
        self.set_depth(0)

    def extra_repr(self) -> str:
        return super().extra_repr() + ", start_dim=1"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=1)
