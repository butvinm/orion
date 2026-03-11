from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn

from ..core import packing
from .module import Module


class BatchNormNd(Module):
    def __init__(
        self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.fused = False
        self.set_depth(2 if affine else 1)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    @abstractmethod
    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError("Subclasses must implement _check_input_dim")

    def init_orion_params(self) -> None:
        self.on_running_mean = self.running_mean.data.clone()  # type: ignore[operator]
        self.on_running_var = self.running_var.data.clone()  # type: ignore[operator]

        if self.affine:
            self.on_weight = self.weight.data.clone()
            self.on_bias = self.bias.data.clone()
        else:
            self.on_weight = torch.ones_like(self.on_running_mean)
            self.on_bias = torch.zeros_like(self.on_running_mean)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", level={self.level}, fused={self.fused}"

    def _compile_with_vectors(
        self,
        context: Any,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor | None,
        d: torch.Tensor | None,
    ) -> None:
        assert self.level is not None
        level = self.level
        encoder = context.encoder

        q1 = encoder.get_moduli_chain()[level]
        q2 = encoder.get_moduli_chain()[level - 1]

        self.on_running_mean_ptxt = encoder.encode(a, level=level, scale=q1)
        self.on_inv_running_std_ptxt = encoder.encode(b, level=level, scale=q1)

        if self.affine:
            self.on_weight_ptxt = encoder.encode(c, level=level - 1, scale=q2)
            self.on_bias_ptxt = encoder.encode(d, level=level - 1, scale=q2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        if self.training:
            exponential_average_factor = 0.0
            if self.momentum is not None:
                exponential_average_factor = self.momentum
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1  # type: ignore[operator,assignment]
                if self.momentum is None:
                    exponential_average_factor = 1.0 / self.num_batches_tracked
        else:
            exponential_average_factor = 0.0

        return torch.nn.functional.batch_norm(
            x,
            self.running_mean,  # type: ignore[arg-type]
            self.running_var,  # type: ignore[arg-type]
            self.weight,
            self.bias,
            self.training,
            exponential_average_factor,
            self.eps,
        )


class BatchNorm1d(BatchNormNd):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 2 and x.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {x.dim()}D input)")

    def compile(self, context: Any) -> None:
        a, b, c, d = packing.pack_bn1d(self)
        self._compile_with_vectors(context, a, b, c, d)


class BatchNorm2d(BatchNormNd):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"expected 4D input (got {x.dim()}D input)")

    def compile(self, context: Any) -> None:
        a, b, c, d = packing.pack_bn2d(self)
        self._compile_with_vectors(context, a, b, c, d)
