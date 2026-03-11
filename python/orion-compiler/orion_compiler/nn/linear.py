import math
from abc import abstractmethod

import torch
import torch.nn as nn

from ..core import packing
from .module import Module


class LinearTransform(Module):
    def __init__(self, bsgs_ratio, level) -> None:
        super().__init__()
        self.bsgs_ratio = float(bsgs_ratio)
        self.set_depth(1)
        self.set_level(level)

        self.diagonals = {}
        self.output_rotations = 0

    def extra_repr(self):
        return super().extra_repr() + f", bsgs_ratio={self.bsgs_ratio}"

    def init_orion_params(self):
        self.on_weight = self.weight.data.clone()
        self.on_bias = (
            self.bias.data.clone()
            if hasattr(self, "bias") and self.bias is not None
            else torch.zeros(self.weight.shape[0])
        )

    @abstractmethod
    def compute_fhe_output_gap(self, **kwargs):
        pass

    @abstractmethod
    def compute_fhe_output_shape(self, **kwargs) -> tuple:
        pass

    @abstractmethod
    def generate_diagonals(self, last: bool):
        pass


class Linear(LinearTransform):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bsgs_ratio: int = 2,
        level: int | None = None,
    ) -> None:
        super().__init__(bsgs_ratio, level)

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            + super().extra_repr()
        )

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def compute_fhe_output_gap(self, **kwargs):
        return 1

    def compute_fhe_output_shape(self, **kwargs) -> tuple:
        return kwargs["clear_output_shape"]

    def generate_diagonals(self, last):
        self.diagonals, self.output_rotations = packing.pack_linear(self, last)

    def forward(self, x):
        if x.dim() != 2:
            extra = " Forgot to call on.Flatten() first?" if x.dim() == 4 else ""
            raise ValueError(
                f"Expected input to {self.__class__.__name__} to have "
                f"2 dimensions (N, in_features), but got {x.dim()} "
                f"dimension(s): {x.shape}." + extra
            )
        return torch.nn.functional.linear(x, self.weight, self.bias)


class Conv2d(LinearTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        bsgs_ratio: int = 2,
        level: int | None = None,
    ) -> None:
        super().__init__(bsgs_ratio, level)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._make_tuple(kernel_size)
        self.stride = self._make_tuple(stride)
        self.padding = self._make_tuple(padding)
        self.dilation = self._make_tuple(dilation)
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def _make_tuple(self, value):
        return (value, value) if isinstance(value, int) else value

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, " + super().extra_repr()
        )

    def compute_fhe_output_gap(self, **kwargs):
        input_gap = kwargs["input_gap"]
        return input_gap * self.stride[0]

    def compute_fhe_output_shape(self, **kwargs) -> tuple:
        input_shape = kwargs["input_shape"]
        clear_output_shape = kwargs["clear_output_shape"]
        input_gap = kwargs["input_gap"]

        Hi, Wi = input_shape[2:]
        N, Co, Ho, Wo = clear_output_shape
        output_gap = self.compute_fhe_output_gap(input_gap=input_gap)

        on_Co = math.ceil(Co / (output_gap**2))
        on_Ho = max(Hi, Ho * output_gap)
        on_Wo = max(Wi, Wo * output_gap)

        return torch.Size((N, on_Co, on_Ho, on_Wo))

    def generate_diagonals(self, last):
        self.diagonals, self.output_rotations = packing.pack_conv2d(self, last)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                f"Expected input to {self.__class__.__name__} to have "
                f" 4 dimensions (N, C, H, W), but got {x.dim()} "
                f"dimension(s): {x.shape}."
            )
        return torch.nn.functional.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
