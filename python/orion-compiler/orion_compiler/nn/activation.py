from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from orion_compiler.nn.module import Module
from orion_compiler.nn.operations import Mult

if TYPE_CHECKING:
    from orion_compiler.core.compiler_backend import CompilationContext


class Activation(Module):
    def __init__(self, coeffs: list[float]) -> None:
        super().__init__()
        self.coeffs = coeffs
        self.output_scale: float | None = None
        self.set_depth()

    def extra_repr(self) -> str:
        return super().extra_repr() + f", degree={len(self.coeffs) - 1}"

    def set_depth(self, depth: int | None = None) -> None:
        self.depth = depth if depth is not None else math.ceil(math.log2(len(self.coeffs)))

    def set_output_scale(self, output_scale: float) -> None:
        self.output_scale = output_scale

    def compile(self, context: CompilationContext) -> None:
        self.poly = context.poly_evaluator.generate_monomial(self.coeffs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Horner's method
        out: torch.Tensor | int = 0
        for coeff in self.coeffs:
            out = coeff + x * out

        return out  # type: ignore[return-value]


class Quad(Module):
    def __init__(self) -> None:
        super().__init__()
        self.set_depth(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class Chebyshev(Module):
    def __init__(self, degree: int, fn: Any, within_composite: bool = False) -> None:
        super().__init__()
        self.degree = degree
        self.fn = fn
        self.within_composite = within_composite
        self.coeffs: list[float] | None = None

        self.output_scale: float | None = None
        self.prescale: float = 1
        self.constant: float = 0

    def extra_repr(self) -> str:
        return super().extra_repr() + f", degree={self.degree}"

    def fit(self, context: CompilationContext) -> None:
        if not self.within_composite:
            margin = context.margin
            center = (self.input_min + self.input_max) / 2
            half_range = (self.input_max - self.input_min) / 2
            self.low = float(center - (margin * half_range))
            self.high = float(center + (margin * half_range))

            nodes = np.polynomial.chebyshev.chebpts1(self.degree + 1)
            if self.low < -1 or self.high > 1:
                self.prescale = 2 / (self.high - self.low)
                self.constant = -self.prescale * (self.low + self.high) / 2
                evals = (nodes + 1) * (self.high - self.low) / 2 + self.low
            else:
                evals = nodes

            evals_t = torch.tensor(evals)
            T = np.polynomial.Chebyshev.fit(nodes, self.fn(evals_t), self.degree)
            self.set_coeffs(T.coef.tolist())
            self.set_depth()

    def set_coeffs(self, coeffs: list[float]) -> None:
        self.coeffs = coeffs

    def set_depth(self, depth: int | None = None) -> None:
        if depth is not None:
            self.depth = depth
        else:
            self.depth = math.ceil(math.log2(self.degree + 1))
            if self.prescale != 1:
                self.depth += 1

    def set_output_scale(self, output_scale: float) -> None:
        self.output_scale = output_scale

    def compile(self, context: CompilationContext) -> None:
        assert self.coeffs is not None
        self.poly = context.poly_evaluator.generate_chebyshev(self.coeffs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)  # type: ignore[no-any-return]


class ELU(Chebyshev):
    def __init__(self, alpha: float = 1.0, degree: int = 31) -> None:
        self.alpha = alpha
        super().__init__(degree, self.fn)

    def fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))


class Hardshrink(Chebyshev):
    def __init__(self, degree: int = 31, lambd: float = 0.5) -> None:
        self.lambd = lambd
        super().__init__(degree, self.fn)

    def fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where((x > self.lambd) | (x < -self.lambd), x, torch.tensor(0.0))


class GELU(Chebyshev):
    def __init__(self, degree: int = 31) -> None:
        super().__init__(degree, self.fn)

    def fn(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class SiLU(Chebyshev):
    def __init__(self, degree: int = 31) -> None:
        super().__init__(degree, self.fn)

    def fn(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class Sigmoid(Chebyshev):
    def __init__(self, degree: int = 31) -> None:
        super().__init__(degree, self.fn)

    def fn(self, x: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(x)


class SELU(Chebyshev):
    def __init__(self, degree: int = 31) -> None:
        super().__init__(degree, self.fn)

    def fn(self, x: torch.Tensor) -> torch.Tensor:
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        return scale * torch.where(x > 0, x, alpha * (torch.exp(x) - 1))


class Softplus(Chebyshev):
    def __init__(self, degree: int = 31) -> None:
        super().__init__(degree, self.fn)

    def fn(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)


class Mish(Chebyshev):
    def __init__(self, degree: int = 31) -> None:
        super().__init__(degree, self.fn)

    def fn(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class _Sign(Module):
    def __init__(
        self,
        degrees: list[int] | None = None,
        prec: int = 128,
        logalpha: int = 6,
        logerr: int = 12,
    ) -> None:
        if degrees is None:
            degrees = [15, 15, 27]
        super().__init__()
        self.degrees = degrees
        self.prec = prec
        self.logalpha = logalpha
        self.logerr = logerr
        self.mult = Mult()

        acts = []
        for i, degree in enumerate(degrees):
            is_last = i == len(degrees) - 1
            fn = self.fn1 if not is_last else self.fn2
            act = Chebyshev(degree, fn, within_composite=True)
            acts.append(act)

        self.acts = nn.Sequential(*acts)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", degrees={self.degrees}"

    def fit(self, context: CompilationContext) -> None:
        self.coeffs = context.poly_evaluator.generate_minimax_sign_coeffs(
            self.degrees, self.prec, self.logalpha, self.logerr
        )

        for i, coeffs in enumerate(self.coeffs):
            self.acts[i].set_coeffs(coeffs)  # type: ignore[operator]
            self.acts[i].set_depth()  # type: ignore[operator]

    def fn1(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x <= 0, torch.tensor(-1.0), torch.tensor(1.0))

    def fn2(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x <= 0, torch.tensor(0.0), torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for act in self.acts:
            x = act(x)
        return x


class ReLU(Module):
    def __init__(
        self,
        degrees: list[int] | None = None,
        prec: int = 128,
        logalpha: int = 6,
        logerr: int = 12,
    ) -> None:
        if degrees is None:
            degrees = [15, 15, 27]
        super().__init__()
        self.degrees = degrees
        self.prec = prec
        self.logalpha = logalpha
        self.logerr = logerr
        self.sign = _Sign(degrees, prec, logalpha, logerr)
        self.mult1 = Mult()
        self.mult2 = Mult()

        self.prescale: float = 1
        self.postscale: float = 1

    def extra_repr(self) -> str:
        return super().extra_repr() + f", degrees={self.degrees}"

    def fit(self, context: CompilationContext) -> None:
        self.input_min = self.mult1.input_min
        self.input_max = self.mult1.input_max

        margin = context.margin
        absmax = max(abs(self.input_min), abs(self.input_max)) * margin
        if absmax > 1:
            self.postscale = math.ceil(absmax)
            self.prescale = 1 / self.postscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mult1(x, self.prescale)
        x = self.mult2(x, self.sign(x))
        x *= self.postscale
        return x
