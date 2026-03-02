import math
import torch

from .module import Module

class Add(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)

    def forward(self, x, y):
        return x + y


class Mult(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(1)

    def forward(self, x, y):
        return x * y


class Bootstrap(Module):
    def __init__(self, input_min, input_max, input_level):
        super().__init__()
        self.input_min = input_min
        self.input_max = input_max
        self.input_level = input_level
        self.prescale = 1
        self.postscale = 1
        self.constant = 0

    def extra_repr(self):
        return f"input_level={self.input_level}"

    def fit(self, context):
        margin = context.margin
        center = (self.input_min + self.input_max) / 2
        half_range = (self.input_max - self.input_min) / 2
        self.low = (center - (margin * half_range)).item()
        self.high = (center + (margin * half_range)).item()

        if self.high - self.low > 2:
            self.postscale = math.ceil((self.high - self.low) / 2)
            self.prescale = 1 / self.postscale

        self.constant = -(self.low + self.high) / 2

    def compile(self, context):
        elements = self.fhe_input_shape.numel()
        curr_slots = 2 ** math.ceil(math.log2(elements))

        prescale_vec = torch.zeros(curr_slots)
        prescale_vec[:elements] = self.prescale

        ql = context.encoder.get_moduli_chain()[self.input_level]
        self.prescale_ptxt = context.encoder.encode(
            prescale_vec, level=self.input_level, scale=ql)

    def forward(self, x):
        return x
