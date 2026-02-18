import time
import functools
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Module(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.level = None
        self.depth = None
        self.fused = False
        self.he_mode = False

    def _set_mode_for_all(self, he_mode=False, training=True):
        for m in self.modules():
            m.training = training
            if hasattr(m, "he_mode"):
                m.he_mode = he_mode

    def _set_attribute_for_all(self, attr, value):
        for m in self.modules():
            setattr(m, attr, value)

    def extra_repr(self):
        torch_repr = super().extra_repr()
        orion_repr = (", " if torch_repr else "") + f"level={self.level}"
        return torch_repr + orion_repr

    def train(self, mode=True):
        self._set_mode_for_all(he_mode=False, training=mode)

    def eval(self):
        self._set_mode_for_all(he_mode=False, training=False)

    def he(self):
        self._set_mode_for_all(he_mode=True, training=False)

    def set_depth(self, depth):
        self.depth = depth

    def set_level(self, level):
        self.level = level

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError(
            f"The 'forward' method is not implemented in {type(self).__name__}. "
            "All Orion modules must override this method with a custom "
            "implementation."
        )


def timer(func):
    @functools.wraps(func)
    @torch.compiler.disable
    def wrapper(self, *args, **kwargs):
        if not self.he_mode:
            return func(self, *args, **kwargs)

        # Read debug status from input tensor's context
        first_arg = args[0] if args else None
        debug_enabled = False
        if first_arg is not None and hasattr(first_arg, 'context'):
            ctx = first_arg.context
            if hasattr(ctx, 'params'):
                debug_enabled = ctx.params.get_debug_status()

        if debug_enabled:
            layer_name = getattr(self, "name", self.__class__.__name__)
            print(f"\n{layer_name}:")
            print(f"Clear input min/max: {self.input_min:.3f} / {self.input_max:.3f}")
            print(f"FHE input min/max: {args[0].min():.3f} / {args[0].max():.3f}")
            start = time.time()

        result = func(self, *args, **kwargs)

        if debug_enabled:
            if hasattr(self, "output_min"):
                output_min = self.output_min
                output_max = self.output_max
            else:
                output_min = self.input_min
                output_max = self.input_max

            elapsed = time.time() - start
            print(f"Clear output min/max: {output_min:.3f} / {output_max:.3f}")
            print(f"FHE output min/max: {result.min():.3f} / {result.max():.3f}")
            print(f"done! [{elapsed:.3f} secs.]")

        return result

    return wrapper
