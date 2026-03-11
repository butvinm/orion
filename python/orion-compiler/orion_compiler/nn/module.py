from abc import ABC, abstractmethod

import torch.nn as nn


class Module(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.level = None
        self.depth = None
        self.fused = False

    def _set_attribute_for_all(self, attr, value):
        for m in self.modules():
            setattr(m, attr, value)

    def extra_repr(self):
        torch_repr = super().extra_repr()
        orion_repr = (", " if torch_repr else "") + f"level={self.level}"
        return torch_repr + orion_repr

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
