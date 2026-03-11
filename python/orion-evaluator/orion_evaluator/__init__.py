"""orion-evaluator — Python bindings for the Orion FHE evaluator.

Usage::

    from orion_evaluator import Model, Evaluator

    model = Model.load(open("model.orion", "rb").read())
    params, manifest, input_level = model.client_params()
    evaluator = Evaluator(params, keys_bytes)
    result_bytes = evaluator.forward(model, ciphertext_bytes)
"""

from .evaluator import Evaluator
from .model import Model

__all__ = ["Evaluator", "Model"]
