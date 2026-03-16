"""orion-evaluator — Python bindings for the Orion FHE evaluator.

Usage::

    from orion_evaluator import Model, Evaluator

    model = Model.load(open("model.orion", "rb").read())
    params, manifest, input_level = model.client_params()
    evaluator = Evaluator(params, keys_bytes)
    result_bytes_list = evaluator.forward(model, [ciphertext_bytes])
"""

from .errors import EvaluatorError, ModelLoadError
from .evaluator import Evaluator
from .model import Model

__all__ = ["Evaluator", "EvaluatorError", "Model", "ModelLoadError"]

__version__ = "2.0.2"
