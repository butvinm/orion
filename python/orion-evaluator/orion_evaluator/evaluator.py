"""Evaluator — runs FHE forward pass on a loaded Model."""

import json

from . import ffi
from .model import Model


class Evaluator:
    """FHE inference evaluator. NOT thread-safe.

    Usage::

        model = Model.load(model_bytes)
        params, manifest, input_level = model.client_params()

        # keys_bytes = MemEvaluationKeySet.MarshalBinary() bytes
        evaluator = Evaluator(params, keys_bytes)
        result_ct_bytes = evaluator.forward(model, input_ct_bytes)
        evaluator.close()
    """
    __slots__ = ('_handle',)

    def __init__(self, params: dict, keys_bytes: bytes, btp_keys_bytes: bytes | None = None):
        """Create evaluator from CKKS params dict and MemEvaluationKeySet binary bytes.

        Args:
            params: CKKS params dict (as returned by Model.client_params()[0])
            keys_bytes: MemEvaluationKeySet.MarshalBinary() output
            btp_keys_bytes: bootstrapping.EvaluationKeys.MarshalBinary() output (optional)
        """
        params_json = json.dumps(params)
        self._handle = ffi.new_evaluator(params_json, keys_bytes, btp_keys_bytes)

    def forward(self, model: Model, ct_bytes: bytes) -> bytes:
        """Run FHE forward pass.

        Args:
            model: Loaded Model
            ct_bytes: Input ciphertext (rlwe.Ciphertext.MarshalBinary() bytes)

        Returns:
            Output ciphertext bytes (rlwe.Ciphertext.MarshalBinary() format)
        """
        if not self._handle:
            raise RuntimeError("Evaluator is closed")
        if not model._handle:
            raise RuntimeError("Model is closed")
        return ffi.evaluator_forward(self._handle, model._handle, ct_bytes)

    def close(self):
        """Release evaluator resources. Idempotent."""
        if self._handle:
            ffi.evaluator_close(self._handle)
            ffi.delete_handle(self._handle)
            self._handle = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
