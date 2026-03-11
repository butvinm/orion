"""Model — loaded .orion v2 file, ready for evaluation."""

import json

from . import ffi
from .errors import EvaluatorError


class Model:
    """Loaded .orion v2 model. Immutable after load.

    Usage::

        model = Model.load(open("model.orion", "rb").read())
        params_json, manifest, input_level = model.client_params()
        # ... use params to create keys, encrypt input ...
        model.close()
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: int):
        self._handle = handle

    @classmethod
    def load(cls, data: bytes) -> "Model":
        """Load a .orion v2 file from bytes."""
        handle = ffi.load_model(data)
        return cls(handle)

    def client_params(self) -> tuple[dict, dict, int]:
        """Return (params_dict, manifest_dict, input_level) for client key generation.

        params_dict has keys: logn, logq, logp, logscale, h, ring_type, boot_logp
        manifest_dict has keys: galois_elements, bootstrap_slots, boot_logp, needs_rlk
        """
        if not self._handle:
            raise EvaluatorError("Model is closed")
        params_json, manifest_json, input_level = ffi.model_client_params(self._handle)
        return json.loads(params_json), json.loads(manifest_json), input_level

    def close(self) -> None:
        """Release model resources. Idempotent."""
        if self._handle:
            ffi.model_close(self._handle)
            ffi.delete_handle(self._handle)
            self._handle = 0

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
