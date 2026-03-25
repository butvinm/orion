"""Model — loaded .orion v2 file, ready for evaluation."""

import json

from . import ffi
from .errors import EvaluatorError, ModelLoadError
from .gohandle import GoHandle


class Model:
    """Loaded .orion v2 model. Immutable after load.

    Usage::

        model = Model.load(open("model.orion", "rb").read())
        params_json, manifest, input_level = model.client_params()
        # ... use params to create keys, encrypt input ...
        model.close()
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: GoHandle):
        self._handle: GoHandle | None = handle

    @classmethod
    def load(cls, data: bytes) -> "Model":
        """Load a .orion v2 file from bytes.

        Raises:
            ModelLoadError: If the model data is invalid or cannot be parsed.
        """
        try:
            handle = ffi.load_model(data)
        except EvaluatorError as e:
            raise ModelLoadError(str(e)) from e
        return cls(handle)

    def client_params(self) -> tuple[dict, dict, int]:
        """Return (params_dict, manifest_dict, input_level) for client key generation.

        params_dict has keys: logn, logq, logp, log_default_scale, h, ring_type, boot_logp
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
            self._handle = None

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
