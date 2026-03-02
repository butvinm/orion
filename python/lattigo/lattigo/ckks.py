"""CKKS scheme types: Parameters and Encoder.

Wraps Lattigo's ckks.Parameters and ckks.Encoder via the bridge FFI.
"""

from __future__ import annotations

import json

from . import ffi
from .gohandle import GoHandle


class Parameters:
    """CKKS scheme parameters. Wraps Lattigo's ckks.Parameters."""

    def __init__(self, handle: GoHandle):
        self._handle = handle

    @classmethod
    def from_logn(
        cls,
        logn: int,
        logq: list[int],
        logp: list[int],
        logscale: int,
        h: int = 192,
        ring_type: str = "conjugate_invariant",
    ) -> Parameters:
        """Create CKKS parameters from log-scale specifications."""
        params_json = json.dumps({
            "logn": logn,
            "logq": logq,
            "logp": logp,
            "logscale": logscale,
            "h": h,
            "ring_type": ring_type,
        })
        handle = ffi.new_ckks_params(params_json)
        return cls(handle)

    @classmethod
    def from_json(cls, params_json: str) -> Parameters:
        """Create from a JSON string (same format as orion.CKKSParams)."""
        handle = ffi.new_ckks_params(params_json)
        return cls(handle)

    @property
    def _h(self) -> GoHandle:
        return self._handle

    def max_slots(self) -> int:
        """Maximum number of plaintext slots."""
        return ffi.ckks_params_max_slots(self._handle)

    def max_level(self) -> int:
        """Maximum multiplicative level (len(logq) - 1)."""
        return ffi.ckks_params_max_level(self._handle)

    def default_scale(self) -> int:
        """Default scale as uint64 (2^logscale)."""
        return ffi.ckks_params_default_scale(self._handle)

    def galois_element(self, rotation: int) -> int:
        """Galois element for a given rotation step."""
        return ffi.ckks_params_galois_element(self._handle, rotation)

    def close(self):
        self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Encoder:
    """CKKS encoder. Wraps Lattigo's ckks.Encoder."""

    def __init__(self, handle: GoHandle, params: Parameters):
        self._handle = handle
        self._params = params

    @classmethod
    def new(cls, params: Parameters) -> Encoder:
        """Create a new CKKS encoder for the given parameters."""
        handle = ffi.new_ckks_encoder(params._h)
        return cls(handle, params)

    def encode(self, values: list[float], level: int, scale: int):
        """Encode float values into a Lattigo rlwe.Plaintext.

        Returns an rlwe.Plaintext object (from lattigo.rlwe module).
        Import it lazily to avoid circular imports.
        """
        from . import rlwe
        pt_h = ffi.ckks_encoder_encode(
            self._handle, self._params._h, values, level, scale,
        )
        return rlwe.Plaintext(pt_h)

    def decode(self, pt, num_slots: int) -> list[float]:
        """Decode a Lattigo rlwe.Plaintext back to float values."""
        return ffi.ckks_encoder_decode(self._handle, pt._handle, num_slots)

    def close(self):
        self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
