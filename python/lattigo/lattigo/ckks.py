"""CKKS scheme types: Parameters and Encoder.

Wraps Lattigo's ckks.Parameters and ckks.Encoder via the bridge FFI.
"""

from typing import Literal

from . import ffi
from .gohandle import GoHandle

RingType = Literal["standard", "conjugate_invariant"]


class Parameters:
    """CKKS scheme parameters. Wraps Lattigo's ckks.Parameters."""

    def __init__(
        self,
        logn: int,
        logq: list[int],
        logp: list[int],
        log_default_scale: int,
        ring_type: RingType,
        *,
        h: int = 0,
        log_nth_root: int = 0,
    ):
        self._handle = ffi.new_ckks_params(
            logn,
            logq,
            logp,
            log_default_scale,
            h,
            ring_type,
            log_nth_root,
        )

    @classmethod
    def _from_handle(cls, handle: GoHandle) -> "Parameters":
        """Wrap an existing Go handle (internal use)."""
        obj = object.__new__(cls)
        obj._handle = handle
        return obj

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

    def moduli_chain(self) -> list[int]:
        """Q primes (ciphertext moduli chain)."""
        return ffi.ckks_params_moduli_chain(self._handle)

    def aux_moduli_chain(self) -> list[int]:
        """P primes (auxiliary moduli chain for key switching)."""
        return ffi.ckks_params_aux_moduli_chain(self._handle)

    def close(self):
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Encoder:
    """CKKS encoder. Wraps Lattigo's ckks.Encoder."""

    def __init__(self, params: Parameters):
        self._handle = ffi.new_encoder(params._h)
        self._params = params

    @classmethod
    def _from_handle(cls, handle: GoHandle, params: Parameters) -> "Encoder":
        """Wrap an existing Go handle (internal use)."""
        obj = object.__new__(cls)
        obj._handle = handle
        obj._params = params
        return obj

    def encode(self, values: list[float], level: int, scale: int):
        """Encode float values into a Lattigo rlwe.Plaintext.

        Returns an rlwe.Plaintext object (from lattigo.rlwe module).
        Import it lazily to avoid circular imports.
        """
        from . import rlwe

        pt_h = ffi.encoder_encode(self._handle, values, level, scale)
        return rlwe.Plaintext(pt_h)

    def decode(self, pt, num_slots: int) -> list[float]:
        """Decode a Lattigo rlwe.Plaintext back to float values."""
        return ffi.encoder_decode(self._handle, pt._handle, num_slots)

    def close(self):
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
