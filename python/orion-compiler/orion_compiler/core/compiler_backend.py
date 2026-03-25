"""Compile-time backend adapter wrapping Lattigo FFI.

Provides NewParameters (adapter), NewEncoder, PolynomialGenerator,
and CompilerBackend — all backed by the lattigo bridge shared library.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from lattigo import ffi as lattigo_ffi
from lattigo.gohandle import GoHandle

from orion_compiler.params import CKKSParams, CompilerConfig

# =========================================================================
# NewParameters — thin adapter over CKKSParams + CompilerConfig
# =========================================================================


class NewParameters:
    """Adapter providing getter methods expected by the compilation pipeline.

    Wraps CKKSParams (cryptographic parameters) and CompilerConfig
    (compilation settings) directly — no intermediate dataclasses.
    """

    def __init__(self, ckks: CKKSParams, config: CompilerConfig):
        self._ckks = ckks
        self._config = config

    @classmethod
    def from_ckks_params(
        cls, ckks_params: CKKSParams, config: CompilerConfig | None = None
    ) -> NewParameters:
        if config is None:
            config = CompilerConfig()
        return cls(ckks_params, config)

    # -- CKKS parameter getters --

    def get_logn(self) -> int:
        return self._ckks.logn

    def get_logq(self) -> list[int]:
        return list(self._ckks.logq)

    def get_logp(self) -> list[int]:
        return list(self._ckks.logp)

    def get_logscale(self) -> int:
        return self._ckks.log_default_scale

    def get_default_scale(self) -> int:
        return 1 << self._ckks.log_default_scale

    def get_hamming_weight(self) -> int:
        return self._ckks.h

    def get_ringtype(self) -> str:
        return self._ckks.ring_type

    def get_max_level(self) -> int:
        return self._ckks.max_level

    def get_slots(self) -> int:
        return self._ckks.max_slots

    def get_ring_degree(self) -> int:
        return self._ckks.ring_degree

    def get_boot_logp(self) -> list[int] | None:
        if self._ckks.boot_logp is None:
            return None
        return list(self._ckks.boot_logp)

    # -- Compiler config getters --

    def get_margin(self) -> int:
        return self._config.margin

    def get_fuse_modules(self) -> bool:
        return self._config.fuse_modules

    def get_embedding_method(self) -> str:
        return self._config.embedding_method

    def get_debug_status(self) -> bool:
        return False

    def get_backend(self) -> str:
        return "lattigo"


# =========================================================================
# CompilerBackend - wraps Lattigo FFI for compile-time operations
# =========================================================================


class CompilerBackend:
    """Adapter providing the LattigoLibrary interface via Lattigo FFI.

    The compiler and its helpers (NewEncoder, PolynomialGenerator)
    call self.backend.XXX(). This class provides those methods by
    delegating to the lattigo bridge FFI using CKKS Parameters and Encoder.
    """

    def __init__(self) -> None:
        self._params_h: GoHandle | None = None
        self._encoder_h: GoHandle | None = None
        self._max_slots: int | None = None

    def setup_bindings(self, params: NewParameters) -> None:
        """Initialize Go backend with the given parameters.

        Creates CKKS Parameters and Encoder for compile-time encode/decode.
        """
        ckks = params._ckks

        log_nth_root = 0
        if ckks.btp_logn and ckks.btp_logn > 0:
            log_nth_root = ckks.btp_logn + 1

        self._params_h = lattigo_ffi.new_ckks_params(
            logn=ckks.logn,
            logq=list(ckks.logq),
            logp=list(ckks.logp),
            log_default_scale=ckks.log_default_scale,
            h=ckks.h,
            ring_type=ckks.ring_type,
            log_nth_root=log_nth_root,
        )
        self._encoder_h = lattigo_ffi.new_encoder(self._params_h)
        self._max_slots = lattigo_ffi.ckks_params_max_slots(self._params_h)

    # -- Encoder operations --

    def NewEncoder(self) -> None:
        """No-op. Encoder created in setup_bindings."""
        pass

    def Encode(self, values: Sequence[float] | list[float], level: int, scale: int) -> GoHandle:
        """Encode values into a plaintext. Returns a GoHandle."""
        assert self._encoder_h is not None
        if not isinstance(values, list):
            values = list(values)
        return lattigo_ffi.encoder_encode(
            self._encoder_h,
            values,
            level,
            scale,
        )

    def Decode(self, pt_h: GoHandle) -> list[float]:
        """Decode a plaintext handle to float values."""
        assert self._encoder_h is not None
        assert self._max_slots is not None
        return lattigo_ffi.encoder_decode(
            self._encoder_h,
            pt_h,
            self._max_slots,
        )

    def DeletePlaintext(self, pt_h: GoHandle) -> None:
        """Delete a plaintext handle."""
        pt_h.close()

    # -- Parameter queries --

    def GetMaxSlots(self) -> int | None:
        return self._max_slots

    def GetGaloisElement(self, rotation: int) -> int:
        assert self._params_h is not None
        return lattigo_ffi.ckks_params_galois_element(self._params_h, rotation)

    def GetModuliChain(self) -> list[int]:
        assert self._params_h is not None
        return lattigo_ffi.ckks_params_moduli_chain(self._params_h)

    def GetAuxModuliChain(self) -> list[int]:
        assert self._params_h is not None
        return lattigo_ffi.ckks_params_aux_moduli_chain(self._params_h)

    # -- Polynomial operations --

    def GenerateMonomial(self, coeffs: Sequence[float] | list[float]) -> GoHandle:
        """Generate a monomial polynomial. Returns a GoHandle."""
        if not isinstance(coeffs, list):
            coeffs = list(coeffs)
        return lattigo_ffi.new_polynomial_monomial(coeffs)

    def GenerateChebyshev(self, coeffs: Sequence[float] | list[float]) -> GoHandle:
        """Generate a Chebyshev polynomial. Returns a GoHandle."""
        if not isinstance(coeffs, list):
            coeffs = list(coeffs)
        return lattigo_ffi.new_polynomial_chebyshev(coeffs)

    def GenerateMinimaxSignCoeffs(
        self, degrees: list[int], prec: int, logalpha: int, logerr: int, debug: int
    ) -> list[float]:
        """Generate minimax sign polynomial coefficients.

        Calls raw Lattigo minimax, applies sign→[0,1] rescaling
        (divide last poly by 2, add 0.5 to constant), and caches results.
        """
        return _minimax_sign_cached(degrees, prec, logalpha, logerr, debug)

    # -- Lifecycle --

    def DeleteScheme(self) -> None:
        """Release Go resources."""
        if self._encoder_h:
            self._encoder_h.close()
            self._encoder_h = None
        if self._params_h:
            self._params_h.close()
            self._params_h = None

    def close(self) -> None:
        """Release the Go backend. Idempotent."""
        self.DeleteScheme()

    def __enter__(self) -> CompilerBackend:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# =========================================================================
# Minimax sign coefficients — caching + sign→[0,1] rescaling
# (Moved from Go orion.GenerateMinimaxSignCoeffs)
# =========================================================================

_minimax_cache: dict[str, list[float]] = {}
_minimax_cache_lock = threading.Lock()


def _minimax_cache_key(degrees: list[int], prec: int, logalpha: int, logerr: int) -> str:
    return f"{','.join(str(d) for d in degrees)}|{prec}|{logalpha}|{logerr}"


def _minimax_sign_cached(
    degrees: list[int],
    prec: int,
    logalpha: int,
    logerr: int,
    debug: int,
) -> list[float]:
    """Generate minimax sign coefficients with caching and sign→[0,1] rescaling."""
    cleaned = [d for d in degrees if d != 0]
    if not cleaned:
        raise ValueError("At least one non-zero degree must be provided")

    key = _minimax_cache_key(cleaned, prec, logalpha, logerr)
    with _minimax_cache_lock:
        if key in _minimax_cache:
            return list(_minimax_cache[key])

    # Call raw Lattigo bridge (no rescaling)
    flat_coeffs, seps = lattigo_ffi.gen_minimax_composite_polynomial(
        cleaned,
        prec,
        logalpha,
        logerr,
        debug,
    )

    # Split into per-polynomial lists using separator indices
    polys = []
    for i, start in enumerate(seps):
        end = seps[i + 1] if i + 1 < len(seps) else len(flat_coeffs)
        polys.append(list(flat_coeffs[start:end]))

    # Scale last polynomial from sign [-1,1] → sigmoid [0,1]:
    # divide by 2, add 0.5 to constant term
    if polys:
        last = polys[-1]
        for j in range(len(last)):
            last[j] /= 2.0
        last[0] += 0.5

    # Flatten back and cache
    result = []
    for poly in polys:
        result.extend(poly)

    with _minimax_cache_lock:
        _minimax_cache[key] = list(result)

    return result


# =========================================================================
# PlainTensor (moved from backend/python/encoder.py)
# =========================================================================


class PlainTensor:
    """Compile-time plaintext wrapper using handle-based FFI."""

    def __init__(
        self,
        context: CompilationContext,
        ptxt_ids: GoHandle | list[GoHandle],
        shape: torch.Size,
        on_shape: torch.Size | None = None,
    ):
        self.context = context
        self.backend: CompilerBackend = context.backend
        self.encoder: NewEncoder = context.encoder
        self.evaluator: Any = getattr(context, "evaluator", None)
        self.ids: list[GoHandle] = [ptxt_ids] if isinstance(ptxt_ids, GoHandle) else ptxt_ids
        self.shape = shape
        self.on_shape = on_shape or shape

    def close(self) -> None:
        """Release all Go handles. Idempotent."""
        for h in self.ids:
            try:
                h.close()
            except Exception:
                pass

    def __enter__(self) -> PlainTensor:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.ids)


# =========================================================================
# NewEncoder (moved from backend/python/encoder.py)
# =========================================================================


class NewEncoder:
    def __init__(self, context: CompilationContext):
        self.context = context
        self.params: NewParameters = context.params
        self.backend: CompilerBackend = context.backend
        self.setup_encoder()

    def setup_encoder(self) -> None:
        self.backend.NewEncoder()

    def encode(
        self,
        values: list[float] | torch.Tensor,
        level: int | None = None,
        scale: int | None = None,
    ) -> PlainTensor:
        if isinstance(values, list):
            values = torch.tensor(values)
        elif not isinstance(values, torch.Tensor):
            raise TypeError(
                f"Expected 'values' passed to encode() to be a either a list "
                f"or a torch.Tensor, but got {type(values)}."
            )

        if level is None:
            level = self.params.get_max_level()
        if scale is None:
            scale = self.params.get_default_scale()

        num_slots = self.params.get_slots()
        num_elements = values.numel()

        values = values.cpu()
        pad_length = (-num_elements) % num_slots
        vector = torch.zeros(num_elements + pad_length)
        vector[:num_elements] = values.flatten()
        num_plaintexts = len(vector) // num_slots

        plaintext_ids: list[GoHandle] = []
        for i in range(num_plaintexts):
            to_encode = vector[i * num_slots : (i + 1) * num_slots].tolist()
            plaintext_id = self.backend.Encode(to_encode, level, scale)
            plaintext_ids.append(plaintext_id)

        return PlainTensor(self.context, plaintext_ids, values.shape)

    def decode(self, plaintensor: PlainTensor) -> torch.Tensor:
        values: list[float] = []
        for plaintext_id in plaintensor.ids:
            values.extend(self.backend.Decode(plaintext_id))

        tensor = torch.tensor(values)[: plaintensor.on_shape.numel()]
        return tensor.reshape(plaintensor.on_shape)

    def get_moduli_chain(self) -> list[int]:
        return self.backend.GetModuliChain()

    def get_aux_moduli_chain(self) -> list[int]:
        return self.backend.GetAuxModuliChain()


# =========================================================================
# PolynomialGenerator (moved from backend/python/poly_evaluator.py)
# =========================================================================


class PolynomialGenerator:
    """Compile-time polynomial generation via Lattigo FFI."""

    def __init__(self, backend: CompilerBackend):
        self.backend = backend

    def generate_monomial(self, coeffs: list[float] | torch.Tensor | np.ndarray) -> GoHandle:
        if isinstance(coeffs, (torch.Tensor, np.ndarray)):
            coeffs = coeffs.tolist()
        return self.backend.GenerateMonomial(coeffs[::-1])

    def generate_chebyshev(self, coeffs: list[float] | torch.Tensor | np.ndarray) -> GoHandle:
        if isinstance(coeffs, (torch.Tensor, np.ndarray)):
            coeffs = coeffs.tolist()
        return self.backend.GenerateChebyshev(coeffs)

    def generate_minimax_sign_coeffs(
        self,
        degrees: int | list[int],
        prec: int = 128,
        logalpha: int = 12,
        logerr: int = 12,
        debug: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        degrees = [degrees] if isinstance(degrees, int) else list(degrees)

        degrees = [d for d in degrees if d != 0]
        if len(degrees) == 0:
            raise ValueError(
                "At least one non-zero degree polynomial must be provided to "
                "generate_minimax_sign_coeffs(). "
            )

        coeffs_flat = self.backend.GenerateMinimaxSignCoeffs(
            degrees, prec, logalpha, logerr, int(debug)
        )

        coeffs_tensor = torch.tensor(coeffs_flat)
        splits = [degree + 1 for degree in degrees]
        return torch.split(coeffs_tensor, splits)


# =========================================================================
# CompilationContext — typed context for Module.fit() / Module.compile()
# =========================================================================


@dataclass
class CompilationContext:
    """Typed context passed to Module.fit() and Module.compile() during compilation."""

    backend: CompilerBackend
    params: NewParameters
    encoder: NewEncoder
    poly_evaluator: PolynomialGenerator
    margin: int
    config: CompilerConfig
