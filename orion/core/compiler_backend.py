"""Compile-time backend adapter wrapping orionclient FFI.

Provides the same interface as the old backend/python/ wrappers
(NewParameters, NewEncoder, PolynomialGenerator) but backed by the
orionclient bridge shared library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch
import numpy as np

from orion.backend.orionclient import ffi

if TYPE_CHECKING:
    from orion.params import CKKSParams, CompilerConfig


# =========================================================================
# NewParameters (moved from backend/python/parameters.py)
# =========================================================================

@dataclass
class CKKSParameters:
    logn: int
    logq: List[int]
    logp: List[int]
    logscale: int = field(default=None)
    h: int = 192
    ringtype: str = "standard"
    boot_logp: List[int] = field(default=None)

    def __post_init__(self):
        if self.logq and self.logp and len(self.logp) > len(self.logq):
            raise ValueError(
                f"Invalid parameters: The length of logp ({len(self.logp)}) "
                f"cannot exceed the length of logq ({len(self.logq)})."
            )
        valid_ringtypes = {"standard", "conjugateinvariant"}
        ring = self.ringtype.lower()
        if ring not in valid_ringtypes:
            raise ValueError(
                f"Invalid ringtype: {self.ringtype}. Only 'Standard' or "
                f"'ConjugateInvariant' ring types are supported."
            )
        self.logscale = self.logscale or self.logq[-1]
        self.boot_logp = self.boot_logp or self.logp
        self.logslots = (
            self.logn - 1 if self.ringtype.lower() == "standard"
            else self.logn
        )


@dataclass
class OrionParameters:
    margin: int = 2
    fuse_modules: bool = True
    debug: bool = True
    embedding_method: Literal["hybrid", "square"] = "hybrid"
    backend: Literal["lattigo", "openfhe", "heaan"] = "lattigo"


@dataclass
class NewParameters:
    params_json: dict
    ckks_params: CKKSParameters = field(init=False)
    orion_params: OrionParameters = field(init=False)

    def __post_init__(self):
        params = self.params_json
        ckks_params = {
            k.lower(): v for k, v in params.get("ckks_params", {}).items()
        }
        boot_params = {
            k.lower(): v for k, v in params.get("boot_params", {}).items()
        }
        orion_params = {
            k.lower(): v for k, v in params.get("orion", {}).items()
        }
        self.ckks_params = CKKSParameters(
            **ckks_params, boot_logp=boot_params.get("logp")
        )
        self.orion_params = OrionParameters(**orion_params)

    def get_logn(self):
        return self.ckks_params.logn

    def get_margin(self):
        return self.orion_params.margin

    def get_fuse_modules(self):
        return self.orion_params.fuse_modules

    def get_debug_status(self):
        return self.orion_params.debug

    def get_backend(self):
        return self.orion_params.backend.lower()

    def get_logq(self):
        return self.ckks_params.logq

    def get_logp(self):
        return self.ckks_params.logp

    def get_logscale(self):
        return self.ckks_params.logscale

    def get_default_scale(self):
        return 1 << self.ckks_params.logscale

    def get_hamming_weight(self):
        return self.ckks_params.h

    def get_ringtype(self):
        return self.ckks_params.ringtype.lower()

    def get_max_level(self):
        return len(self.ckks_params.logq) - 1

    def get_slots(self):
        return int(1 << self.ckks_params.logslots)

    def get_ring_degree(self):
        return int(1 << self.ckks_params.logn)

    def get_embedding_method(self):
        return self.orion_params.embedding_method.lower()

    def get_boot_logp(self):
        return self.ckks_params.boot_logp

    @classmethod
    def from_ckks_params(
        cls, ckks_params: CKKSParams, config: CompilerConfig | None = None
    ) -> NewParameters:
        from orion.params import CompilerConfig as _CC

        if config is None:
            config = _CC()
        ring_type_map = {
            "conjugate_invariant": "ConjugateInvariant",
            "standard": "Standard",
        }
        legacy_ring_type = ring_type_map[ckks_params.ring_type]
        params_json = {
            "ckks_params": {
                "LogN": ckks_params.logn,
                "LogQ": list(ckks_params.logq),
                "LogP": list(ckks_params.logp),
                "LogScale": ckks_params.logscale,
                "H": ckks_params.h,
                "RingType": legacy_ring_type,
            },
            "boot_params": {},
            "orion": {
                "margin": config.margin,
                "embedding_method": config.embedding_method,
                "fuse_modules": config.fuse_modules,
                "backend": "lattigo",
                "debug": False,
            },
        }
        if ckks_params.boot_logp is not None:
            params_json["boot_params"]["LogP"] = list(ckks_params.boot_logp)
        return cls(params_json=params_json)


# =========================================================================
# CompilerBackend - wraps orionclient FFI for compile-time operations
# =========================================================================

class CompilerBackend:
    """Adapter providing the LattigoLibrary interface via orionclient FFI.

    The compiler and its helpers (NewEncoder, PolynomialGenerator)
    call self.backend.XXX(). This class provides those methods by
    delegating to the orionclient FFI.
    """

    def __init__(self):
        self._client_h = None
        self._params_json_str = None

    def setup_bindings(self, params: NewParameters):
        """Initialize Go backend with the given parameters.

        Creates a Client internally (generates a throwaway key pair, but
        the compiler only uses the encoder and parameter queries).
        """
        # Build the JSON that orionclient expects
        p = params.ckks_params
        ring_map = {"standard": "standard", "conjugateinvariant": "conjugate_invariant"}
        ring_type = ring_map.get(p.ringtype.lower(), "conjugate_invariant")

        params_dict = {
            "logn": p.logn,
            "logq": list(p.logq),
            "logp": list(p.logp),
            "logscale": p.logscale,
            "h": p.h,
            "ring_type": ring_type,
        }
        if p.boot_logp:
            params_dict["boot_logp"] = list(p.boot_logp)

        self._params_json_str = json.dumps(params_dict)
        self._client_h = ffi.new_client(self._params_json_str)

    # -- Encoder operations --

    def NewEncoder(self):
        """No-op. The encoder is part of the Client."""
        pass

    def Encode(self, values, level, scale):
        """Encode values into a plaintext. Returns a handle (int)."""
        if not isinstance(values, list):
            values = list(values)
        return ffi.client_encode(self._client_h, values, level, scale)

    def Decode(self, pt_h):
        """Decode a plaintext handle to float values."""
        return ffi.client_decode(self._client_h, pt_h)

    def DeletePlaintext(self, pt_h):
        """Delete a plaintext handle."""
        pt_h.close()

    # -- Parameter queries --

    def GetMaxSlots(self):
        return ffi.client_max_slots(self._client_h)

    def GetGaloisElement(self, rotation):
        return ffi.client_galois_element(self._client_h, rotation)

    def GetModuliChain(self):
        return ffi.client_moduli_chain(self._client_h)

    def GetAuxModuliChain(self):
        return ffi.client_aux_moduli_chain(self._client_h)

    # -- Polynomial operations --

    def GenerateMonomial(self, coeffs):
        """Generate a monomial polynomial. Returns a handle (int)."""
        if not isinstance(coeffs, list):
            coeffs = list(coeffs)
        return ffi.generate_polynomial_monomial(coeffs)

    def GenerateChebyshev(self, coeffs):
        """Generate a Chebyshev polynomial. Returns a handle (int)."""
        if not isinstance(coeffs, list):
            coeffs = list(coeffs)
        return ffi.generate_polynomial_chebyshev(coeffs)

    def GenerateMinimaxSignCoeffs(self, degrees, prec, logalpha, logerr, debug):
        """Generate minimax sign polynomial coefficients."""
        return ffi.generate_minimax_sign_coeffs(degrees, prec, logalpha, logerr, debug)

    # -- Lifecycle --

    def DeleteScheme(self):
        """Release the Go client."""
        if self._client_h:
            ffi.client_close(self._client_h)   # step 1: zeros SK in Go
            self._client_h.close()              # step 2: DeleteHandle (frees cgo slot)
            self._client_h = None

    def close(self):
        """Release the Go backend. Idempotent."""
        self.DeleteScheme()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# =========================================================================
# PlainTensor (moved from backend/python/encoder.py)
# =========================================================================

class PlainTensor:
    """Compile-time plaintext wrapper using handle-based FFI."""

    def __init__(self, context, ptxt_ids, shape, on_shape=None):
        self.context = context
        self.backend = context.backend
        self.encoder = context.encoder
        self.evaluator = getattr(context, "evaluator", None)
        self.ids = [ptxt_ids] if isinstance(ptxt_ids, ffi.GoHandle) else ptxt_ids
        self.shape = shape
        self.on_shape = on_shape or shape

    def close(self):
        """Release all Go handles. Idempotent."""
        for h in self.ids:
            try:
                h.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return len(self.ids)


# =========================================================================
# NewEncoder (moved from backend/python/encoder.py)
# =========================================================================

class NewEncoder:
    def __init__(self, context):
        self.context = context
        self.params = context.params
        self.backend = context.backend
        self.setup_encoder()

    def setup_encoder(self):
        self.backend.NewEncoder()

    def encode(self, values, level=None, scale=None):
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

        plaintext_ids = []
        for i in range(num_plaintexts):
            to_encode = vector[i * num_slots:(i + 1) * num_slots].tolist()
            plaintext_id = self.backend.Encode(to_encode, level, scale)
            plaintext_ids.append(plaintext_id)

        return PlainTensor(self.context, plaintext_ids, values.shape)

    def decode(self, plaintensor: PlainTensor):
        values = []
        for plaintext_id in plaintensor.ids:
            values.extend(self.backend.Decode(plaintext_id))

        values = torch.tensor(values)[:plaintensor.on_shape.numel()]
        return values.reshape(plaintensor.on_shape)

    def get_moduli_chain(self):
        return self.backend.GetModuliChain()

    def get_aux_moduli_chain(self):
        return self.backend.GetAuxModuliChain()


# =========================================================================
# PolynomialGenerator (moved from backend/python/poly_evaluator.py)
# =========================================================================

class PolynomialGenerator:
    """Compile-time polynomial generation via orionclient FFI."""

    def __init__(self, backend):
        self.backend = backend

    def generate_monomial(self, coeffs):
        if isinstance(coeffs, (torch.Tensor, np.ndarray)):
            coeffs = coeffs.tolist()
        return self.backend.GenerateMonomial(coeffs[::-1])

    def generate_chebyshev(self, coeffs):
        if isinstance(coeffs, (torch.Tensor, np.ndarray)):
            coeffs = coeffs.tolist()
        return self.backend.GenerateChebyshev(coeffs)

    def generate_minimax_sign_coeffs(self, degrees, prec=128, logalpha=12,
                                     logerr=12, debug=False):
        if isinstance(degrees, int):
            degrees = [degrees]
        else:
            degrees = list(degrees)

        degrees = [d for d in degrees if d != 0]
        if len(degrees) == 0:
            raise ValueError(
                "At least one non-zero degree polynomial must be provided to "
                "generate_minimax_sign_coeffs(). "
            )

        coeffs_flat = self.backend.GenerateMinimaxSignCoeffs(
            degrees, prec, logalpha, logerr, int(debug)
        )

        coeffs_flat = torch.tensor(coeffs_flat)
        splits = [degree + 1 for degree in degrees]
        return torch.split(coeffs_flat, splits)


