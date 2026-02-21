"""New v2 parameter dataclasses for Orion.

CKKSParams holds CKKS cryptographic parameters.
CompilerConfig holds compilation settings.
Both are frozen (immutable) and validated at construction.
"""

import json
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CKKSParams:
    """CKKS scheme parameters — immutable after construction.

    These define the cryptographic parameter set for the CKKS scheme.
    Both Client and Compiler need identical params to interoperate.
    """

    logn: int
    logq: tuple[int, ...]
    logp: tuple[int, ...]
    logscale: int
    h: int = 192
    ring_type: Literal["conjugate_invariant", "standard"] = "conjugate_invariant"
    boot_logp: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.logn <= 0:
            raise ValueError(f"logn must be positive, got {self.logn}")
        if not self.logq:
            raise ValueError("logq must be non-empty")
        if not self.logp:
            raise ValueError("logp must be non-empty")
        if len(self.logp) > len(self.logq):
            raise ValueError(
                f"logp length ({len(self.logp)}) cannot exceed "
                f"logq length ({len(self.logq)})"
            )
        valid_ring_types = {"conjugate_invariant", "standard"}
        if self.ring_type not in valid_ring_types:
            raise ValueError(
                f"ring_type must be one of {valid_ring_types}, "
                f"got '{self.ring_type}'"
            )
        # Coerce list inputs to tuples for frozen dataclass
        if isinstance(self.logq, list):
            object.__setattr__(self, "logq", tuple(self.logq))
        if isinstance(self.logp, list):
            object.__setattr__(self, "logp", tuple(self.logp))
        if isinstance(self.boot_logp, list):
            object.__setattr__(self, "boot_logp", tuple(self.boot_logp))

    @property
    def max_level(self) -> int:
        """Maximum multiplicative depth = len(logq) - 1."""
        return len(self.logq) - 1

    @property
    def max_slots(self) -> int:
        """Number of plaintext slots.

        For conjugate_invariant: 2^logn (full ring).
        For standard: 2^(logn-1) (half ring, complex packing).
        """
        if self.ring_type == "conjugate_invariant":
            return 1 << self.logn
        return 1 << (self.logn - 1)

    @property
    def ring_degree(self) -> int:
        """Ring degree N = 2^logn."""
        return 1 << self.logn

    def to_bridge_json(self) -> str:
        """Serialize to JSON for the Go bridge."""
        d = {
            "logn": self.logn,
            "logq": list(self.logq),
            "logp": list(self.logp),
            "logscale": self.logscale,
            "h": self.h,
            "ring_type": self.ring_type,
        }
        if self.boot_logp is not None:
            d["boot_logp"] = list(self.boot_logp)
        return json.dumps(d)


@dataclass(frozen=True)
class CompilerConfig:
    """Compilation settings — immutable after construction."""

    margin: int = 2
    embedding_method: Literal["hybrid", "square"] = "hybrid"
    fuse_modules: bool = True

    def __post_init__(self):
        valid_methods = {"hybrid", "square"}
        if self.embedding_method not in valid_methods:
            raise ValueError(
                f"embedding_method must be one of {valid_methods}, "
                f"got '{self.embedding_method}'"
            )
