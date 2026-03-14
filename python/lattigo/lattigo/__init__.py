"""lattigo — Python bindings for Lattigo CKKS primitives.

Usage:
    from lattigo import ckks, rlwe
    from lattigo.ckks import Parameters, Encoder
    from lattigo.rlwe import KeyGenerator, SecretKey, Encryptor, Decryptor, ...
"""

from . import ckks, rlwe
from .errors import FFIError, HandleClosedError, LatticeError

__all__ = ["FFIError", "HandleClosedError", "LatticeError", "ckks", "rlwe"]
