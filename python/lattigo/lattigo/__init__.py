"""lattigo — Python bindings for Lattigo CKKS primitives.

All handle-owning classes (Parameters, Encoder, KeyGenerator, Encryptor,
Decryptor, etc.) are NOT thread-safe. Do not share instances across threads.

Usage:
    from lattigo import ckks, rlwe
    from lattigo.ckks import Parameters, Encoder
    from lattigo.rlwe import KeyGenerator, SecretKey, Encryptor, Decryptor, ...
"""

from . import ckks, rlwe
from .errors import FFIError, HandleClosedError, LatticeError

__all__ = ["FFIError", "HandleClosedError", "LatticeError", "ckks", "rlwe"]

__version__ = "6.2.2"
