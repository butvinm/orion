"""RLWE types: keys, ciphertext, plaintext, encryptor, decryptor, key generator.

Wraps raw Lattigo rlwe types via the bridge FFI.
"""

from __future__ import annotations

from . import ffi
from .ckks import Parameters
from .gohandle import GoHandle


# =========================================================================
# Key types
# =========================================================================


class SecretKey:
    """Wraps Lattigo's rlwe.SecretKey."""

    def __init__(self, handle: GoHandle):
        self._handle = handle

    def marshal_binary(self) -> bytes:
        return ffi.secret_key_marshal(self._handle)

    @classmethod
    def unmarshal_binary(cls, data: bytes) -> SecretKey:
        return cls(ffi.secret_key_unmarshal(data))

    def close(self):
        self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class PublicKey:
    """Wraps Lattigo's rlwe.PublicKey."""

    def __init__(self, handle: GoHandle):
        self._handle = handle

    def marshal_binary(self) -> bytes:
        return ffi.public_key_marshal(self._handle)

    @classmethod
    def unmarshal_binary(cls, data: bytes) -> PublicKey:
        return cls(ffi.public_key_unmarshal(data))

    def close(self):
        self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RelinearizationKey:
    """Wraps Lattigo's rlwe.RelinearizationKey."""

    def __init__(self, handle: GoHandle):
        self._handle = handle

    def marshal_binary(self) -> bytes:
        return ffi.relinearization_key_marshal(self._handle)

    @classmethod
    def unmarshal_binary(cls, data: bytes) -> RelinearizationKey:
        return cls(ffi.relinearization_key_unmarshal(data))

    def close(self):
        self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class GaloisKey:
    """Wraps Lattigo's rlwe.GaloisKey."""

    def __init__(self, handle: GoHandle):
        self._handle = handle

    def marshal_binary(self) -> bytes:
        return ffi.galois_key_marshal(self._handle)

    @classmethod
    def unmarshal_binary(cls, data: bytes) -> GaloisKey:
        return cls(ffi.galois_key_unmarshal(data))

    def close(self):
        self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# =========================================================================
# Ciphertext / Plaintext
# =========================================================================


class Ciphertext:
    """Wraps a raw Lattigo rlwe.Ciphertext (single ciphertext, no shape)."""

    def __init__(self, handle: GoHandle):
        self._handle = handle

    def level(self) -> int:
        return ffi.rlwe_ciphertext_level(self._handle)

    def marshal_binary(self) -> bytes:
        return ffi.rlwe_ciphertext_marshal(self._handle)

    @classmethod
    def unmarshal_binary(cls, data: bytes) -> Ciphertext:
        return cls(ffi.rlwe_ciphertext_unmarshal(data))

    def close(self):
        self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Plaintext:
    """Wraps a raw Lattigo rlwe.Plaintext."""

    def __init__(self, handle: GoHandle):
        self._handle = handle

    def level(self) -> int:
        return ffi.rlwe_plaintext_level(self._handle)

    def marshal_binary(self) -> bytes:
        return ffi.rlwe_plaintext_marshal(self._handle)

    @classmethod
    def unmarshal_binary(cls, data: bytes) -> Plaintext:
        return cls(ffi.rlwe_plaintext_unmarshal(data))

    def close(self):
        self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# =========================================================================
# KeyGenerator
# =========================================================================


class KeyGenerator:
    """Wraps Lattigo's rlwe.KeyGenerator."""

    def __init__(self, handle: GoHandle):
        self._handle = handle

    @classmethod
    def new(cls, params: Parameters) -> KeyGenerator:
        handle = ffi.new_key_generator(params._h)
        return cls(handle)

    def gen_secret_key(self) -> SecretKey:
        return SecretKey(ffi.keygen_gen_secret_key(self._handle))

    def gen_public_key(self, sk: SecretKey) -> PublicKey:
        return PublicKey(ffi.keygen_gen_public_key(self._handle, sk._handle))

    def gen_relinearization_key(self, sk: SecretKey) -> RelinearizationKey:
        return RelinearizationKey(
            ffi.keygen_gen_relinearization_key(self._handle, sk._handle)
        )

    def gen_galois_key(self, sk: SecretKey, galois_element: int) -> GaloisKey:
        return GaloisKey(
            ffi.keygen_gen_galois_key(self._handle, sk._handle, galois_element)
        )

    def close(self):
        self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# =========================================================================
# Encryptor / Decryptor
# =========================================================================


class Encryptor:
    """Wraps Lattigo's rlwe.Encryptor (public-key encryption)."""

    def __init__(self, handle: GoHandle, params: Parameters):
        self._handle = handle
        self._params = params

    @classmethod
    def new(cls, params: Parameters, pk: PublicKey) -> Encryptor:
        handle = ffi.new_ckks_encryptor(params._h, pk._handle)
        return cls(handle, params)

    def encrypt_new(self, pt: Plaintext) -> Ciphertext:
        ct_h = ffi.encryptor_encrypt_new(
            self._handle, pt._handle, self._params._h,
        )
        return Ciphertext(ct_h)

    def close(self):
        self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Decryptor:
    """Wraps Lattigo's rlwe.Decryptor."""

    def __init__(self, handle: GoHandle, params: Parameters):
        self._handle = handle
        self._params = params

    @classmethod
    def new(cls, params: Parameters, sk: SecretKey) -> Decryptor:
        handle = ffi.new_ckks_decryptor(params._h, sk._handle)
        return cls(handle, params)

    def decrypt_new(self, ct: Ciphertext) -> Plaintext:
        pt_h = ffi.decryptor_decrypt_new(
            self._handle, ct._handle, self._params._h,
        )
        return Plaintext(pt_h)

    def close(self):
        self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# =========================================================================
# MemEvaluationKeySet
# =========================================================================


class MemEvaluationKeySet:
    """Wraps Lattigo's rlwe.MemEvaluationKeySet."""

    def __init__(self, handle: GoHandle):
        self._handle = handle

    @classmethod
    def new(
        cls,
        rlk: RelinearizationKey | None = None,
        galois_keys: list[GaloisKey] | None = None,
    ) -> MemEvaluationKeySet:
        handle = ffi.new_mem_eval_key_set(
            rlk._handle if rlk else None,
            [gk._handle for gk in (galois_keys or [])],
        )
        return cls(handle)

    def marshal_binary(self) -> bytes:
        return ffi.mem_eval_key_set_marshal(self._handle)

    @classmethod
    def unmarshal_binary(cls, data: bytes) -> MemEvaluationKeySet:
        return cls(ffi.mem_eval_key_set_unmarshal(data))

    def close(self):
        self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
