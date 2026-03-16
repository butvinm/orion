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

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> SecretKey:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
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

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> PublicKey:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class RelinearizationKey:
    """Wraps Lattigo's rlwe.RelinearizationKey."""

    def __init__(self, handle: GoHandle):
        self._handle = handle

    def marshal_binary(self) -> bytes:
        return ffi.relin_key_marshal(self._handle)

    @classmethod
    def unmarshal_binary(cls, data: bytes) -> RelinearizationKey:
        return cls(ffi.relin_key_unmarshal(data))

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> RelinearizationKey:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
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

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> GaloisKey:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
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

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Ciphertext:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
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

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Plaintext:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# =========================================================================
# KeyGenerator
# =========================================================================


class KeyGenerator:
    """Wraps Lattigo's rlwe.KeyGenerator."""

    def __init__(self, params: Parameters):
        self._handle = ffi.new_key_generator(params._h)

    @classmethod
    def _from_handle(cls, handle: GoHandle) -> KeyGenerator:
        """Wrap an existing Go handle (internal use)."""
        obj = object.__new__(cls)
        obj._handle = handle
        return obj

    def gen_secret_key(self) -> SecretKey:
        return SecretKey(ffi.keygen_gen_secret_key(self._handle))

    def gen_public_key(self, sk: SecretKey) -> PublicKey:
        return PublicKey(ffi.keygen_gen_public_key(self._handle, sk._handle))

    def gen_relin_key(self, sk: SecretKey) -> RelinearizationKey:
        return RelinearizationKey(ffi.keygen_gen_relin_key(self._handle, sk._handle))

    def gen_galois_key(self, sk: SecretKey, galois_element: int) -> GaloisKey:
        return GaloisKey(ffi.keygen_gen_galois_key(self._handle, sk._handle, galois_element))

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> KeyGenerator:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# =========================================================================
# Encryptor / Decryptor
# =========================================================================


class Encryptor:
    """Wraps Lattigo's rlwe.Encryptor (public-key encryption)."""

    def __init__(self, params: Parameters, pk: PublicKey):
        self._handle = ffi.new_encryptor(params._h, pk._handle)

    @classmethod
    def _from_handle(cls, handle: GoHandle) -> Encryptor:
        """Wrap an existing Go handle (internal use)."""
        obj = object.__new__(cls)
        obj._handle = handle
        return obj

    def encrypt_new(self, pt: Plaintext) -> Ciphertext:
        ct_h = ffi.encryptor_encrypt_new(self._handle, pt._handle)
        return Ciphertext(ct_h)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Encryptor:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Decryptor:
    """Wraps Lattigo's rlwe.Decryptor."""

    def __init__(self, params: Parameters, sk: SecretKey):
        self._handle = ffi.new_decryptor(params._h, sk._handle)

    @classmethod
    def _from_handle(cls, handle: GoHandle) -> Decryptor:
        """Wrap an existing Go handle (internal use)."""
        obj = object.__new__(cls)
        obj._handle = handle
        return obj

    def decrypt_new(self, ct: Ciphertext) -> Plaintext:
        pt_h = ffi.decryptor_decrypt_new(self._handle, ct._handle)
        return Plaintext(pt_h)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Decryptor:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# =========================================================================
# MemEvaluationKeySet
# =========================================================================


class MemEvaluationKeySet:
    """Wraps Lattigo's rlwe.MemEvaluationKeySet."""

    def __init__(
        self,
        rlk: RelinearizationKey | None = None,
        galois_keys: list[GaloisKey] | None = None,
    ):
        self._handle = ffi.new_mem_eval_key_set(
            rlk._handle if rlk else None,
            [gk._handle for gk in (galois_keys or [])],
        )

    @classmethod
    def _from_handle(cls, handle: GoHandle) -> MemEvaluationKeySet:
        """Wrap an existing Go handle (internal use)."""
        obj = object.__new__(cls)
        obj._handle = handle
        return obj

    def marshal_binary(self) -> bytes:
        return ffi.mem_eval_key_set_marshal(self._handle)

    @classmethod
    def unmarshal_binary(cls, data: bytes) -> MemEvaluationKeySet:
        return cls._from_handle(ffi.mem_eval_key_set_unmarshal(data))

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> MemEvaluationKeySet:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# =========================================================================
# Bootstrap
# =========================================================================


class BootstrapEvalKeys:
    """Wraps Lattigo's bootstrapping.EvaluationKeys (serializable bootstrap keys)."""

    def __init__(self, handle: GoHandle):
        self._handle = handle

    def marshal_binary(self) -> bytes:
        return ffi.bootstrap_eval_keys_marshal(self._handle)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> BootstrapEvalKeys:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class BootstrapParams:
    """Wraps Lattigo's bootstrapping parameters.

    Used to generate the evaluation keys required for bootstrapping
    (ciphertext refresh) during FHE inference.

    Example::

        btp = BootstrapParams(params, logn=14, logp=[61]*6, h=192, log_slots=7)
        evk, btp_keys = btp.gen_eval_keys(sk)
        btp_keys_bytes = btp_keys.marshal_binary()
    """

    def __init__(
        self,
        params: Parameters,
        *,
        logn: int = 0,
        logp: list[int] | None = None,
        h: int = 0,
        log_slots: int = 0,
    ):
        self._handle = ffi.new_bootstrap_params(
            params._h, logn=logn, logp=logp, h=h, log_slots=log_slots,
        )

    def gen_eval_keys(
        self, sk: SecretKey,
    ) -> tuple[MemEvaluationKeySet, BootstrapEvalKeys]:
        """Generate bootstrap evaluation keys.

        Returns:
            A tuple of (evk, btp_keys) where:
            - evk: MemEvaluationKeySet for the evaluator
            - btp_keys: BootstrapEvalKeys for serialization
        """
        evk_h, btp_evk_h = ffi.bootstrap_params_gen_eval_keys(
            self._handle, sk._handle,
        )
        return MemEvaluationKeySet._from_handle(evk_h), BootstrapEvalKeys(btp_evk_h)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> BootstrapParams:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
