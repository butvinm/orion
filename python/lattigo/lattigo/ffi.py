"""Low-level ctypes FFI bindings for Lattigo primitive bridge exports.

Wraps the new bridge functions (NewCKKSParams, NewKeyGenerator, etc.)
that work with raw Lattigo types, not Orion wrapper types.
"""

import ctypes

from .gohandle import GoHandle, get_lib

_uintptr = ctypes.c_size_t
_errout = ctypes.POINTER(ctypes.c_char_p)

_prototypes_set = False


def _check_err(err_ptr):
    """Check errOut and raise if non-NULL, then free the C string."""
    if err_ptr and err_ptr.value:
        msg = err_ptr.value.decode("utf-8")
        get_lib().FreeCArray(ctypes.cast(err_ptr, ctypes.c_void_p))
        raise RuntimeError(msg)


def _make_errout():
    return ctypes.c_char_p(None)


def _doubles_ptr(values):
    n = len(values)
    arr = (ctypes.c_double * n)(*values)
    return arr, ctypes.c_int(n)


def _bytes_ptr(data):
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    return ctypes.cast(buf, ctypes.c_void_p), ctypes.c_ulong(len(data))


def _setup_prototypes(lib):
    """Declare C function prototypes for Lattigo primitive exports."""
    # --- CKKS Parameters ---
    lib.NewCKKSParams.argtypes = [ctypes.c_char_p, _errout]
    lib.NewCKKSParams.restype = _uintptr

    lib.CKKSParamsMaxSlots.argtypes = [_uintptr]
    lib.CKKSParamsMaxSlots.restype = ctypes.c_int

    lib.CKKSParamsMaxLevel.argtypes = [_uintptr]
    lib.CKKSParamsMaxLevel.restype = ctypes.c_int

    lib.CKKSParamsDefaultScale.argtypes = [_uintptr]
    lib.CKKSParamsDefaultScale.restype = ctypes.c_ulonglong

    lib.CKKSParamsGaloisElement.argtypes = [_uintptr, ctypes.c_int]
    lib.CKKSParamsGaloisElement.restype = ctypes.c_ulonglong

    lib.CKKSParamsModuliChain.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_int), _errout]
    lib.CKKSParamsModuliChain.restype = ctypes.POINTER(ctypes.c_ulonglong)

    lib.CKKSParamsAuxModuliChain.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_int), _errout]
    lib.CKKSParamsAuxModuliChain.restype = ctypes.POINTER(ctypes.c_ulonglong)

    # --- KeyGenerator ---
    lib.NewKeyGenerator.argtypes = [_uintptr, _errout]
    lib.NewKeyGenerator.restype = _uintptr

    lib.KeyGenGenSecretKey.argtypes = [_uintptr, _errout]
    lib.KeyGenGenSecretKey.restype = _uintptr

    lib.KeyGenGenPublicKey.argtypes = [_uintptr, _uintptr, _errout]
    lib.KeyGenGenPublicKey.restype = _uintptr

    lib.KeyGenGenRelinearizationKey.argtypes = [_uintptr, _uintptr, _errout]
    lib.KeyGenGenRelinearizationKey.restype = _uintptr

    lib.KeyGenGenGaloisKey.argtypes = [_uintptr, _uintptr, ctypes.c_ulonglong, _errout]
    lib.KeyGenGenGaloisKey.restype = _uintptr

    # --- CKKS Encoder ---
    lib.NewCKKSEncoder.argtypes = [_uintptr, _errout]
    lib.NewCKKSEncoder.restype = _uintptr

    lib.CKKSEncoderEncode.argtypes = [
        _uintptr, _uintptr,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_int, ctypes.c_ulonglong, _errout,
    ]
    lib.CKKSEncoderEncode.restype = _uintptr

    lib.CKKSEncoderDecode.argtypes = [
        _uintptr, _uintptr, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), _errout,
    ]
    lib.CKKSEncoderDecode.restype = ctypes.POINTER(ctypes.c_double)

    # --- Encryptor ---
    lib.NewCKKSEncryptor.argtypes = [_uintptr, _uintptr, _errout]
    lib.NewCKKSEncryptor.restype = _uintptr

    lib.EncryptorEncryptNew.argtypes = [_uintptr, _uintptr, _uintptr, _errout]
    lib.EncryptorEncryptNew.restype = _uintptr

    # --- Decryptor ---
    lib.NewCKKSDecryptor.argtypes = [_uintptr, _uintptr, _errout]
    lib.NewCKKSDecryptor.restype = _uintptr

    lib.DecryptorDecryptNew.argtypes = [_uintptr, _uintptr, _uintptr, _errout]
    lib.DecryptorDecryptNew.restype = _uintptr

    # --- SecretKey serialization ---
    lib.SecretKeyMarshal.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_ulong), _errout]
    lib.SecretKeyMarshal.restype = ctypes.c_void_p

    lib.SecretKeyUnmarshal.argtypes = [ctypes.c_void_p, ctypes.c_ulong, _errout]
    lib.SecretKeyUnmarshal.restype = _uintptr

    # --- PublicKey serialization ---
    lib.PublicKeyMarshal.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_ulong), _errout]
    lib.PublicKeyMarshal.restype = ctypes.c_void_p

    lib.PublicKeyUnmarshal.argtypes = [ctypes.c_void_p, ctypes.c_ulong, _errout]
    lib.PublicKeyUnmarshal.restype = _uintptr

    # --- RelinearizationKey serialization ---
    lib.RelinearizationKeyMarshal.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_ulong), _errout]
    lib.RelinearizationKeyMarshal.restype = ctypes.c_void_p

    lib.RelinearizationKeyUnmarshal.argtypes = [ctypes.c_void_p, ctypes.c_ulong, _errout]
    lib.RelinearizationKeyUnmarshal.restype = _uintptr

    # --- GaloisKey serialization ---
    lib.GaloisKeyMarshal.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_ulong), _errout]
    lib.GaloisKeyMarshal.restype = ctypes.c_void_p

    lib.GaloisKeyUnmarshal.argtypes = [ctypes.c_void_p, ctypes.c_ulong, _errout]
    lib.GaloisKeyUnmarshal.restype = _uintptr

    # --- rlwe.Ciphertext serialization ---
    lib.RLWECiphertextMarshal.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_ulong), _errout]
    lib.RLWECiphertextMarshal.restype = ctypes.c_void_p

    lib.RLWECiphertextUnmarshal.argtypes = [ctypes.c_void_p, ctypes.c_ulong, _errout]
    lib.RLWECiphertextUnmarshal.restype = _uintptr

    lib.RLWECiphertextLevel.argtypes = [_uintptr]
    lib.RLWECiphertextLevel.restype = ctypes.c_int

    # --- rlwe.Plaintext serialization ---
    lib.RLWEPlaintextMarshal.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_ulong), _errout]
    lib.RLWEPlaintextMarshal.restype = ctypes.c_void_p

    lib.RLWEPlaintextUnmarshal.argtypes = [ctypes.c_void_p, ctypes.c_ulong, _errout]
    lib.RLWEPlaintextUnmarshal.restype = _uintptr

    lib.RLWEPlaintextLevel.argtypes = [_uintptr]
    lib.RLWEPlaintextLevel.restype = ctypes.c_int

    # --- MemEvaluationKeySet ---
    lib.NewMemEvalKeySet.argtypes = [
        _uintptr, ctypes.POINTER(_uintptr), ctypes.c_int, _errout,
    ]
    lib.NewMemEvalKeySet.restype = _uintptr

    lib.MemEvalKeySetMarshal.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_ulong), _errout]
    lib.MemEvalKeySetMarshal.restype = ctypes.c_void_p

    lib.MemEvalKeySetUnmarshal.argtypes = [ctypes.c_void_p, ctypes.c_ulong, _errout]
    lib.MemEvalKeySetUnmarshal.restype = _uintptr

    # --- Polynomial generation ---
    lib.GeneratePolynomialMonomial.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, _errout,
    ]
    lib.GeneratePolynomialMonomial.restype = _uintptr

    lib.GeneratePolynomialChebyshev.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, _errout,
    ]
    lib.GeneratePolynomialChebyshev.restype = _uintptr

    # --- Minimax sign coefficients ---
    lib.GenerateMinimaxSignCoeffs.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), _errout,
    ]
    lib.GenerateMinimaxSignCoeffs.restype = ctypes.POINTER(ctypes.c_double)


def _ensure_prototypes():
    """Ensure prototypes are set up (call once on first FFI use)."""
    global _prototypes_set
    if not _prototypes_set:
        _setup_prototypes(get_lib())
        _prototypes_set = True


def _lib_call():
    """Get library with prototypes guaranteed set up."""
    _ensure_prototypes()
    return get_lib()


# =========================================================================
# CKKS Parameters
# =========================================================================


def new_ckks_params(params_json: str) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()
    h = lib.NewCKKSParams(params_json.encode("utf-8"), ctypes.byref(err))
    _check_err(err)
    return GoHandle(h, tag="CKKSParams")


def ckks_params_max_slots(h: GoHandle) -> int:
    return int(_lib_call().CKKSParamsMaxSlots(_uintptr(h.raw)))


def ckks_params_max_level(h: GoHandle) -> int:
    return int(_lib_call().CKKSParamsMaxLevel(_uintptr(h.raw)))


def ckks_params_default_scale(h: GoHandle) -> int:
    return int(_lib_call().CKKSParamsDefaultScale(_uintptr(h.raw)))


def ckks_params_galois_element(h: GoHandle, rotation: int) -> int:
    return int(_lib_call().CKKSParamsGaloisElement(
        _uintptr(h.raw), ctypes.c_int(rotation),
    ))


def ckks_params_moduli_chain(h: GoHandle) -> list[int]:
    lib = _lib_call()
    err = _make_errout()
    out_len = ctypes.c_int(0)
    ptr = lib.CKKSParamsModuliChain(_uintptr(h.raw), ctypes.byref(out_len), ctypes.byref(err))
    _check_err(err)
    n = out_len.value
    if n == 0:
        return []
    result = [int(ptr[i]) for i in range(n)]
    lib.FreeCArray(ctypes.cast(ptr, ctypes.c_void_p))
    return result


def ckks_params_aux_moduli_chain(h: GoHandle) -> list[int]:
    lib = _lib_call()
    err = _make_errout()
    out_len = ctypes.c_int(0)
    ptr = lib.CKKSParamsAuxModuliChain(_uintptr(h.raw), ctypes.byref(out_len), ctypes.byref(err))
    _check_err(err)
    n = out_len.value
    if n == 0:
        return []
    result = [int(ptr[i]) for i in range(n)]
    lib.FreeCArray(ctypes.cast(ptr, ctypes.c_void_p))
    return result


# =========================================================================
# KeyGenerator
# =========================================================================


def new_key_generator(params_h: GoHandle) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()
    h = lib.NewKeyGenerator(_uintptr(params_h.raw), ctypes.byref(err))
    _check_err(err)
    return GoHandle(h, tag="KeyGenerator")


def keygen_gen_secret_key(kg_h: GoHandle) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()
    h = lib.KeyGenGenSecretKey(_uintptr(kg_h.raw), ctypes.byref(err))
    _check_err(err)
    return GoHandle(h, tag="SecretKey")


def keygen_gen_public_key(kg_h: GoHandle, sk_h: GoHandle) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()
    h = lib.KeyGenGenPublicKey(
        _uintptr(kg_h.raw), _uintptr(sk_h.raw), ctypes.byref(err),
    )
    _check_err(err)
    return GoHandle(h, tag="PublicKey")


def keygen_gen_relinearization_key(kg_h: GoHandle, sk_h: GoHandle) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()
    h = lib.KeyGenGenRelinearizationKey(
        _uintptr(kg_h.raw), _uintptr(sk_h.raw), ctypes.byref(err),
    )
    _check_err(err)
    return GoHandle(h, tag="RelinearizationKey")


def keygen_gen_galois_key(kg_h: GoHandle, sk_h: GoHandle, galois_element: int) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()
    h = lib.KeyGenGenGaloisKey(
        _uintptr(kg_h.raw), _uintptr(sk_h.raw),
        ctypes.c_ulonglong(galois_element), ctypes.byref(err),
    )
    _check_err(err)
    return GoHandle(h, tag="GaloisKey")


# =========================================================================
# CKKS Encoder
# =========================================================================


def new_ckks_encoder(params_h: GoHandle) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()
    h = lib.NewCKKSEncoder(_uintptr(params_h.raw), ctypes.byref(err))
    _check_err(err)
    return GoHandle(h, tag="CKKSEncoder")


def ckks_encoder_encode(
    enc_h: GoHandle, params_h: GoHandle,
    values: list[float], level: int, scale: int,
) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()
    arr, n = _doubles_ptr(values)
    h = lib.CKKSEncoderEncode(
        _uintptr(enc_h.raw), _uintptr(params_h.raw),
        arr, n, ctypes.c_int(level), ctypes.c_ulonglong(scale),
        ctypes.byref(err),
    )
    _check_err(err)
    return GoHandle(h, tag="RLWEPlaintext")


def ckks_encoder_decode(enc_h: GoHandle, pt_h: GoHandle, num_slots: int) -> list[float]:
    lib = _lib_call()
    err = _make_errout()
    out_len = ctypes.c_int(0)
    ptr = lib.CKKSEncoderDecode(
        _uintptr(enc_h.raw), _uintptr(pt_h.raw),
        ctypes.c_int(num_slots), ctypes.byref(out_len), ctypes.byref(err),
    )
    _check_err(err)
    n = out_len.value
    result = [float(ptr[i]) for i in range(n)]
    lib.FreeCArray(ctypes.cast(ptr, ctypes.c_void_p))
    return result


# =========================================================================
# Encryptor
# =========================================================================


def new_ckks_encryptor(params_h: GoHandle, pk_h: GoHandle) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()
    h = lib.NewCKKSEncryptor(
        _uintptr(params_h.raw), _uintptr(pk_h.raw), ctypes.byref(err),
    )
    _check_err(err)
    return GoHandle(h, tag="Encryptor")


def encryptor_encrypt_new(
    enc_h: GoHandle, pt_h: GoHandle, params_h: GoHandle,
) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()
    h = lib.EncryptorEncryptNew(
        _uintptr(enc_h.raw), _uintptr(pt_h.raw),
        _uintptr(params_h.raw), ctypes.byref(err),
    )
    _check_err(err)
    return GoHandle(h, tag="RLWECiphertext")


# =========================================================================
# Decryptor
# =========================================================================


def new_ckks_decryptor(params_h: GoHandle, sk_h: GoHandle) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()
    h = lib.NewCKKSDecryptor(
        _uintptr(params_h.raw), _uintptr(sk_h.raw), ctypes.byref(err),
    )
    _check_err(err)
    return GoHandle(h, tag="Decryptor")


def decryptor_decrypt_new(
    dec_h: GoHandle, ct_h: GoHandle, params_h: GoHandle,
) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()
    h = lib.DecryptorDecryptNew(
        _uintptr(dec_h.raw), _uintptr(ct_h.raw),
        _uintptr(params_h.raw), ctypes.byref(err),
    )
    _check_err(err)
    return GoHandle(h, tag="RLWEPlaintext")


# =========================================================================
# Serialization helpers
# =========================================================================


def _marshal(func_name: str, h: GoHandle) -> bytes:
    lib = _lib_call()
    err = _make_errout()
    out_len = ctypes.c_ulong(0)
    fn = getattr(lib, func_name)
    ptr = fn(_uintptr(h.raw), ctypes.byref(out_len), ctypes.byref(err))
    _check_err(err)
    data = ctypes.string_at(ptr, out_len.value)
    lib.FreeCArray(ptr)
    return data


def _unmarshal(func_name: str, data: bytes, tag: str) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()
    ptr, length = _bytes_ptr(data)
    fn = getattr(lib, func_name)
    h = fn(ptr, length, ctypes.byref(err))
    _check_err(err)
    return GoHandle(h, tag=tag)


def secret_key_marshal(h: GoHandle) -> bytes:
    return _marshal("SecretKeyMarshal", h)


def secret_key_unmarshal(data: bytes) -> GoHandle:
    return _unmarshal("SecretKeyUnmarshal", data, "SecretKey")


def public_key_marshal(h: GoHandle) -> bytes:
    return _marshal("PublicKeyMarshal", h)


def public_key_unmarshal(data: bytes) -> GoHandle:
    return _unmarshal("PublicKeyUnmarshal", data, "PublicKey")


def relinearization_key_marshal(h: GoHandle) -> bytes:
    return _marshal("RelinearizationKeyMarshal", h)


def relinearization_key_unmarshal(data: bytes) -> GoHandle:
    return _unmarshal("RelinearizationKeyUnmarshal", data, "RelinearizationKey")


def galois_key_marshal(h: GoHandle) -> bytes:
    return _marshal("GaloisKeyMarshal", h)


def galois_key_unmarshal(data: bytes) -> GoHandle:
    return _unmarshal("GaloisKeyUnmarshal", data, "GaloisKey")


def rlwe_ciphertext_marshal(h: GoHandle) -> bytes:
    return _marshal("RLWECiphertextMarshal", h)


def rlwe_ciphertext_unmarshal(data: bytes) -> GoHandle:
    return _unmarshal("RLWECiphertextUnmarshal", data, "RLWECiphertext")


def rlwe_ciphertext_level(h: GoHandle) -> int:
    return int(_lib_call().RLWECiphertextLevel(_uintptr(h.raw)))


def rlwe_plaintext_marshal(h: GoHandle) -> bytes:
    return _marshal("RLWEPlaintextMarshal", h)


def rlwe_plaintext_unmarshal(data: bytes) -> GoHandle:
    return _unmarshal("RLWEPlaintextUnmarshal", data, "RLWEPlaintext")


def rlwe_plaintext_level(h: GoHandle) -> int:
    return int(_lib_call().RLWEPlaintextLevel(_uintptr(h.raw)))


# =========================================================================
# MemEvaluationKeySet
# =========================================================================


def new_mem_eval_key_set(
    rlk_h: GoHandle | None,
    gk_handles: list[GoHandle],
) -> GoHandle:
    lib = _lib_call()
    err = _make_errout()

    rlk_raw = _uintptr(rlk_h.raw) if rlk_h else _uintptr(0)

    n = len(gk_handles)
    if n > 0:
        gk_arr = (_uintptr * n)(*[_uintptr(h.raw) for h in gk_handles])
        gk_ptr = ctypes.cast(gk_arr, ctypes.POINTER(_uintptr))
    else:
        gk_ptr = ctypes.POINTER(_uintptr)()

    h = lib.NewMemEvalKeySet(rlk_raw, gk_ptr, ctypes.c_int(n), ctypes.byref(err))
    _check_err(err)
    return GoHandle(h, tag="MemEvaluationKeySet")


def mem_eval_key_set_marshal(h: GoHandle) -> bytes:
    return _marshal("MemEvalKeySetMarshal", h)


def mem_eval_key_set_unmarshal(data: bytes) -> GoHandle:
    return _unmarshal("MemEvalKeySetUnmarshal", data, "MemEvaluationKeySet")


# =========================================================================
# Polynomial generation
# =========================================================================


def generate_polynomial_monomial(coeffs: list[float]) -> GoHandle:
    arr, n = _doubles_ptr(coeffs)
    err = _make_errout()
    r = _lib_call().GeneratePolynomialMonomial(arr, n, ctypes.byref(err))
    _check_err(err)
    return GoHandle(r, tag="Polynomial")


def generate_polynomial_chebyshev(coeffs: list[float]) -> GoHandle:
    arr, n = _doubles_ptr(coeffs)
    err = _make_errout()
    r = _lib_call().GeneratePolynomialChebyshev(arr, n, ctypes.byref(err))
    _check_err(err)
    return GoHandle(r, tag="Polynomial")


# =========================================================================
# Minimax sign coefficients
# =========================================================================


def generate_minimax_sign_coeffs(
    degrees: list[int], prec: int, log_alpha: int, log_err: int, debug: int,
) -> list[float]:
    lib = _lib_call()
    err = _make_errout()
    out_len = ctypes.c_int(0)
    n = len(degrees)
    deg_arr = (ctypes.c_int * n)(*degrees)
    ptr = lib.GenerateMinimaxSignCoeffs(
        deg_arr, ctypes.c_int(n),
        ctypes.c_int(prec),
        ctypes.c_int(log_alpha),
        ctypes.c_int(log_err),
        ctypes.c_int(debug),
        ctypes.byref(out_len),
        ctypes.byref(err),
    )
    _check_err(err)
    n_out = out_len.value
    result = [float(ptr[i]) for i in range(n_out)]
    lib.FreeCArray(ctypes.cast(ptr, ctypes.c_void_p))
    return result
