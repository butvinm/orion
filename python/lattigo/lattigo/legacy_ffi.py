"""Legacy FFI wrappers for Client-based bridge functions.

Used by orion_compiler's CompilerBackend during fit() for encode/decode,
parameter queries, and polynomial generation. These functions wrap the
old Client* bridge exports that still exist in the shared library.

This module will be deleted when the compiler backend is rewritten to
use Lattigo primitives directly.
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
    """Declare C function prototypes for legacy Client* bridge exports."""
    # --- Client ---
    lib.NewClient.argtypes = [ctypes.c_char_p, _errout]
    lib.NewClient.restype = _uintptr

    lib.NewClientFromSecretKey.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, ctypes.c_ulong, _errout,
    ]
    lib.NewClientFromSecretKey.restype = _uintptr

    lib.ClientClose.argtypes = [_uintptr]
    lib.ClientClose.restype = None

    lib.ClientSecretKey.argtypes = [
        _uintptr, ctypes.POINTER(ctypes.c_ulong), _errout,
    ]
    lib.ClientSecretKey.restype = ctypes.c_void_p

    lib.ClientEncode.argtypes = [
        _uintptr, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_int, ctypes.c_ulonglong, _errout,
    ]
    lib.ClientEncode.restype = _uintptr

    lib.ClientDecode.argtypes = [
        _uintptr, _uintptr, ctypes.POINTER(ctypes.c_int), _errout,
    ]
    lib.ClientDecode.restype = ctypes.POINTER(ctypes.c_double)

    lib.ClientEncrypt.argtypes = [_uintptr, _uintptr, _errout]
    lib.ClientEncrypt.restype = _uintptr

    lib.ClientDecrypt.argtypes = [
        _uintptr, _uintptr, ctypes.POINTER(ctypes.c_int), _errout,
    ]
    lib.ClientDecrypt.restype = ctypes.POINTER(_uintptr)

    lib.ClientGenerateRLK.argtypes = [
        _uintptr, ctypes.POINTER(ctypes.c_ulong), _errout,
    ]
    lib.ClientGenerateRLK.restype = ctypes.c_void_p

    lib.ClientGenerateGaloisKey.argtypes = [
        _uintptr, ctypes.c_ulonglong,
        ctypes.POINTER(ctypes.c_ulong), _errout,
    ]
    lib.ClientGenerateGaloisKey.restype = ctypes.c_void_p

    lib.ClientGenerateBootstrapKeys.argtypes = [
        _uintptr, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_ulong), _errout,
    ]
    lib.ClientGenerateBootstrapKeys.restype = ctypes.c_void_p

    lib.ClientGenerateKeys.argtypes = [_uintptr, ctypes.c_char_p, _errout]
    lib.ClientGenerateKeys.restype = _uintptr

    lib.ClientMaxSlots.argtypes = [_uintptr]
    lib.ClientMaxSlots.restype = ctypes.c_int

    lib.ClientDefaultScale.argtypes = [_uintptr]
    lib.ClientDefaultScale.restype = ctypes.c_ulonglong

    lib.ClientGaloisElement.argtypes = [_uintptr, ctypes.c_int]
    lib.ClientGaloisElement.restype = ctypes.c_ulonglong

    # --- Ciphertext type ops ---
    lib.CiphertextMarshal.argtypes = [
        _uintptr, ctypes.POINTER(ctypes.c_ulong), _errout,
    ]
    lib.CiphertextMarshal.restype = ctypes.c_void_p

    lib.CiphertextUnmarshal.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, _errout,
    ]
    lib.CiphertextUnmarshal.restype = _uintptr

    lib.CiphertextLevel.argtypes = [_uintptr]
    lib.CiphertextLevel.restype = ctypes.c_int

    lib.CiphertextScale.argtypes = [_uintptr]
    lib.CiphertextScale.restype = ctypes.c_ulonglong

    lib.CiphertextSetScale.argtypes = [_uintptr, ctypes.c_ulonglong]
    lib.CiphertextSetScale.restype = None

    lib.CiphertextSlots.argtypes = [_uintptr]
    lib.CiphertextSlots.restype = ctypes.c_int

    lib.CiphertextDegree.argtypes = [_uintptr]
    lib.CiphertextDegree.restype = ctypes.c_int

    lib.CiphertextShape.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_int)]
    lib.CiphertextShape.restype = ctypes.POINTER(ctypes.c_int)

    lib.CiphertextNumCiphertexts.argtypes = [_uintptr]
    lib.CiphertextNumCiphertexts.restype = ctypes.c_int

    try:
        lib.CombineSingleCiphertexts.argtypes = [
            ctypes.POINTER(_uintptr), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            _errout,
        ]
        lib.CombineSingleCiphertexts.restype = _uintptr
    except AttributeError:
        pass

    # --- Plaintext type ops ---
    lib.PlaintextLevel.argtypes = [_uintptr]
    lib.PlaintextLevel.restype = ctypes.c_int

    lib.PlaintextScale.argtypes = [_uintptr]
    lib.PlaintextScale.restype = ctypes.c_ulonglong

    lib.PlaintextSetScale.argtypes = [_uintptr, ctypes.c_ulonglong]
    lib.PlaintextSetScale.restype = None

    lib.PlaintextSlots.argtypes = [_uintptr]
    lib.PlaintextSlots.restype = ctypes.c_int

    # --- Polynomial ops ---
    lib.GeneratePolynomialMonomial.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, _errout,
    ]
    lib.GeneratePolynomialMonomial.restype = _uintptr

    lib.GeneratePolynomialChebyshev.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, _errout,
    ]
    lib.GeneratePolynomialChebyshev.restype = _uintptr

    # --- Client moduli chain ---
    lib.ClientModuliChain.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_int), _errout]
    lib.ClientModuliChain.restype = ctypes.POINTER(ctypes.c_ulonglong)

    lib.ClientAuxModuliChain.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_int), _errout]
    lib.ClientAuxModuliChain.restype = ctypes.POINTER(ctypes.c_ulonglong)

    # --- Minimax sign coefficients ---
    lib.GenerateMinimaxSignCoeffs.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), _errout,
    ]
    lib.GenerateMinimaxSignCoeffs.restype = ctypes.POINTER(ctypes.c_double)


def _ensure_prototypes():
    global _prototypes_set
    if not _prototypes_set:
        _setup_prototypes(get_lib())
        _prototypes_set = True


def _lib_call():
    _ensure_prototypes()
    return get_lib()


# =========================================================================
# Client
# =========================================================================


def new_client(params_json):
    lib = _lib_call()
    err = _make_errout()
    h = lib.NewClient(params_json.encode("utf-8"), ctypes.byref(err))
    _check_err(err)
    return GoHandle(h, tag="Client")


def client_close(h):
    _lib_call().ClientClose(_uintptr(h.raw))


def client_encode(h, values, level, scale):
    lib = _lib_call()
    err = _make_errout()
    arr, n = _doubles_ptr(values)
    pt_h = lib.ClientEncode(
        _uintptr(h.raw), arr, n, ctypes.c_int(level),
        ctypes.c_ulonglong(scale), ctypes.byref(err),
    )
    _check_err(err)
    return GoHandle(pt_h, tag="PlainText")


def client_decode(h, pt_h):
    lib = _lib_call()
    err = _make_errout()
    out_len = ctypes.c_int(0)
    ptr = lib.ClientDecode(
        _uintptr(h.raw), _uintptr(pt_h.raw), ctypes.byref(out_len), ctypes.byref(err),
    )
    _check_err(err)
    n = out_len.value
    result = [float(ptr[i]) for i in range(n)]
    lib.FreeCArray(ctypes.cast(ptr, ctypes.c_void_p))
    return result


def client_max_slots(h):
    return int(_lib_call().ClientMaxSlots(_uintptr(h.raw)))


def client_default_scale(h):
    return int(_lib_call().ClientDefaultScale(_uintptr(h.raw)))


def client_galois_element(h, rotation):
    return int(_lib_call().ClientGaloisElement(_uintptr(h.raw), ctypes.c_int(rotation)))


# =========================================================================
# Ciphertext type ops
# =========================================================================


def ciphertext_marshal(h):
    lib = _lib_call()
    err = _make_errout()
    out_len = ctypes.c_ulong(0)
    ptr = lib.CiphertextMarshal(_uintptr(h.raw), ctypes.byref(out_len), ctypes.byref(err))
    _check_err(err)
    data = ctypes.string_at(ptr, out_len.value)
    lib.FreeCArray(ptr)
    return data


def ciphertext_unmarshal(data):
    lib = _lib_call()
    err = _make_errout()
    ptr, length = _bytes_ptr(data)
    h = lib.CiphertextUnmarshal(ptr, length, ctypes.byref(err))
    _check_err(err)
    return GoHandle(h, tag="Ciphertext")


# =========================================================================
# Polynomial ops
# =========================================================================


def generate_polynomial_monomial(coeffs):
    arr, n = _doubles_ptr(coeffs)
    err = _make_errout()
    r = _lib_call().GeneratePolynomialMonomial(arr, n, ctypes.byref(err))
    _check_err(err)
    return GoHandle(r, tag="Polynomial")


def generate_polynomial_chebyshev(coeffs):
    arr, n = _doubles_ptr(coeffs)
    err = _make_errout()
    r = _lib_call().GeneratePolynomialChebyshev(arr, n, ctypes.byref(err))
    _check_err(err)
    return GoHandle(r, tag="Polynomial")


# =========================================================================
# Client moduli chain
# =========================================================================


def client_moduli_chain(h):
    lib = _lib_call()
    err = _make_errout()
    out_len = ctypes.c_int(0)
    ptr = lib.ClientModuliChain(_uintptr(h.raw), ctypes.byref(out_len), ctypes.byref(err))
    _check_err(err)
    n = out_len.value
    if n == 0:
        return []
    result = [int(ptr[i]) for i in range(n)]
    lib.FreeCArray(ctypes.cast(ptr, ctypes.c_void_p))
    return result


def client_aux_moduli_chain(h):
    lib = _lib_call()
    err = _make_errout()
    out_len = ctypes.c_int(0)
    ptr = lib.ClientAuxModuliChain(_uintptr(h.raw), ctypes.byref(out_len), ctypes.byref(err))
    _check_err(err)
    n = out_len.value
    if n == 0:
        return []
    result = [int(ptr[i]) for i in range(n)]
    lib.FreeCArray(ctypes.cast(ptr, ctypes.c_void_p))
    return result


# =========================================================================
# Minimax sign coefficients
# =========================================================================


def generate_minimax_sign_coeffs(degrees, prec, log_alpha, log_err, debug):
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
        ctypes.c_int(int(debug)),
        ctypes.byref(out_len),
        ctypes.byref(err),
    )
    _check_err(err)
    n_out = out_len.value
    result = [float(ptr[i]) for i in range(n_out)]
    lib.FreeCArray(ctypes.cast(ptr, ctypes.c_void_p))
    return result
