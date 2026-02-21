"""ctypes bindings for the orionclient bridge shared library.

Replaces backend/lattigo/bindings.py with handle-based FFI.
Every bridge function that can fail takes an errOut parameter;
this module checks it and raises RuntimeError on non-NULL.
"""

import ctypes
import os
import platform
import sys

# C types
_uintptr = ctypes.c_size_t  # uintptr_t
_handle = _uintptr
_errout = ctypes.POINTER(ctypes.c_char_p)

_lib = None
_lib_lock = None


def _load_library():
    """Load the platform-specific orionclient shared library."""
    if platform.system() == "Linux":
        lib_name = "orionclient-linux.so"
    elif platform.system() == "Darwin":
        if platform.machine().lower() in ("arm64", "aarch64"):
            lib_name = "orionclient-mac-arm64.dylib"
        else:
            lib_name = "orionclient-mac.dylib"
    elif platform.system() == "Windows":
        lib_name = "orionclient-windows.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {platform.system()}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(current_dir, lib_name)

    try:
        return ctypes.CDLL(lib_path)
    except OSError as e:
        raise RuntimeError(f"Failed to load orionclient library at {lib_path}: {e}")


def _get_lib():
    """Get the shared library, loading it on first access."""
    global _lib
    if _lib is None:
        _lib = _load_library()
        _setup_prototypes(_lib)
    return _lib


def _check_err(err_ptr):
    """Check errOut and raise if non-NULL, then free the C string."""
    if err_ptr and err_ptr.value:
        msg = err_ptr.value.decode("utf-8")
        _get_lib().FreeCArray(ctypes.cast(err_ptr, ctypes.c_void_p))
        raise RuntimeError(msg)


def _make_errout():
    """Create a fresh errOut pointer for a bridge call."""
    return ctypes.c_char_p(None)


def _setup_prototypes(lib):
    """Declare all C function prototypes."""

    # --- Lifecycle ---
    lib.DeleteHandle.argtypes = [_uintptr]
    lib.DeleteHandle.restype = None

    lib.FreeCArray.argtypes = [ctypes.c_void_p]
    lib.FreeCArray.restype = None

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

    # --- Evaluator ---
    lib.NewEvaluator.argtypes = [ctypes.c_char_p, _uintptr, _errout]
    lib.NewEvaluator.restype = _uintptr

    lib.EvaluatorClose.argtypes = [_uintptr]
    lib.EvaluatorClose.restype = None

    lib.EvalEncode.argtypes = [
        _uintptr, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_int, ctypes.c_ulonglong, _errout,
    ]
    lib.EvalEncode.restype = _uintptr

    # ct-ct arithmetic
    for name in ("EvalAdd", "EvalSub", "EvalMul"):
        fn = getattr(lib, name)
        fn.argtypes = [_uintptr, _uintptr, _uintptr, _errout]
        fn.restype = _uintptr

    # ct-pt arithmetic
    for name in ("EvalAddPlaintext", "EvalSubPlaintext", "EvalMulPlaintext"):
        fn = getattr(lib, name)
        fn.argtypes = [_uintptr, _uintptr, _uintptr, _errout]
        fn.restype = _uintptr

    # scalar arithmetic
    lib.EvalAddScalar.argtypes = [_uintptr, _uintptr, ctypes.c_double, _errout]
    lib.EvalAddScalar.restype = _uintptr

    lib.EvalMulScalar.argtypes = [_uintptr, _uintptr, ctypes.c_double, _errout]
    lib.EvalMulScalar.restype = _uintptr

    lib.EvalNegate.argtypes = [_uintptr, _uintptr, _errout]
    lib.EvalNegate.restype = _uintptr

    lib.EvalRotate.argtypes = [_uintptr, _uintptr, ctypes.c_int, _errout]
    lib.EvalRotate.restype = _uintptr

    lib.EvalRescale.argtypes = [_uintptr, _uintptr, _errout]
    lib.EvalRescale.restype = _uintptr

    lib.EvalPoly.argtypes = [
        _uintptr, _uintptr, _uintptr, ctypes.c_ulonglong, _errout,
    ]
    lib.EvalPoly.restype = _uintptr

    lib.EvalLinearTransform.argtypes = [_uintptr, _uintptr, _uintptr, _errout]
    lib.EvalLinearTransform.restype = _uintptr

    lib.EvalBootstrap.argtypes = [_uintptr, _uintptr, ctypes.c_int, _errout]
    lib.EvalBootstrap.restype = _uintptr

    lib.EvalMaxSlots.argtypes = [_uintptr]
    lib.EvalMaxSlots.restype = ctypes.c_int

    lib.EvalGaloisElement.argtypes = [_uintptr, ctypes.c_int]
    lib.EvalGaloisElement.restype = ctypes.c_ulonglong

    lib.EvalModuliChain.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_int)]
    lib.EvalModuliChain.restype = ctypes.POINTER(ctypes.c_ulonglong)

    lib.EvalDefaultScale.argtypes = [_uintptr]
    lib.EvalDefaultScale.restype = ctypes.c_ulonglong

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

    # --- Plaintext type ops ---
    lib.PlaintextLevel.argtypes = [_uintptr]
    lib.PlaintextLevel.restype = ctypes.c_int

    lib.PlaintextScale.argtypes = [_uintptr]
    lib.PlaintextScale.restype = ctypes.c_ulonglong

    lib.PlaintextSetScale.argtypes = [_uintptr, ctypes.c_ulonglong]
    lib.PlaintextSetScale.restype = None

    lib.PlaintextSlots.argtypes = [_uintptr]
    lib.PlaintextSlots.restype = ctypes.c_int

    lib.PlaintextShape.argtypes = [_uintptr, ctypes.POINTER(ctypes.c_int)]
    lib.PlaintextShape.restype = ctypes.POINTER(ctypes.c_int)

    # --- EvalKeyBundle ops ---
    lib.NewEvalKeyBundle.argtypes = []
    lib.NewEvalKeyBundle.restype = _uintptr

    lib.EvalKeyBundleSetRLK.argtypes = [_uintptr, ctypes.c_void_p, ctypes.c_ulong]
    lib.EvalKeyBundleSetRLK.restype = None

    lib.EvalKeyBundleAddGaloisKey.argtypes = [
        _uintptr, ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_ulong,
    ]
    lib.EvalKeyBundleAddGaloisKey.restype = None

    lib.EvalKeyBundleAddBootstrapKey.argtypes = [
        _uintptr, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong,
    ]
    lib.EvalKeyBundleAddBootstrapKey.restype = None

    lib.EvalKeyBundleSetBootLogP.argtypes = [
        _uintptr, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ]
    lib.EvalKeyBundleSetBootLogP.restype = None

    # --- LinearTransform ops ---
    lib.LinearTransformMarshal.argtypes = [
        _uintptr, ctypes.POINTER(ctypes.c_ulong), _errout,
    ]
    lib.LinearTransformMarshal.restype = ctypes.c_void_p

    lib.LinearTransformUnmarshal.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, _errout,
    ]
    lib.LinearTransformUnmarshal.restype = _uintptr

    # --- Polynomial ops ---
    lib.GeneratePolynomialMonomial.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ]
    lib.GeneratePolynomialMonomial.restype = _uintptr

    lib.GeneratePolynomialChebyshev.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ]
    lib.GeneratePolynomialChebyshev.restype = _uintptr


# =========================================================================
# High-level wrapper functions
# =========================================================================


def _doubles_ptr(values):
    """Convert a list/tuple of floats to a ctypes double array + length."""
    n = len(values)
    arr = (ctypes.c_double * n)(*values)
    return arr, ctypes.c_int(n)


def _bytes_ptr(data):
    """Convert bytes to (void_ptr, c_ulong) for bridge calls.

    We need to use c_void_p (or a buffer) instead of c_char_p
    because c_char_p truncates at null bytes.
    """
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    return ctypes.cast(buf, ctypes.c_void_p), ctypes.c_ulong(len(data))


# --- Handle lifecycle ---

def delete_handle(h):
    _get_lib().DeleteHandle(_uintptr(h))


def free_c_array(ptr):
    _get_lib().FreeCArray(ctypes.cast(ptr, ctypes.c_void_p))


# --- Client ---

def new_client(params_json):
    lib = _get_lib()
    err = _make_errout()
    h = lib.NewClient(params_json.encode("utf-8"), ctypes.byref(err))
    _check_err(err)
    return h


def new_client_from_secret_key(params_json, sk_bytes):
    lib = _get_lib()
    err = _make_errout()
    sk_ptr, sk_len = _bytes_ptr(sk_bytes)
    h = lib.NewClientFromSecretKey(
        params_json.encode("utf-8"), sk_ptr, sk_len, ctypes.byref(err),
    )
    _check_err(err)
    return h


def client_close(h):
    _get_lib().ClientClose(_uintptr(h))


def client_secret_key(h):
    lib = _get_lib()
    err = _make_errout()
    out_len = ctypes.c_ulong(0)
    ptr = lib.ClientSecretKey(_uintptr(h), ctypes.byref(out_len), ctypes.byref(err))
    _check_err(err)
    data = ctypes.string_at(ptr, out_len.value)  # ptr is c_void_p (int)
    lib.FreeCArray(ptr)
    return data


def client_encode(h, values, level, scale):
    lib = _get_lib()
    err = _make_errout()
    arr, n = _doubles_ptr(values)
    pt_h = lib.ClientEncode(
        _uintptr(h), arr, n, ctypes.c_int(level),
        ctypes.c_ulonglong(scale), ctypes.byref(err),
    )
    _check_err(err)
    return pt_h


def client_decode(h, pt_h):
    lib = _get_lib()
    err = _make_errout()
    out_len = ctypes.c_int(0)
    ptr = lib.ClientDecode(
        _uintptr(h), _uintptr(pt_h), ctypes.byref(out_len), ctypes.byref(err),
    )
    _check_err(err)
    n = out_len.value
    result = [float(ptr[i]) for i in range(n)]
    lib.FreeCArray(ctypes.cast(ptr, ctypes.c_void_p))
    return result


def client_encrypt(h, pt_h):
    lib = _get_lib()
    err = _make_errout()
    ct_h = lib.ClientEncrypt(_uintptr(h), _uintptr(pt_h), ctypes.byref(err))
    _check_err(err)
    return ct_h


def client_decrypt(h, ct_h):
    lib = _get_lib()
    err = _make_errout()
    num_out = ctypes.c_int(0)
    ptr = lib.ClientDecrypt(
        _uintptr(h), _uintptr(ct_h), ctypes.byref(num_out), ctypes.byref(err),
    )
    _check_err(err)
    n = num_out.value
    handles = [ptr[i] for i in range(n)]
    if ptr:
        lib.FreeCArray(ctypes.cast(ptr, ctypes.c_void_p))
    return handles


def client_generate_keys(h, manifest_json):
    lib = _get_lib()
    err = _make_errout()
    bundle_h = lib.ClientGenerateKeys(
        _uintptr(h), manifest_json.encode("utf-8"), ctypes.byref(err),
    )
    _check_err(err)
    return bundle_h


def client_max_slots(h):
    return int(_get_lib().ClientMaxSlots(_uintptr(h)))


def client_default_scale(h):
    return int(_get_lib().ClientDefaultScale(_uintptr(h)))


def client_galois_element(h, rotation):
    return int(_get_lib().ClientGaloisElement(_uintptr(h), ctypes.c_int(rotation)))


# --- Evaluator ---

def new_evaluator(params_json, keys_h):
    lib = _get_lib()
    err = _make_errout()
    h = lib.NewEvaluator(
        params_json.encode("utf-8"), _uintptr(keys_h), ctypes.byref(err),
    )
    _check_err(err)
    return h


def evaluator_close(h):
    _get_lib().EvaluatorClose(_uintptr(h))


def eval_encode(h, values, level, scale):
    lib = _get_lib()
    err = _make_errout()
    arr, n = _doubles_ptr(values)
    pt_h = lib.EvalEncode(
        _uintptr(h), arr, n, ctypes.c_int(level),
        ctypes.c_ulonglong(scale), ctypes.byref(err),
    )
    _check_err(err)
    return pt_h


def eval_add(h, ct0_h, ct1_h):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalAdd(_uintptr(h), _uintptr(ct0_h), _uintptr(ct1_h), ctypes.byref(err))
    _check_err(err)
    return r


def eval_sub(h, ct0_h, ct1_h):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalSub(_uintptr(h), _uintptr(ct0_h), _uintptr(ct1_h), ctypes.byref(err))
    _check_err(err)
    return r


def eval_mul(h, ct0_h, ct1_h):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalMul(_uintptr(h), _uintptr(ct0_h), _uintptr(ct1_h), ctypes.byref(err))
    _check_err(err)
    return r


def eval_add_plaintext(h, ct_h, pt_h):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalAddPlaintext(
        _uintptr(h), _uintptr(ct_h), _uintptr(pt_h), ctypes.byref(err),
    )
    _check_err(err)
    return r


def eval_sub_plaintext(h, ct_h, pt_h):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalSubPlaintext(
        _uintptr(h), _uintptr(ct_h), _uintptr(pt_h), ctypes.byref(err),
    )
    _check_err(err)
    return r


def eval_mul_plaintext(h, ct_h, pt_h):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalMulPlaintext(
        _uintptr(h), _uintptr(ct_h), _uintptr(pt_h), ctypes.byref(err),
    )
    _check_err(err)
    return r


def eval_add_scalar(h, ct_h, scalar):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalAddScalar(
        _uintptr(h), _uintptr(ct_h), ctypes.c_double(scalar), ctypes.byref(err),
    )
    _check_err(err)
    return r


def eval_mul_scalar(h, ct_h, scalar):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalMulScalar(
        _uintptr(h), _uintptr(ct_h), ctypes.c_double(scalar), ctypes.byref(err),
    )
    _check_err(err)
    return r


def eval_negate(h, ct_h):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalNegate(_uintptr(h), _uintptr(ct_h), ctypes.byref(err))
    _check_err(err)
    return r


def eval_rotate(h, ct_h, amount):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalRotate(
        _uintptr(h), _uintptr(ct_h), ctypes.c_int(amount), ctypes.byref(err),
    )
    _check_err(err)
    return r


def eval_rescale(h, ct_h):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalRescale(_uintptr(h), _uintptr(ct_h), ctypes.byref(err))
    _check_err(err)
    return r


def eval_poly(h, ct_h, poly_h, out_scale):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalPoly(
        _uintptr(h), _uintptr(ct_h), _uintptr(poly_h),
        ctypes.c_ulonglong(out_scale), ctypes.byref(err),
    )
    _check_err(err)
    return r


def eval_linear_transform(h, ct_h, lt_h):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalLinearTransform(
        _uintptr(h), _uintptr(ct_h), _uintptr(lt_h), ctypes.byref(err),
    )
    _check_err(err)
    return r


def eval_bootstrap(h, ct_h, num_slots):
    lib = _get_lib()
    err = _make_errout()
    r = lib.EvalBootstrap(
        _uintptr(h), _uintptr(ct_h), ctypes.c_int(num_slots), ctypes.byref(err),
    )
    _check_err(err)
    return r


def eval_max_slots(h):
    return int(_get_lib().EvalMaxSlots(_uintptr(h)))


def eval_galois_element(h, rotation):
    return int(_get_lib().EvalGaloisElement(_uintptr(h), ctypes.c_int(rotation)))


def eval_moduli_chain(h):
    lib = _get_lib()
    out_len = ctypes.c_int(0)
    ptr = lib.EvalModuliChain(_uintptr(h), ctypes.byref(out_len))
    n = out_len.value
    result = [int(ptr[i]) for i in range(n)]
    lib.FreeCArray(ctypes.cast(ptr, ctypes.c_void_p))
    return result


def eval_default_scale(h):
    return int(_get_lib().EvalDefaultScale(_uintptr(h)))


# --- Ciphertext type ops ---

def ciphertext_marshal(h):
    lib = _get_lib()
    err = _make_errout()
    out_len = ctypes.c_ulong(0)
    ptr = lib.CiphertextMarshal(_uintptr(h), ctypes.byref(out_len), ctypes.byref(err))
    _check_err(err)
    data = ctypes.string_at(ptr, out_len.value)  # ptr is c_void_p (int)
    lib.FreeCArray(ptr)
    return data


def ciphertext_unmarshal(data):
    lib = _get_lib()
    err = _make_errout()
    ptr, length = _bytes_ptr(data)
    h = lib.CiphertextUnmarshal(ptr, length, ctypes.byref(err))
    _check_err(err)
    return h


def ciphertext_level(h):
    return int(_get_lib().CiphertextLevel(_uintptr(h)))


def ciphertext_scale(h):
    return int(_get_lib().CiphertextScale(_uintptr(h)))


def ciphertext_set_scale(h, scale):
    _get_lib().CiphertextSetScale(_uintptr(h), ctypes.c_ulonglong(scale))


def ciphertext_slots(h):
    return int(_get_lib().CiphertextSlots(_uintptr(h)))


def ciphertext_degree(h):
    return int(_get_lib().CiphertextDegree(_uintptr(h)))


def ciphertext_shape(h):
    lib = _get_lib()
    out_len = ctypes.c_int(0)
    ptr = lib.CiphertextShape(_uintptr(h), ctypes.byref(out_len))
    n = out_len.value
    result = [int(ptr[i]) for i in range(n)]
    lib.FreeCArray(ctypes.cast(ptr, ctypes.c_void_p))
    return result


def ciphertext_num_ciphertexts(h):
    return int(_get_lib().CiphertextNumCiphertexts(_uintptr(h)))


# --- Plaintext type ops ---

def plaintext_level(h):
    return int(_get_lib().PlaintextLevel(_uintptr(h)))


def plaintext_scale(h):
    return int(_get_lib().PlaintextScale(_uintptr(h)))


def plaintext_set_scale(h, scale):
    _get_lib().PlaintextSetScale(_uintptr(h), ctypes.c_ulonglong(scale))


def plaintext_slots(h):
    return int(_get_lib().PlaintextSlots(_uintptr(h)))


# --- EvalKeyBundle ops ---

def new_eval_key_bundle():
    return _get_lib().NewEvalKeyBundle()


def eval_key_bundle_set_rlk(h, data):
    ptr, length = _bytes_ptr(data)
    _get_lib().EvalKeyBundleSetRLK(_uintptr(h), ptr, length)


def eval_key_bundle_add_galois_key(h, gal_el, data):
    ptr, length = _bytes_ptr(data)
    _get_lib().EvalKeyBundleAddGaloisKey(
        _uintptr(h), ctypes.c_ulonglong(gal_el), ptr, length,
    )


def eval_key_bundle_add_bootstrap_key(h, slots, data):
    ptr, length = _bytes_ptr(data)
    _get_lib().EvalKeyBundleAddBootstrapKey(
        _uintptr(h), ctypes.c_int(slots), ptr, length,
    )


def eval_key_bundle_set_boot_logp(h, logp):
    n = len(logp)
    arr = (ctypes.c_int * n)(*logp)
    _get_lib().EvalKeyBundleSetBootLogP(_uintptr(h), arr, ctypes.c_int(n))


# --- LinearTransform ops ---

def linear_transform_unmarshal(data):
    lib = _get_lib()
    err = _make_errout()
    ptr, length = _bytes_ptr(data)
    h = lib.LinearTransformUnmarshal(ptr, length, ctypes.byref(err))
    _check_err(err)
    return h


# --- Polynomial ops ---

def generate_polynomial_monomial(coeffs):
    arr, n = _doubles_ptr(coeffs)
    return _get_lib().GeneratePolynomialMonomial(arr, n)


def generate_polynomial_chebyshev(coeffs):
    arr, n = _doubles_ptr(coeffs)
    return _get_lib().GeneratePolynomialChebyshev(arr, n)
