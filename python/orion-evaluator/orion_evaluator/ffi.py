"""Low-level ctypes FFI bindings for the orion-evaluator bridge.

All evaluator operations accept/return raw bytes (Lattigo MarshalBinary format).
No Orion-specific types cross the FFI boundary.
"""

import ctypes
import os
import platform
import threading

_uintptr = ctypes.c_size_t
_errout = ctypes.POINTER(ctypes.c_char_p)

_lib = None
_lib_lock = threading.Lock()
_prototypes_set = False


def _load_library():
    """Load the platform-specific shared library."""
    if platform.system() == "Linux":
        lib_name = "orion-evaluator-linux.so"
    elif platform.system() == "Darwin":
        if platform.machine().lower() in ("arm64", "aarch64"):
            lib_name = "orion-evaluator-mac-arm64.dylib"
        else:
            lib_name = "orion-evaluator-mac.dylib"
    elif platform.system() == "Windows":
        lib_name = "orion-evaluator-windows.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {platform.system()}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(current_dir, lib_name)

    try:
        return ctypes.CDLL(lib_path)
    except OSError as e:
        raise RuntimeError(f"Failed to load library at {lib_path}: {e}")


def _get_lib():
    """Get the shared library, loading on first access."""
    global _lib
    if _lib is None:
        with _lib_lock:
            if _lib is None:
                lib = _load_library()
                lib.DeleteHandle.argtypes = [_uintptr]
                lib.DeleteHandle.restype = None
                lib.FreeCArray.argtypes = [ctypes.c_void_p]
                lib.FreeCArray.restype = None
                _lib = lib
    return _lib


def _setup_prototypes(lib):
    """Declare C function prototypes."""
    # --- Model ---
    lib.EvalLoadModel.argtypes = [ctypes.c_void_p, ctypes.c_ulong, _errout]
    lib.EvalLoadModel.restype = _uintptr

    lib.EvalModelClientParams.argtypes = [
        _uintptr,
        ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_ulong),  # params
        ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_ulong),  # manifest
        ctypes.POINTER(ctypes.c_int),  # input level
        _errout,
    ]
    lib.EvalModelClientParams.restype = None

    lib.EvalModelClose.argtypes = [_uintptr]
    lib.EvalModelClose.restype = None

    # --- Evaluator ---
    lib.EvalNewEvaluator.argtypes = [
        ctypes.c_char_p,  # params JSON
        ctypes.c_void_p, ctypes.c_ulong,  # keys data
        _errout,
    ]
    lib.EvalNewEvaluator.restype = _uintptr

    lib.EvalForward.argtypes = [
        _uintptr,  # eval handle
        _uintptr,  # model handle
        ctypes.c_void_p, ctypes.c_ulong,  # ct data
        ctypes.POINTER(ctypes.c_ulong),  # out len
        _errout,
    ]
    lib.EvalForward.restype = ctypes.c_void_p

    lib.EvalClose.argtypes = [_uintptr]
    lib.EvalClose.restype = None


def _ensure_prototypes():
    global _prototypes_set
    if not _prototypes_set:
        _setup_prototypes(_get_lib())
        _prototypes_set = True


def _lib_call():
    _ensure_prototypes()
    return _get_lib()


def _check_err(err_ptr):
    if err_ptr and err_ptr.value:
        msg = err_ptr.value.decode("utf-8")
        _get_lib().FreeCArray(ctypes.cast(err_ptr, ctypes.c_void_p))
        raise RuntimeError(msg)


def _make_errout():
    return ctypes.c_char_p(None)


def _bytes_ptr(data):
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    return ctypes.cast(buf, ctypes.c_void_p), ctypes.c_ulong(len(data))


# =========================================================================
# Model
# =========================================================================


def load_model(data: bytes) -> int:
    """Load a .orion v2 file. Returns raw handle (uintptr_t)."""
    lib = _lib_call()
    err = _make_errout()
    ptr, length = _bytes_ptr(data)
    h = lib.EvalLoadModel(ptr, length, ctypes.byref(err))
    _check_err(err)
    return int(h)


def model_client_params(handle: int) -> tuple[str, str, int]:
    """Get client params from model. Returns (params_json, manifest_json, input_level)."""
    lib = _lib_call()
    err = _make_errout()

    params_out = ctypes.c_char_p(None)
    params_len = ctypes.c_ulong(0)
    manifest_out = ctypes.c_char_p(None)
    manifest_len = ctypes.c_ulong(0)
    input_level = ctypes.c_int(0)

    lib.EvalModelClientParams(
        _uintptr(handle),
        ctypes.byref(params_out), ctypes.byref(params_len),
        ctypes.byref(manifest_out), ctypes.byref(manifest_len),
        ctypes.byref(input_level),
        ctypes.byref(err),
    )
    _check_err(err)

    params_json = ctypes.string_at(params_out, params_len.value).decode("utf-8")
    manifest_json = ctypes.string_at(manifest_out, manifest_len.value).decode("utf-8")

    lib.FreeCArray(ctypes.cast(params_out, ctypes.c_void_p))
    lib.FreeCArray(ctypes.cast(manifest_out, ctypes.c_void_p))

    return params_json, manifest_json, input_level.value


def model_close(handle: int):
    """Close model resources."""
    _lib_call().EvalModelClose(_uintptr(handle))


def delete_handle(handle: int):
    """Delete a cgo handle."""
    _get_lib().DeleteHandle(_uintptr(handle))


# =========================================================================
# Evaluator
# =========================================================================


def new_evaluator(params_json: str, keys_bytes: bytes) -> int:
    """Create evaluator from params JSON and MemEvaluationKeySet bytes. Returns raw handle."""
    lib = _lib_call()
    err = _make_errout()
    ptr, length = _bytes_ptr(keys_bytes)
    h = lib.EvalNewEvaluator(
        params_json.encode("utf-8"),
        ptr, length,
        ctypes.byref(err),
    )
    _check_err(err)
    return int(h)


def evaluator_forward(eval_handle: int, model_handle: int, ct_bytes: bytes) -> bytes:
    """Run forward pass. Accepts/returns Lattigo ciphertext binary bytes."""
    lib = _lib_call()
    err = _make_errout()
    out_len = ctypes.c_ulong(0)
    ptr, length = _bytes_ptr(ct_bytes)

    result_ptr = lib.EvalForward(
        _uintptr(eval_handle),
        _uintptr(model_handle),
        ptr, length,
        ctypes.byref(out_len),
        ctypes.byref(err),
    )
    _check_err(err)

    result = ctypes.string_at(result_ptr, out_len.value)
    lib.FreeCArray(result_ptr)
    return result


def evaluator_close(eval_handle: int):
    """Close evaluator resources."""
    _lib_call().EvalClose(_uintptr(eval_handle))
