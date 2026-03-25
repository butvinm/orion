"""Low-level ctypes FFI bindings for the orion-evaluator bridge.

All evaluator operations accept/return raw bytes (Lattigo MarshalBinary format).
No Orion-specific types cross the FFI boundary.
"""

import ctypes
import os
import platform
import struct
import threading
from typing import Any

from .errors import EvaluatorError
from .gohandle import GoHandle

_uintptr = ctypes.c_size_t
_errout = ctypes.POINTER(ctypes.c_char_p)

_lib = None
_lib_lock = threading.Lock()
_prototypes_set = False


def _load_library() -> ctypes.CDLL:
    """Load the Linux shared library. Only Linux x86_64 is supported."""
    if platform.system() != "Linux":
        raise EvaluatorError(
            f"Unsupported platform: {platform.system()}. "
            "Only Linux x86_64 is supported. See README for building from source."
        )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(current_dir, "orion-evaluator-linux.so")

    try:
        return ctypes.CDLL(lib_path)
    except OSError as e:
        raise EvaluatorError(f"Failed to load library at {lib_path}: {e}") from e


def _get_lib() -> ctypes.CDLL:
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


def _setup_prototypes(lib: ctypes.CDLL) -> None:
    """Declare C function prototypes."""
    # --- Model ---
    lib.EvalLoadModel.argtypes = [ctypes.c_void_p, ctypes.c_ulong, _errout]
    lib.EvalLoadModel.restype = _uintptr

    lib.EvalModelClientParams.argtypes = [
        _uintptr,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_ulong),  # params
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_ulong),  # manifest
        ctypes.POINTER(ctypes.c_int),  # input level
        _errout,
    ]
    lib.EvalModelClientParams.restype = None

    lib.EvalModelClose.argtypes = [_uintptr]
    lib.EvalModelClose.restype = None

    # --- Evaluator ---
    lib.EvalNewEvaluator.argtypes = [
        ctypes.c_char_p,  # params JSON
        ctypes.c_void_p,
        ctypes.c_ulong,  # keys data
        ctypes.c_void_p,
        ctypes.c_ulong,  # btp keys data (optional, NULL/0 when not needed)
        _errout,
    ]
    lib.EvalNewEvaluator.restype = _uintptr

    lib.EvalForward.argtypes = [
        _uintptr,  # eval handle
        _uintptr,  # model handle
        ctypes.c_void_p,
        ctypes.c_ulong,  # ct data (length-prefixed)
        ctypes.c_int,  # numCTs
        ctypes.POINTER(ctypes.c_ulong),  # out len
        _errout,
    ]
    lib.EvalForward.restype = ctypes.c_void_p

    lib.EvalClose.argtypes = [_uintptr]
    lib.EvalClose.restype = None


def _ensure_prototypes() -> None:
    global _prototypes_set
    if _prototypes_set:
        return
    lib = _get_lib()  # acquires/releases _lib_lock internally
    with _lib_lock:
        if not _prototypes_set:
            _setup_prototypes(lib)
            _prototypes_set = True


def _lib_call() -> ctypes.CDLL:
    _ensure_prototypes()
    return _get_lib()


def _check_err(err_ptr: ctypes.c_char_p) -> None:
    if err_ptr and err_ptr.value:
        msg = err_ptr.value.decode("utf-8")
        _get_lib().FreeCArray(ctypes.cast(err_ptr, ctypes.c_void_p))
        raise EvaluatorError(msg)


def _make_errout() -> ctypes.c_char_p:
    return ctypes.c_char_p(None)


def _bytes_ptr(data: bytes) -> tuple[Any, ctypes.c_ulong]:
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    return ctypes.cast(buf, ctypes.c_void_p), ctypes.c_ulong(len(data))


# =========================================================================
# Model
# =========================================================================


def _delete_handle(raw: int) -> None:
    """Delete a cgo handle by raw value."""
    _get_lib().DeleteHandle(_uintptr(raw))


def load_model(data: bytes) -> GoHandle:
    """Load a .orion v2 file. Returns GoHandle wrapping the model."""
    lib = _lib_call()
    err = _make_errout()
    ptr, length = _bytes_ptr(data)
    h = lib.EvalLoadModel(ptr, length, ctypes.byref(err))
    _check_err(err)
    return GoHandle(int(h), tag="EvalModel", delete_fn=_delete_handle)


def model_client_params(handle: GoHandle) -> tuple[str, str, int]:
    """Get client params from model. Returns (params_json, manifest_json, input_level)."""
    lib = _lib_call()
    err = _make_errout()

    params_out = ctypes.c_char_p(None)
    params_len = ctypes.c_ulong(0)
    manifest_out = ctypes.c_char_p(None)
    manifest_len = ctypes.c_ulong(0)
    input_level = ctypes.c_int(0)

    lib.EvalModelClientParams(
        _uintptr(handle.raw),
        ctypes.byref(params_out),
        ctypes.byref(params_len),
        ctypes.byref(manifest_out),
        ctypes.byref(manifest_len),
        ctypes.byref(input_level),
        ctypes.byref(err),
    )
    _check_err(err)

    params_json = ctypes.string_at(params_out, params_len.value).decode("utf-8")
    manifest_json = ctypes.string_at(manifest_out, manifest_len.value).decode("utf-8")

    lib.FreeCArray(ctypes.cast(params_out, ctypes.c_void_p))
    lib.FreeCArray(ctypes.cast(manifest_out, ctypes.c_void_p))

    return params_json, manifest_json, input_level.value


def model_close(handle: GoHandle) -> None:
    """Close model Go-side resources, then release the cgo handle."""
    _lib_call().EvalModelClose(_uintptr(handle.raw))
    handle.close()


# =========================================================================
# Evaluator
# =========================================================================


def new_evaluator(
    params_json: str, keys_bytes: bytes, btp_keys_bytes: bytes | None = None
) -> GoHandle:
    """Create evaluator from params JSON and MemEvaluationKeySet bytes."""
    lib = _lib_call()
    err = _make_errout()
    ptr, length = _bytes_ptr(keys_bytes)

    if btp_keys_bytes is not None:
        btp_ptr, btp_length = _bytes_ptr(btp_keys_bytes)
    else:
        btp_ptr = None
        btp_length = ctypes.c_ulong(0)

    h = lib.EvalNewEvaluator(
        params_json.encode("utf-8"),
        ptr,
        length,
        btp_ptr,
        btp_length,
        ctypes.byref(err),
    )
    _check_err(err)
    return GoHandle(int(h), tag="Evaluator", delete_fn=_delete_handle)


def _pack_ct_list(ct_bytes_list: list[bytes]) -> bytes:
    """Pack a list of CT byte blobs into length-prefixed format."""
    parts = []
    for ct_bytes in ct_bytes_list:
        parts.append(struct.pack("<Q", len(ct_bytes)))
        parts.append(ct_bytes)
    return b"".join(parts)


def _unpack_ct_list(data: bytes) -> list[bytes]:
    """Unpack length-prefixed CT byte blobs."""
    result = []
    offset = 0
    while offset < len(data):
        ct_len = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        result.append(data[offset : offset + ct_len])
        offset += ct_len
    return result


def evaluator_forward(
    eval_handle: GoHandle, model_handle: GoHandle, ct_bytes_list: list[bytes]
) -> list[bytes]:
    """Run forward pass. Accepts/returns lists of Lattigo ciphertext binary bytes."""
    lib = _lib_call()
    err = _make_errout()
    out_len = ctypes.c_ulong(0)

    packed = _pack_ct_list(ct_bytes_list)
    ptr, length = _bytes_ptr(packed)

    result_ptr = lib.EvalForward(
        _uintptr(eval_handle.raw),
        _uintptr(model_handle.raw),
        ptr,
        length,
        ctypes.c_int(len(ct_bytes_list)),
        ctypes.byref(out_len),
        ctypes.byref(err),
    )
    _check_err(err)

    n = out_len.value
    buf = bytearray(n)
    ctypes.memmove((ctypes.c_char * n).from_buffer(buf), result_ptr, n)
    lib.FreeCArray(result_ptr)
    return _unpack_ct_list(bytes(buf))


def evaluator_close(eval_handle: GoHandle) -> None:
    """Close evaluator Go-side resources, then release the cgo handle."""
    _lib_call().EvalClose(_uintptr(eval_handle.raw))
    eval_handle.close()
