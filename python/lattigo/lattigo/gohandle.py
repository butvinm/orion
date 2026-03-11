"""GoHandle — RAII wrapper for cgo.Handle values.

Standalone module with no dependency on the old orionclient FFI.
Library loading is deferred to first use.
"""

import ctypes
import os
import platform
import threading

_uintptr = ctypes.c_size_t

_lib = None
_lib_lock = threading.Lock()


def _load_library():
    """Load the platform-specific shared library."""
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
        raise RuntimeError(f"Failed to load library at {lib_path}: {e}") from e


def get_lib():
    """Get the shared library, loading it on first access."""
    global _lib
    if _lib is None:
        with _lib_lock:
            if _lib is None:
                lib = _load_library()
                # Minimal prototype setup for handle lifecycle
                lib.DeleteHandle.argtypes = [_uintptr]
                lib.DeleteHandle.restype = None
                lib.FreeCArray.argtypes = [ctypes.c_void_p]
                lib.FreeCArray.restype = None
                _lib = lib
    return _lib


class GoHandle:
    """RAII wrapper for a cgo.Handle (opaque uintptr_t). Idempotent close."""

    __slots__ = ("_raw", "_tag")

    def __init__(self, raw: int, tag: str = ""):
        self._raw = raw
        self._tag = tag

    def __repr__(self):
        if self._raw:
            return f"GoHandle({self._raw} {self._tag})" if self._tag else f"GoHandle({self._raw})"
        return f"GoHandle(closed {self._tag})" if self._tag else "GoHandle(closed)"

    @property
    def raw(self) -> int:
        if not self._raw:
            raise RuntimeError("Use of closed handle")
        return self._raw

    def close(self):
        """Release the Go object. Idempotent — second call is a no-op."""
        if self._raw:
            get_lib().DeleteHandle(_uintptr(self._raw))
            self._raw = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __bool__(self):
        return bool(self._raw)
