"""GoHandle — RAII wrapper for cgo.Handle values.

Standalone module — loads the orionclient shared library at first use.
Library loading is deferred to first use.
"""

import ctypes
import os
import platform
import threading

from .errors import FFIError, HandleClosedError

_uintptr = ctypes.c_size_t

_lib = None
_lib_lock = threading.Lock()


_LIB_FILENAMES: dict[tuple[str, str], str] = {
    ("Linux", "x86_64"): "orionclient-linux-amd64.so",
    ("Darwin", "x86_64"): "orionclient-darwin-amd64.dylib",
    ("Darwin", "arm64"): "orionclient-darwin-arm64.dylib",
}


def _load_library() -> ctypes.CDLL:
    """Load the platform-specific shared library (Linux x86_64, macOS x86_64/ARM64)."""
    key = (platform.system(), platform.machine())
    filename = _LIB_FILENAMES.get(key)
    if filename is None:
        raise FFIError(
            f"Unsupported platform: {key[0]} {key[1]}. "
            "Supported: Linux x86_64, macOS x86_64, macOS arm64."
        )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(current_dir, filename)

    try:
        return ctypes.CDLL(lib_path)
    except OSError as e:
        raise FFIError(f"Failed to load library at {lib_path}: {e}") from e


def get_lib() -> ctypes.CDLL:
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

    def __repr__(self) -> str:
        if self._raw:
            return f"GoHandle({self._raw} {self._tag})" if self._tag else f"GoHandle({self._raw})"
        return f"GoHandle(closed {self._tag})" if self._tag else "GoHandle(closed)"

    @property
    def raw(self) -> int:
        if not self._raw:
            raise HandleClosedError("Use of closed handle")
        return self._raw

    def close(self) -> None:
        """Release the Go object. Idempotent — second call is a no-op."""
        if self._raw:
            get_lib().DeleteHandle(_uintptr(self._raw))
            self._raw = 0

    def __enter__(self) -> "GoHandle":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __bool__(self) -> bool:
        return bool(self._raw)
