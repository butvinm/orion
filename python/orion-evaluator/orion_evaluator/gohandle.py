"""GoHandle — RAII wrapper for cgo.Handle values.

Mirrors lattigo.gohandle.GoHandle for consistent handle lifecycle management
across all Orion FFI packages.
"""

from collections.abc import Callable

from .errors import EvaluatorError


class HandleClosedError(EvaluatorError):
    """Raised when accessing a closed handle."""


class GoHandle:
    """RAII wrapper for a cgo.Handle (opaque uintptr_t). Idempotent close."""

    __slots__ = ("_delete_fn", "_raw", "_tag")

    def __init__(self, raw: int, tag: str = "", *, delete_fn: Callable[[int], None] | None = None):
        self._raw = raw
        self._tag = tag
        self._delete_fn = delete_fn

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
        if self._raw and self._delete_fn is not None:
            self._delete_fn(self._raw)
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
