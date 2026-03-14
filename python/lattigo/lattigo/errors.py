"""Library-specific exceptions for the lattigo Python bindings."""


class LatticeError(Exception):
    """Base exception for all lattigo errors."""


class HandleClosedError(LatticeError):
    """Raised when attempting to use a closed GoHandle."""


class FFIError(LatticeError):
    """Raised when a Go FFI call returns an error."""
