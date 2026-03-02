"""Unified Ciphertext type wrapping a cgo.Handle.

Replaces both CipherText (from client.py) and CipherTensor (from tensors.py).
A single type for transport and serialization.
"""

import struct

import torch

from orion.backend.orionclient import ffi


class Ciphertext:
    """Encrypted tensor wrapping a Go *orionclient.Ciphertext via cgo.Handle.

    Supports serialization and metadata queries. The Go Ciphertext internally
    holds multiple underlying rlwe.Ciphertexts and a shape -- multi-slot
    iteration happens in Go, not Python.
    """

    def __init__(self, handle, shape=None):
        self._handle = handle
        self._shape = torch.Size(shape) if shape is not None else None

    @property
    def handle(self):
        return self._handle

    @property
    def shape(self):
        if self._shape is not None:
            return self._shape
        # Fall back to querying Go
        dims = ffi.ciphertext_shape(self._handle)
        self._shape = torch.Size(dims)
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = torch.Size(value) if value is not None else None

    def close(self):
        """Release the Go handle. Idempotent."""
        if self._handle:
            self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return ffi.ciphertext_num_ciphertexts(self._handle)

    # ---- Serialization ----

    def to_bytes(self) -> bytes:
        go_bytes = ffi.ciphertext_marshal(self._handle)
        # Prepend Python shape: [ndim, d0, d1, ...] as little-endian int32s
        shape = list(self.shape)
        header = struct.pack("<i", len(shape))
        for d in shape:
            header += struct.pack("<i", d)
        return header + go_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> "Ciphertext":
        # Read shape header
        (ndim,) = struct.unpack("<i", data[:4])
        shape = []
        offset = 4
        for _ in range(ndim):
            (d,) = struct.unpack("<i", data[offset : offset + 4])
            shape.append(d)
            offset += 4
        go_bytes = data[offset:]
        h = ffi.ciphertext_unmarshal(go_bytes)
        return cls(h, shape=shape)

    # ---- Metadata queries ----

    def level(self):
        return ffi.ciphertext_level(self._handle)

    def scale(self):
        return ffi.ciphertext_scale(self._handle)

    def set_scale(self, scale):
        ffi.ciphertext_set_scale(self._handle, int(scale))

    def slots(self):
        return ffi.ciphertext_slots(self._handle)

    def degree(self):
        return ffi.ciphertext_degree(self._handle)

    def __str__(self):
        return f"<Ciphertext(shape={self.shape}, level={self.level()})>"


class PlainText:
    """Wrapper around a Go *orionclient.Plaintext via cgo.Handle.

    Local-only -- no cross-process serialization.
    """

    def __init__(self, handle, shape=None):
        self._handle = handle
        self._shape = torch.Size(shape) if shape is not None else torch.Size([])

    @property
    def handle(self):
        return self._handle

    @property
    def shape(self):
        return self._shape

    def level(self):
        return ffi.plaintext_level(self._handle)

    def scale(self):
        return ffi.plaintext_scale(self._handle)

    def set_scale(self, scale):
        ffi.plaintext_set_scale(self._handle, int(scale))

    def slots(self):
        return ffi.plaintext_slots(self._handle)

    def close(self):
        """Release the Go handle. Idempotent."""
        if self._handle:
            self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return 1  # Single plaintext per handle
