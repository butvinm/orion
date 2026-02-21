"""Unified Ciphertext type wrapping a cgo.Handle.

Replaces both CipherText (from client.py) and CipherTensor (from tensors.py).
A single type for transport and computation.
"""

import math
import struct
import sys

import torch

from orion.backend.orionclient import ffi


class Ciphertext:
    """Encrypted tensor wrapping a Go *orionclient.Ciphertext via cgo.Handle.

    Supports arithmetic (when an evaluator context is set), serialization,
    and metadata queries. The Go Ciphertext internally holds multiple
    underlying rlwe.Ciphertexts and a shape -- multi-slot iteration happens
    in Go, not Python.
    """

    def __init__(self, handle, shape=None, context=None):
        self._handle = handle
        self._shape = torch.Size(shape) if shape is not None else None
        self.context = context
        # on_shape tracks the "original" (pre-padding) shape for nn modules
        self.on_shape = self._shape

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

    def __del__(self):
        if "sys" in globals() and sys.modules and self._handle:
            try:
                ffi.delete_handle(self._handle)
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

    # ---- Arithmetic (requires evaluator context) ----

    def _eval_h(self):
        """Get the evaluator handle from context."""
        if self.context is None or not hasattr(self.context, "eval_handle"):
            raise RuntimeError("No evaluator context set on this Ciphertext")
        return self.context.eval_handle

    def _wrap(self, handle):
        """Wrap a result handle with same context and shape."""
        ct = Ciphertext(handle, shape=self.shape, context=self.context)
        ct.on_shape = self.on_shape
        return ct

    def add(self, other, in_place=False):
        if isinstance(other, (int, float)):
            r = ffi.eval_add_scalar(self._eval_h(), self._handle, float(other))
        elif isinstance(other, PlainText):
            r = ffi.eval_add_plaintext(self._eval_h(), self._handle, other.handle)
        elif isinstance(other, Ciphertext):
            r = ffi.eval_add(self._eval_h(), self._handle, other._handle)
        else:
            raise TypeError(f"Cannot add Ciphertext and {type(other)}")
        if in_place:
            ffi.delete_handle(self._handle)
            self._handle = r
            return self
        return self._wrap(r)

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        return self.add(other, in_place=True)

    def sub(self, other, in_place=False):
        if isinstance(other, (int, float)):
            # sub scalar = add negative scalar
            r = ffi.eval_add_scalar(self._eval_h(), self._handle, -float(other))
        elif isinstance(other, PlainText):
            r = ffi.eval_sub_plaintext(self._eval_h(), self._handle, other.handle)
        elif isinstance(other, Ciphertext):
            r = ffi.eval_sub(self._eval_h(), self._handle, other._handle)
        else:
            raise TypeError(f"Cannot sub Ciphertext and {type(other)}")
        if in_place:
            ffi.delete_handle(self._handle)
            self._handle = r
            return self
        return self._wrap(r)

    def __sub__(self, other):
        return self.sub(other)

    def __isub__(self, other):
        return self.sub(other, in_place=True)

    def mul(self, other, in_place=False):
        need_rescale = False
        if isinstance(other, (int, float)):
            r = ffi.eval_mul_scalar(self._eval_h(), self._handle, float(other))
            # Float scalar multiplication changes scale; int does not.
            # Match old backend behavior: rescale after float, not after int.
            if not (isinstance(other, int) or
                    (isinstance(other, float) and other == int(other))):
                need_rescale = True
        elif isinstance(other, PlainText):
            r = ffi.eval_mul_plaintext(self._eval_h(), self._handle, other.handle)
            need_rescale = True
        elif isinstance(other, Ciphertext):
            r = ffi.eval_mul(self._eval_h(), self._handle, other._handle)
            need_rescale = True
        else:
            raise TypeError(f"Cannot mul Ciphertext and {type(other)}")
        if need_rescale:
            r2 = ffi.eval_rescale(self._eval_h(), r)
            ffi.delete_handle(r)
            r = r2
        if in_place:
            ffi.delete_handle(self._handle)
            self._handle = r
            return self
        return self._wrap(r)

    def __mul__(self, other):
        return self.mul(other)

    def __imul__(self, other):
        return self.mul(other, in_place=True)

    def __neg__(self):
        r = ffi.eval_negate(self._eval_h(), self._handle)
        return self._wrap(r)

    def roll(self, amount, in_place=False):
        r = ffi.eval_rotate(self._eval_h(), self._handle, amount)
        if in_place:
            ffi.delete_handle(self._handle)
            self._handle = r
            return self
        return self._wrap(r)

    def bootstrap(self):
        elements = self.on_shape.numel()
        slots = 2 ** math.ceil(math.log2(elements))
        slots = int(min(self.slots(), slots))
        r = ffi.eval_bootstrap(self._eval_h(), self._handle, slots)
        return self._wrap(r)

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

    def mul(self, other, in_place=False):
        """Multiply this plaintext by a ciphertext (result is a ciphertext)."""
        if not isinstance(other, Ciphertext):
            raise TypeError(f"PlainText.mul requires a Ciphertext, got {type(other)}")
        eval_h = other._eval_h()
        r = ffi.eval_mul_plaintext(eval_h, other._handle, self._handle)
        # Rescale after plaintext multiplication (matches old backend behavior)
        r2 = ffi.eval_rescale(eval_h, r)
        ffi.delete_handle(r)
        r = r2
        if in_place:
            ffi.delete_handle(other._handle)
            other._handle = r
            return other
        return other._wrap(r)

    def __mul__(self, other):
        return self.mul(other)

    def __del__(self):
        if "sys" in globals() and sys.modules and self._handle:
            try:
                ffi.delete_handle(self._handle)
            except Exception:
                pass

    def __len__(self):
        return 1  # Single plaintext per handle
