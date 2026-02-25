"""Client: key generation, encode/decode, encrypt/decrypt.

Has secret key. Produces EvalKeys for the Evaluator.
Thin wrapper over orionclient FFI -- one FFI call per method.
"""

import torch

from orion.params import CKKSParams
from orion.compiled_model import KeyManifest, EvalKeys
from orion.ciphertext import Ciphertext, PlainText
from orion.backend.orionclient import ffi


class Client:
    """Client-side operations: key generation, encode/decode, encrypt/decrypt.

    Usage:
        client = Client(compiled.params)
        keys = client.generate_keys(compiled.manifest)
        pt = client.encode(tensor, level=compiled.input_level)
        ct = client.encrypt(pt)
        ...
        result = client.decode(client.decrypt(ct_result))

    To recreate a Client with the same secret key:
        sk_bytes = client.secret_key
        ...
        client2 = Client(params, secret_key=sk_bytes)
    """

    def __init__(self, params: CKKSParams, secret_key: bytes | None = None):
        self.ckks_params = params
        self._params_json_str = params.to_bridge_json()

        if secret_key is not None:
            self._handle = ffi.new_client_from_secret_key(
                self._params_json_str, secret_key,
            )
        else:
            self._handle = ffi.new_client(self._params_json_str)

    @property
    def secret_key(self) -> bytes:
        """Serialize the secret key for later restoration."""
        return ffi.client_secret_key(self._handle)

    def generate_keys(self, manifest: KeyManifest) -> EvalKeys:
        """Generate all evaluation keys specified by manifest.

        Returns an EvalKeys with serialized key blobs. The Evaluator
        deserializes these when constructing the Go EvalKeyBundle.
        """
        keys = EvalKeys()

        if manifest.needs_rlk:
            out_len = ffi.ctypes.c_ulong(0)
            err = ffi._make_errout()
            lib = ffi._get_lib()
            ptr = lib.ClientGenerateRLK(
                ffi._uintptr(self._handle.raw),
                ffi.ctypes.byref(out_len),
                ffi.ctypes.byref(err),
            )
            ffi._check_err(err)
            keys.rlk_data = ffi.ctypes.string_at(ptr, out_len.value)
            lib.FreeCArray(ptr)

        for gal_el in sorted(manifest.galois_elements):
            out_len = ffi.ctypes.c_ulong(0)
            err = ffi._make_errout()
            lib = ffi._get_lib()
            ptr = lib.ClientGenerateGaloisKey(
                ffi._uintptr(self._handle.raw),
                ffi.ctypes.c_ulonglong(gal_el),
                ffi.ctypes.byref(out_len),
                ffi.ctypes.byref(err),
            )
            ffi._check_err(err)
            keys.galois_keys[gal_el] = ffi.ctypes.string_at(ptr, out_len.value)
            lib.FreeCArray(ptr)

        if manifest.bootstrap_slots:
            logp = list(manifest.boot_logp)
            for slot_count in manifest.bootstrap_slots:
                out_len = ffi.ctypes.c_ulong(0)
                err = ffi._make_errout()
                lib = ffi._get_lib()
                logp_arr = (ffi.ctypes.c_int * len(logp))(*logp)
                ptr = lib.ClientGenerateBootstrapKeys(
                    ffi._uintptr(self._handle.raw),
                    ffi.ctypes.c_int(slot_count),
                    logp_arr, ffi.ctypes.c_int(len(logp)),
                    ffi.ctypes.byref(out_len),
                    ffi.ctypes.byref(err),
                )
                ffi._check_err(err)
                keys.bootstrap_keys[slot_count] = ffi.ctypes.string_at(
                    ptr, out_len.value
                )
                lib.FreeCArray(ptr)

        return keys

    def encode(self, tensor, level=None, scale=None):
        """Encode a tensor into a PlainText."""
        if isinstance(tensor, list):
            tensor = torch.tensor(tensor)
        elif not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expected tensor to be a list or torch.Tensor, "
                f"got {type(tensor)}"
            )

        if level is None:
            level = self.ckks_params.max_level
        if scale is None:
            scale = ffi.client_default_scale(self._handle)

        max_slots = ffi.client_max_slots(self._handle)
        num_elements = tensor.numel()

        tensor = tensor.cpu()
        pad_length = (-num_elements) % max_slots
        vector = torch.zeros(num_elements + pad_length, dtype=torch.float64)
        vector[:num_elements] = tensor.flatten().to(torch.float64)
        num_plaintexts = len(vector) // max_slots

        pt_handles = []
        for i in range(num_plaintexts):
            chunk = vector[i * max_slots : (i + 1) * max_slots].tolist()
            pt_h = ffi.client_encode(self._handle, chunk, level, scale)
            pt_handles.append(pt_h)

        if len(pt_handles) == 1:
            return PlainText(pt_handles[0], shape=tensor.shape)

        return _MultiPlainText(pt_handles, tensor.shape)

    def decode(self, plaintext):
        """Decode a PlainText back to a tensor."""
        if isinstance(plaintext, _MultiPlainText):
            values = []
            for pt_h in plaintext.handles:
                vals = ffi.client_decode(self._handle, pt_h)
                values.extend(vals)
        else:
            values = ffi.client_decode(self._handle, plaintext.handle)

        values = torch.tensor(values)[: plaintext.shape.numel()]
        return values.reshape(plaintext.shape)

    def encrypt(self, plaintext):
        """Encrypt a PlainText into a Ciphertext."""
        if isinstance(plaintext, _MultiPlainText):
            ct_handles = []
            for pt_h in plaintext.handles:
                ct_h = ffi.client_encrypt(self._handle, pt_h)
                ct_handles.append(ct_h)
            if len(ct_handles) == 1:
                return Ciphertext(ct_handles[0], shape=plaintext.shape)
            # Combine single-ct handles into one multi-ct Ciphertext
            try:
                combined_h = ffi.combine_single_ciphertexts(
                    ct_handles, list(plaintext.shape),
                )
            except:
                for h in ct_handles:
                    h.close()
                raise
            # Combine succeeded — close individuals, they're now inside combined
            for h in ct_handles:
                h.close()
            return Ciphertext(combined_h, shape=plaintext.shape)
        else:
            ct_h = ffi.client_encrypt(self._handle, plaintext.handle)
            return Ciphertext(ct_h, shape=plaintext.shape)

    def decrypt(self, ciphertext):
        """Decrypt a Ciphertext into a PlainText."""
        pt_handles = ffi.client_decrypt(self._handle, ciphertext.handle)
        if len(pt_handles) == 1:
            return PlainText(pt_handles[0], shape=ciphertext.shape)
        return _MultiPlainText(pt_handles, ciphertext.shape)

    def close(self):
        if hasattr(self, "_handle") and self._handle:
            ffi.client_close(self._handle)   # step 1: zeros SK in Go
            self._handle.close()             # step 2: DeleteHandle (frees cgo slot)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            try:
                self.close()
            except:
                pass


class _MultiPlainText:
    """Internal helper for multi-slot plaintext (multiple Go handles)."""

    def __init__(self, handles, shape):
        self.handles = handles
        self._shape = torch.Size(shape)

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return len(self.handles)

    def __del__(self):
        import sys as _sys
        if _sys and _sys.modules:
            for h in self.handles:
                try:
                    h.close()
                except Exception:
                    pass
