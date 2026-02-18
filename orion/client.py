"""Client: key generation, encode/decode, encrypt/decrypt.

Has secret key. Produces EvalKeys for the Evaluator.
"""

import numpy as np
import torch

from orion.params import CKKSParams
from orion.compiled_model import KeyManifest, EvalKeys
from orion.backend.lattigo import bindings as lgo
from orion.backend.python import parameters, key_generator, encoder, encryptor


class PlainText:
    """Wrapper around backend plaintext IDs.

    Local-only -- no cross-process serialization.
    """

    def __init__(self, ptxt_ids, shape, backend, encoder_ref):
        self.ids = [ptxt_ids] if isinstance(ptxt_ids, int) else ptxt_ids
        self.shape = shape
        self._backend = backend
        self._encoder = encoder_ref

    def decode(self, client):
        """Decode this plaintext via the client's encoder."""
        return client.decode(self)

    def __len__(self):
        return len(self.ids)


class CipherText:
    """Wrapper around backend ciphertext IDs with serialization support.

    Supports to_bytes()/from_bytes() for cross-process transfer.
    """

    def __init__(self, ctxt_ids, shape, backend):
        self.ids = [ctxt_ids] if isinstance(ctxt_ids, int) else ctxt_ids
        self.shape = shape
        self._backend = backend

    def to_bytes(self) -> bytes:
        """Serialize all ciphertext slots into a single bytes blob.

        Format: [4 bytes num_cts] + for each ct: [4 bytes shape_len] + shape +
                [8 bytes ct_len] + ct_data
        """
        import struct

        parts = [struct.pack("<I", len(self.ids))]
        # Store shape
        shape_list = list(self.shape)
        parts.append(struct.pack("<I", len(shape_list)))
        for dim in shape_list:
            parts.append(struct.pack("<i", dim))

        for ctxt_id in self.ids:
            ct_data, ptr = self._backend.SerializeCiphertext(ctxt_id)
            ct_bytes = bytes(ct_data)
            self._backend.FreeCArray(ptr)
            parts.append(struct.pack("<Q", len(ct_bytes)))
            parts.append(ct_bytes)

        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes, backend) -> "CipherText":
        """Deserialize ciphertexts from bytes. Backend must have same params."""
        import struct

        offset = 0
        (num_cts,) = struct.unpack_from("<I", data, offset)
        offset += 4

        (shape_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        shape_dims = []
        for _ in range(shape_len):
            (dim,) = struct.unpack_from("<i", data, offset)
            offset += 4
            shape_dims.append(dim)
        shape = torch.Size(shape_dims)

        ctxt_ids = []
        for _ in range(num_cts):
            (ct_len,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            ct_bytes = data[offset : offset + ct_len]
            offset += ct_len
            ct_arr = np.frombuffer(ct_bytes, dtype=np.uint8)
            ctxt_id = backend.LoadCiphertext(ct_arr)
            ctxt_ids.append(ctxt_id)

        return cls(ctxt_ids, shape, backend)

    def __len__(self):
        return len(self.ids)


class Client:
    """Client-side operations: key generation, encode/decode, encrypt/decrypt.

    Usage:
        client = Client(compiled.params)
        keys = client.generate_keys(compiled.manifest)
        pt = client.encode(tensor, level=compiled.input_level)
        ct = client.encrypt(pt)
        ...
        result = client.decode(client.decrypt(ct_result))

    To recreate a Client with the same secret key (required when the Go
    backend singleton was destroyed between encrypt and decrypt):

        sk_bytes = client.secret_key
        ...
        client2 = Client(params, secret_key=sk_bytes)
    """

    def __init__(self, params: CKKSParams, secret_key: bytes | None = None):
        self.ckks_params = params
        self.params = parameters.NewParameters.from_ckks_params(params)

        # Initialize Go backend
        self.backend = lgo.LattigoLibrary()
        self.backend.setup_bindings(self.params)

        if secret_key is not None:
            # Restore from serialized secret key
            self.backend.NewKeyGenerator()
            sk_arr = np.frombuffer(secret_key, dtype=np.uint8)
            self.backend.LoadSecretKey(sk_arr)
            self.backend.GeneratePublicKey()
            self.backend.GenerateRelinearizationKey()
            self.backend.GenerateEvaluationKeys()
        else:
            # Full key generation (generates new SK)
            self._keygen = key_generator.NewKeyGenerator(self)

        self._encoder = encoder.NewEncoder(self)
        self._encryptor = encryptor.NewEncryptor(self)

    @property
    def secret_key(self) -> bytes:
        """Serialize the secret key for later restoration."""
        sk_data, ptr = self.backend.SerializeSecretKey()
        sk_bytes = bytes(sk_data)
        self.backend.FreeCArray(ptr)
        return sk_bytes

    def generate_keys(self, manifest: KeyManifest) -> EvalKeys:
        """Generate and serialize all evaluation keys specified by manifest."""
        keys = EvalKeys()

        # RLK
        if manifest.needs_rlk:
            rlk_data, ptr = self.backend.SerializeRelinKey()
            keys.rlk_data = bytes(rlk_data)
            self.backend.FreeCArray(ptr)

        # Galois (rotation) keys
        for gal_el in sorted(manifest.galois_elements):
            key_data, ptr = self.backend.GenerateAndSerializeRotationKey(
                gal_el
            )
            keys.galois_keys[gal_el] = bytes(key_data)
            self.backend.FreeCArray(ptr)

        # Bootstrap keys
        if manifest.bootstrap_slots:
            logp = list(manifest.boot_logp)
            for slot_count in manifest.bootstrap_slots:
                key_data, ptr = self.backend.SerializeBootstrapKeys(
                    slot_count, logp
                )
                keys.bootstrap_keys[slot_count] = bytes(key_data)
                self.backend.FreeCArray(ptr)

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
            level = self.params.get_max_level()
        if scale is None:
            scale = self.params.get_default_scale()

        num_slots = self.params.get_slots()
        num_elements = tensor.numel()

        tensor = tensor.cpu()
        pad_length = (-num_elements) % num_slots
        vector = torch.zeros(num_elements + pad_length)
        vector[:num_elements] = tensor.flatten()
        num_plaintexts = len(vector) // num_slots

        ptxt_ids = []
        for i in range(num_plaintexts):
            to_encode = vector[i * num_slots : (i + 1) * num_slots].tolist()
            ptxt_id = self.backend.Encode(to_encode, level, scale)
            ptxt_ids.append(ptxt_id)

        return PlainText(ptxt_ids, tensor.shape, self.backend, self._encoder)

    def decode(self, plaintext):
        """Decode a PlainText back to a tensor."""
        values = []
        for ptxt_id in plaintext.ids:
            values.extend(self.backend.Decode(ptxt_id))

        values = torch.tensor(values)[: plaintext.shape.numel()]
        return values.reshape(plaintext.shape)

    def encrypt(self, plaintext):
        """Encrypt a PlainText into a CipherText."""
        ctxt_ids = []
        for ptxt_id in plaintext.ids:
            ctxt_id = self.backend.Encrypt(ptxt_id)
            ctxt_ids.append(ctxt_id)

        return CipherText(ctxt_ids, plaintext.shape, self.backend)

    def decrypt(self, ciphertext):
        """Decrypt a CipherText into a PlainText."""
        ptxt_ids = []
        for ctxt_id in ciphertext.ids:
            ptxt_id = self.backend.Decrypt(ctxt_id)
            ptxt_ids.append(ptxt_id)

        return PlainText(ptxt_ids, ciphertext.shape, self.backend, self._encoder)

    def __del__(self):
        if hasattr(self, "backend") and self.backend:
            try:
                self.backend.DeleteScheme()
            except Exception:
                pass
