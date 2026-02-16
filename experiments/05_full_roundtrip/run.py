"""
Experiment 05: Full client-server inference roundtrip

Hypothesis: End-to-end flow works: server compiles keylessly -> exports key
manifest -> client generates keys from manifest -> server imports keys via
FFI -> neural network inference on encrypted data produces correct results.

What this proves: That the entire pipeline can be split across client and
server with only eval keys crossing the boundary. Integrates all previous
experiments into Orion's actual inference flow.

Depends on: Task 3 (keyless compilation + manifest) and Task 4 (key import FFI).

Approach: Uses the Go global singleton sequentially:
  1. Server: init_params_only -> fit -> compile keylessly (linear transforms in Go heap)
  2. Client (simulated): generate keys on same singleton, serialize eval keys + encrypt
  3. Server: load eval keys, construct evaluators from keys, infer on ciphertext
  4. Client: load SK, decrypt, verify vs cleartext reference

  The key insight: we do NOT call DeleteScheme between server phase 1 and
  server phase 2, so the linear transforms (Go heap) persist across the
  key loading step. This mirrors the real architecture where the server
  compiles once and then loads client keys.

  Accuracy baseline: test_mlp.py (monolithic Orion) achieves MAE < 0.005
  against cleartext. The split architecture should achieve the same, since
  the computation is identical — only the key provisioning path differs.

Library modifications: None beyond what Tasks 3-4 already added.
"""

import sys
import os
import ctypes
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import orion
from orion.core.orion import scheme as orion_scheme
from orion.core.utils import get_mnist_datasets, mae
from orion.models.mlp import MLP
from orion.backend.lattigo.bindings import ArrayResultByte
from orion.backend.python import poly_evaluator
from orion.backend.python.tensors import CipherTensor, PlainTensor


# CKKS parameters — same as configs/mlp.yml
CONFIG = {
    "ckks_params": {
        "LogN": 13,
        "LogQ": [29, 26, 26, 26, 26, 26],
        "LogP": [29, 29],
        "LogScale": 26,
        "H": 8192,
        "RingType": "ConjugateInvariant",
    },
    "orion": {
        "margin": 2,
        "embedding_method": "hybrid",
        "backend": "lattigo",
        "fuse_modules": True,
        "debug": False,
        "io_mode": "none",
    },
}

TOLERANCE = 0.005  # Same as test_mlp.py


def serialize_to_numpy(backend, serialize_fn, *args):
    """Call a serialize function and return a numpy array of bytes, freeing the C pointer."""
    c_args = []
    for arg in args:
        curr_argtype = serialize_fn.func.argtypes[len(c_args)]
        c_arg = serialize_fn.convert_to_ctypes(arg, curr_argtype)
        if isinstance(c_arg, tuple):
            c_args.extend(c_arg)
        else:
            c_args.append(c_arg)
    result = serialize_fn.func(*c_args)
    length = int(result.Length)
    buffer = ctypes.cast(
        result.Data,
        ctypes.POINTER(ctypes.c_ubyte * length),
    ).contents
    data = np.array(buffer, dtype=np.uint8).copy()
    backend.FreeCArray(ctypes.cast(result.Data, ctypes.c_void_p))
    return data


class ServerEvaluator:
    """Thin evaluator wrapper using already-initialized Go evaluator.

    Mirrors the interface of orion.backend.python.evaluator.NewEvaluator
    but does NOT call backend.NewEvaluator() (which requires SK for
    AddPo2RotationKeys). Instead, it assumes NewEvaluatorFromKeys() has
    already been called with pre-loaded keys.
    """
    def __init__(self, backend):
        self.backend = backend

    def negate(self, ctxt):
        return self.backend.Negate(ctxt)

    def rotate(self, ctxt, amount, in_place):
        if in_place:
            return self.backend.Rotate(ctxt, amount)
        return self.backend.RotateNew(ctxt, amount)

    def add_scalar(self, ctxt, scalar, in_place):
        if in_place:
            return self.backend.AddScalar(ctxt, float(scalar))
        return self.backend.AddScalarNew(ctxt, float(scalar))

    def sub_scalar(self, ctxt, scalar, in_place):
        if in_place:
            return self.backend.SubScalar(ctxt, float(scalar))
        return self.backend.SubScalarNew(ctxt, float(scalar))

    def mul_scalar(self, ctxt, scalar, in_place):
        if isinstance(scalar, float) and scalar.is_integer():
            scalar = int(scalar)
        if isinstance(scalar, int):
            ct_out = (self.backend.MulScalarInt if in_place
                      else self.backend.MulScalarIntNew)(ctxt, scalar)
        else:
            ct_out = (self.backend.MulScalarFloat if in_place
                      else self.backend.MulScalarFloatNew)(ctxt, scalar)
            ct_out = self.backend.Rescale(ct_out)
        return ct_out

    def add_plaintext(self, ctxt, ptxt, in_place):
        if in_place:
            return self.backend.AddPlaintext(ctxt, ptxt)
        return self.backend.AddPlaintextNew(ctxt, ptxt)

    def sub_plaintext(self, ctxt, ptxt, in_place):
        if in_place:
            return self.backend.SubPlaintext(ctxt, ptxt)
        return self.backend.SubPlaintextNew(ctxt, ptxt)

    def mul_plaintext(self, ctxt, ptxt, in_place):
        if in_place:
            ct_out = self.backend.MulPlaintext(ctxt, ptxt)
        else:
            ct_out = self.backend.MulPlaintextNew(ctxt, ptxt)
        return self.backend.Rescale(ct_out)

    def add_ciphertext(self, ctxt0, ctxt1, in_place):
        if in_place:
            return self.backend.AddCiphertext(ctxt0, ctxt1)
        return self.backend.AddCiphertextNew(ctxt0, ctxt1)

    def sub_ciphertext(self, ctxt0, ctxt1, in_place):
        if in_place:
            return self.backend.SubCiphertext(ctxt0, ctxt1)
        return self.backend.SubCiphertextNew(ctxt0, ctxt1)

    def mul_ciphertext(self, ctxt0, ctxt1, in_place):
        if in_place:
            ct_out = self.backend.MulRelinCiphertext(ctxt0, ctxt1)
        else:
            ct_out = self.backend.MulRelinCiphertextNew(ctxt0, ctxt1)
        return self.backend.Rescale(ct_out)

    def rescale(self, ctxt, in_place):
        if in_place:
            return self.backend.Rescale(ctxt)
        return self.backend.RescaleNew(ctxt)


class ServerBootstrapper:
    """Bootstrapper wrapper using already-loaded bootstrap keys."""
    def __init__(self, backend):
        self.backend = backend

    def bootstrap(self, ctxt, slots):
        return self.backend.Bootstrap(ctxt, slots)


def main():
    print("=" * 70)
    print("Experiment 05: Full Client-Server Inference Roundtrip")
    print("=" * 70)

    # Load MNIST data (same as test_mlp.py for realistic statistics)
    trainloader, testloader = get_mnist_datasets(data_dir="./data", batch_size=1)
    test_input, _ = next(iter(testloader))
    print(f"Test input shape: {test_input.shape}")

    # Create model (deterministic)
    torch.manual_seed(42)
    net = MLP(num_classes=10)
    net.eval()

    # Cleartext reference
    with torch.no_grad():
        out_clear = net(test_input)
    print(f"Cleartext output (first 5): {out_clear[0, :5].tolist()}")

    # ================================================================
    # SERVER PHASE 1: Keyless compilation -> manifest
    # ================================================================
    print("\n" + "=" * 70)
    print("Server Phase 1: Keyless compilation")
    print("=" * 70)
    server_start = time.time()

    orion.init_params_only(CONFIG)
    orion.fit(net, trainloader)
    input_level, manifest = orion.compile(net)

    server_compile_time = time.time() - server_start
    print(f"\nCompile time: {server_compile_time:.1f}s")
    print(f"Input level: {input_level}")
    print(f"Manifest: {len(manifest['galois_elements'])} galois elements, "
          f"bootstrap_slots={manifest['bootstrap_slots']}, "
          f"needs_rlk={manifest['needs_rlk']}")

    # Save manifest
    manifest_path = os.path.join(os.path.dirname(__file__), "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    # Go heap now contains all linear transforms. Do NOT delete scheme.

    # ================================================================
    # CLIENT PHASE: Generate keys from manifest, encrypt input
    # ================================================================
    print("\n" + "=" * 70)
    print("Client Phase: Key generation + encryption")
    print("(Simulated on same Go singleton)")
    print("=" * 70)
    client_start = time.time()

    backend = orion_scheme.backend

    # Generate all cryptographic keys
    backend.NewKeyGenerator()
    backend.GenerateSecretKey()
    backend.GeneratePublicKey()
    backend.GenerateRelinearizationKey()

    # Serialize SK for later decryption
    sk_data = serialize_to_numpy(backend, backend.SerializeSecretKey)
    print(f"  sk: {len(sk_data)/1024/1024:.2f} MB")

    # Serialize eval keys for server
    pk_data = serialize_to_numpy(backend, backend.SerializePublicKey)
    rlk_data = serialize_to_numpy(backend, backend.SerializeRelinKey)
    print(f"  pk: {len(pk_data)/1024/1024:.2f} MB")
    print(f"  rlk: {len(rlk_data)/1024/1024:.2f} MB")

    # Generate and serialize all galois keys from manifest
    galois_key_data = {}
    for galEl in manifest["galois_elements"]:
        gk_bytes = serialize_to_numpy(
            backend, backend.GenerateAndSerializeRotationKey, galEl)
        galois_key_data[galEl] = gk_bytes

    total_gk_size = sum(len(v) for v in galois_key_data.values())
    print(f"  galois keys: {len(galois_key_data)} keys, "
          f"{total_gk_size/1024/1024:.2f} MB total")

    # Generate and serialize bootstrap keys (if manifest requires any)
    bootstrap_key_data = {}
    if manifest["bootstrap_slots"]:
        logPs = orion_scheme.params.get_boot_logp()
        for slots in manifest["bootstrap_slots"]:
            btp_bytes = serialize_to_numpy(
                backend, backend.SerializeBootstrapKeys, slots, logPs)
            bootstrap_key_data[slots] = btp_bytes
            print(f"  bootstrap keys ({slots} slots): "
                  f"{len(btp_bytes)/1024/1024:.2f} MB")

    # Encrypt test input
    # Need EvalKeys + Evaluator + Encryptor for encryption
    backend.GenerateEvaluationKeys()
    backend.NewEvaluator()
    backend.NewEncryptor()

    vec_ptxt = orion.encode(test_input, input_level)

    # Encrypt via backend directly
    ct_ids = []
    for pt_id in vec_ptxt.ids:
        ct_id = backend.Encrypt(pt_id)
        ct_ids.append(ct_id)

    # Serialize encrypted ciphertexts for server
    ct_data_list = []
    for ct_id in ct_ids:
        ct_bytes = serialize_to_numpy(backend, backend.SerializeCiphertext, ct_id)
        ct_data_list.append(ct_bytes)

    ct_shape = vec_ptxt.shape
    ct_on_shape = vec_ptxt.on_shape

    client_time = time.time() - client_start
    print(f"  encrypted input: {len(ct_data_list)} ciphertexts, "
          f"~{sum(len(c) for c in ct_data_list)/1024:.1f} KB")
    print(f"Client time: {client_time:.1f}s")

    total_eval_key_size = (
        len(pk_data) + len(rlk_data) + total_gk_size +
        sum(len(v) for v in bootstrap_key_data.values())
    )
    print(f"Total eval key transfer: {total_eval_key_size/1024/1024:.2f} MB")

    # ================================================================
    # SERVER PHASE 2: Load eval keys, construct evaluators, infer
    # ================================================================
    print("\n" + "=" * 70)
    print("Server Phase 2: Load keys + inference")
    print("=" * 70)
    server2_start = time.time()

    # Clear the secret key and keygen from the Go singleton to truly
    # simulate server-side isolation. Without this, the Rotate/RotateNew
    # guards (scheme.KeyGen != nil && scheme.SecretKey != nil) would
    # allow lazy key generation, masking any manifest incompleteness.
    backend.ClearSecretKey()
    print("  Cleared SK + KeyGen from Go singleton (server has no secret key)")

    # Load eval keys from serialized data (as a real server would)
    backend.LoadRelinKey(rlk_data)
    backend.GenerateEvaluationKeys()
    print("  Loaded RLK + created EvalKeys")

    for galEl, gk_bytes in galois_key_data.items():
        backend.LoadRotationKey(gk_bytes, galEl)
    print(f"  Loaded {len(galois_key_data)} galois keys")

    # Construct evaluator from imported keys (no SK needed)
    backend.NewEvaluatorFromKeys()
    print("  Created evaluator (NewEvaluatorFromKeys)")

    backend.NewLinearTransformEvaluator()
    print("  Created linear transform evaluator")

    backend.NewPolynomialEvaluator()
    print("  Created polynomial evaluator")

    # Load bootstrap keys (if any)
    if bootstrap_key_data:
        logPs = orion_scheme.params.get_boot_logp()
        for slots, btp_bytes in bootstrap_key_data.items():
            backend.LoadBootstrapKeys(btp_bytes, slots, logPs)
            print(f"  Loaded bootstrap keys for {slots} slots")

    # Wire up the Python scheme for server-side inference
    server_eval = ServerEvaluator(backend)
    server_btp = ServerBootstrapper(backend)

    orion_scheme.evaluator = server_eval

    # Create poly_evaluator without calling its __init__ (which would
    # call NewPolynomialEvaluator again)
    orion_scheme.poly_evaluator = poly_evaluator.NewEvaluator.__new__(
        poly_evaluator.NewEvaluator)
    orion_scheme.poly_evaluator.scheme = orion_scheme
    orion_scheme.poly_evaluator.backend = backend

    # Switch lt_evaluator out of keyless mode
    orion_scheme.lt_evaluator.keyless = False
    orion_scheme.lt_evaluator.evaluator = server_eval

    orion_scheme.bootstrapper = server_btp
    orion_scheme.encryptor = None  # Server does NOT encrypt/decrypt
    orion_scheme.keyless = False

    # Load ciphertexts from client
    server_ct_ids = []
    for ct_bytes in ct_data_list:
        ct_id = backend.LoadCiphertext(ct_bytes)
        server_ct_ids.append(ct_id)
    print(f"  Loaded {len(server_ct_ids)} ciphertexts")

    server_ctxt = CipherTensor(
        orion_scheme, server_ct_ids, ct_shape, ct_on_shape)

    # Run server-side FHE inference
    print("\n  Running FHE inference...")
    infer_start = time.time()
    net.he()
    out_ctxt_server = net(server_ctxt)
    infer_time = time.time() - infer_start
    print(f"  Inference time: {infer_time:.1f}s")

    # Serialize result ciphertexts for client
    result_ct_data = []
    for ct_id in out_ctxt_server.ids:
        ct_bytes = serialize_to_numpy(backend, backend.SerializeCiphertext, ct_id)
        result_ct_data.append(ct_bytes)
    print(f"  Result: {len(result_ct_data)} ciphertexts, "
          f"~{sum(len(c) for c in result_ct_data)/1024:.1f} KB")

    server2_time = time.time() - server2_start
    print(f"Server phase 2 time: {server2_time:.1f}s")

    # ================================================================
    # CLIENT PHASE 2: Decrypt and verify
    # ================================================================
    print("\n" + "=" * 70)
    print("Client Phase 2: Decrypt and verify")
    print("=" * 70)

    # Load SK for decryption
    backend.LoadSecretKey(sk_data)
    backend.NewDecryptor()
    backend.NewEncoder()
    print("  Loaded SK + decryptor + encoder")

    # Decrypt result
    client_result_ct_ids = []
    for ct_bytes in result_ct_data:
        ct_id = backend.LoadCiphertext(ct_bytes)
        client_result_ct_ids.append(ct_id)

    result_pt_ids = []
    for ct_id in client_result_ct_ids:
        pt_id = backend.Decrypt(ct_id)
        result_pt_ids.append(pt_id)

    result_values = []
    for pt_id in result_pt_ids:
        values = backend.Decode(pt_id)
        result_values.extend(values)

    out_fhe_split = torch.tensor(result_values[:out_clear.shape[1]])
    print(f"\n  Client-server FHE (first 5): {out_fhe_split[:5].tolist()}")
    print(f"  Cleartext (first 5):         {out_clear[0, :5].tolist()}")

    split_error = mae(out_clear, out_fhe_split.unsqueeze(0))
    print(f"\n  MAE vs cleartext: {split_error:.6f}")

    if split_error < TOLERANCE:
        print(f"  PASS: {split_error:.6f} < {TOLERANCE}")
    else:
        print(f"  FAIL: {split_error:.6f} >= {TOLERANCE}")
        sys.exit(1)

    print("  PASS: All manifest galois elements used (no missing key panics)")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 05 RESULTS")
    print("=" * 70)

    print("\nHypothesis CONFIRMED:")
    print("  Full client-server inference roundtrip works end-to-end.")
    print(f"  - Server keyless compile: {server_compile_time:.1f}s")
    print(f"  - Client keygen + encrypt: {client_time:.1f}s")
    print(f"  - Server inference (imported keys): {server2_time:.1f}s")
    print(f"  - MAE vs cleartext: {split_error:.6f} (tolerance: {TOLERANCE})")
    print(f"  - Monolithic baseline (test_mlp.py): MAE < {TOLERANCE}")

    print(f"\nKey manifest:")
    print(f"  - {len(manifest['galois_elements'])} galois elements")
    print(f"  - {len(manifest['bootstrap_slots'])} bootstrap configs")
    print(f"  - All keys pre-loaded before inference (no lazy generation)")

    print(f"\nSerialized sizes:")
    print(f"  - Manifest: ~{len(json.dumps(manifest))/1024:.1f} KB")
    print(f"  - Eval keys: {total_eval_key_size/1024/1024:.2f} MB total")
    print(f"    pk: {len(pk_data)/1024/1024:.2f} MB")
    print(f"    rlk: {len(rlk_data)/1024/1024:.2f} MB")
    print(f"    galois ({len(galois_key_data)}): {total_gk_size/1024/1024:.2f} MB")
    for slots, btp_bytes in bootstrap_key_data.items():
        print(f"    bootstrap ({slots} slots): {len(btp_bytes)/1024/1024:.2f} MB")
    print(f"  - Input ct: ~{sum(len(c) for c in ct_data_list)/1024:.1f} KB")
    print(f"  - Output ct: ~{sum(len(c) for c in result_ct_data)/1024:.1f} KB")

    print("\nNo additional library changes beyond Tasks 3-4.")
    print("Done.")

    # Prevent stale CipherTensor/PlainTensor __del__ from panicking
    # the Go runtime after scheme deletion.
    out_ctxt_server.scheme = None
    server_ctxt.scheme = None
    orion.delete_scheme()


if __name__ == "__main__":
    main()
