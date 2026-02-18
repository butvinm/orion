"""Tests for LinearTransform serialization/deserialization roundtrip via Go FFI."""

import numpy as np

from orion.core.orion import scheme


# Small CKKS params sufficient for testing serialization
TEST_CONFIG = {
    "ckks_params": {
        "LogN": 14,
        "LogQ": [45, 30, 30, 30, 30, 45],
        "LogP": [50, 51],
        "LogScale": 30,
        "H": 192,
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


def test_serialize_deserialize_roundtrip():
    """Create a LinearTransform, serialize it, deserialize it, and verify
    the loaded transform has the same Galois elements as the original."""
    scheme.init_scheme(TEST_CONFIG)

    try:
        backend = scheme.backend
        max_slots = backend.GetMaxSlots()

        # Create simple diagonal data: identity-like transform with 2 diagonals
        diag_idxs = [0, 1]
        diag_data = []
        for _ in diag_idxs:
            diag_data.extend([1.0] * max_slots)

        level = 4
        bsgs_ratio = 1.0

        # Generate the linear transform (encodes diagonals)
        orig_id = backend.GenerateLinearTransform(
            diag_idxs, diag_data, level, bsgs_ratio, "none"
        )

        # Get Galois elements from original
        orig_gal_els = backend.GetLinearTransformRotationKeys(orig_id)
        assert len(orig_gal_els) > 0, "Original transform should have Galois elements"

        # Serialize
        serialized_data, c_ptr = backend.SerializeLinearTransform(orig_id)
        assert len(serialized_data) > 0, "Serialized data should be non-empty"

        # Deserialize into a new LinearTransform on the Go heap
        loaded_id = backend.LoadLinearTransform(serialized_data)

        # Free the C memory from serialization
        backend.FreeCArray(c_ptr)

        # Get Galois elements from loaded transform
        loaded_gal_els = backend.GetLinearTransformRotationKeys(loaded_id)

        # Galois elements should match
        assert sorted(orig_gal_els) == sorted(loaded_gal_els), (
            f"Galois elements mismatch: orig={sorted(orig_gal_els)}, "
            f"loaded={sorted(loaded_gal_els)}"
        )

        # Clean up
        backend.DeleteLinearTransform(orig_id)
        backend.DeleteLinearTransform(loaded_id)

    finally:
        scheme.delete_scheme()


def test_serialize_deserialize_evaluate():
    """Serialize/deserialize a LinearTransform, then verify the deserialized
    transform can be used to evaluate a ciphertext and produce correct results."""
    scheme.init_scheme(TEST_CONFIG)

    try:
        backend = scheme.backend
        max_slots = backend.GetMaxSlots()

        # Create a simple diagonal: only diagonal 0 (identity-like)
        diag_idxs = [0]
        # Scale by 2.0 so we can verify the transform effect
        diag_data = [2.0] * max_slots

        level = 4
        bsgs_ratio = 1.0

        # Generate and encode the transform
        orig_id = backend.GenerateLinearTransform(
            diag_idxs, diag_data, level, bsgs_ratio, "none"
        )

        # Serialize
        serialized_data, c_ptr = backend.SerializeLinearTransform(orig_id)

        # Deserialize
        loaded_id = backend.LoadLinearTransform(serialized_data)
        backend.FreeCArray(c_ptr)

        # Generate rotation keys needed for this transform
        gal_els = backend.GetLinearTransformRotationKeys(loaded_id)
        for gal_el in gal_els:
            backend.GenerateLinearTransformRotationKey(gal_el)

        # Create input data and encrypt
        # Use the moduli chain to get the correct scale for the level
        moduli_chain = backend.GetModuliChain()
        input_scale = moduli_chain[level]

        input_data = [float(i) for i in range(max_slots)]
        ptxt_id = backend.Encode(input_data, level, input_scale)
        ctxt_id = backend.Encrypt(ptxt_id)

        # Evaluate with the deserialized transform
        result_ctxt_id = backend.EvaluateLinearTransform(loaded_id, ctxt_id)

        # Decrypt and decode
        result_ptxt_id = backend.Decrypt(result_ctxt_id)
        result_data = backend.Decode(result_ptxt_id)

        # Verify: each slot should be approximately 2x the input
        for i in range(min(16, max_slots)):
            expected = input_data[i] * 2.0
            actual = result_data[i]
            assert abs(actual - expected) < 0.1, (
                f"Slot {i}: expected ~{expected}, got {actual}"
            )

        # Clean up
        backend.DeleteLinearTransform(orig_id)
        backend.DeleteLinearTransform(loaded_id)
        backend.DeleteCiphertext(ctxt_id)
        backend.DeleteCiphertext(result_ctxt_id)
        backend.DeletePlaintext(ptxt_id)
        backend.DeletePlaintext(result_ptxt_id)

    finally:
        scheme.delete_scheme()


def test_serialized_size_reasonable():
    """Verify that serialized size is reasonable (non-trivially large for
    encoded diagonals, but not absurdly so)."""
    scheme.init_scheme(TEST_CONFIG)

    try:
        backend = scheme.backend
        max_slots = backend.GetMaxSlots()

        diag_idxs = [0, 1, -1]
        diag_data = []
        for _ in diag_idxs:
            diag_data.extend([1.0] * max_slots)

        level = 4
        bsgs_ratio = 1.0

        lt_id = backend.GenerateLinearTransform(
            diag_idxs, diag_data, level, bsgs_ratio, "none"
        )

        serialized_data, c_ptr = backend.SerializeLinearTransform(lt_id)
        backend.FreeCArray(c_ptr)

        # Serialized data should contain metadata + 3 encoded polynomials.
        # Each polynomial at level 4 with LogP primes should be substantial.
        assert len(serialized_data) > 1000, (
            f"Serialized data too small: {len(serialized_data)} bytes"
        )

        backend.DeleteLinearTransform(lt_id)

    finally:
        scheme.delete_scheme()
