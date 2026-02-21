"""Tests for LinearTransform serialization/deserialization roundtrip via Go FFI.

These tests exercise the old Lattigo backend directly and will be removed
in Phase 6 when the old backend is deleted. They are skipped when the
orionclient shared library is already loaded (two Go runtimes conflict).
"""

import gc

import pytest

from orion import CKKSParams
from orion.compiled_model import CompiledModel
from orion.backend.lattigo import bindings as lgo
from orion.backend.python import parameters

# Skip all tests in this module -- the old Lattigo backend cannot coexist
# with the orionclient library when loaded in the same process.
# These tests will be removed in Phase 6.
pytestmark = pytest.mark.skip(reason="old backend tests, removed in Phase 6")


PARAMS = CKKSParams(
    logn=14,
    logq=(45, 30, 30, 30, 30, 45),
    logp=(50, 51),
    logscale=30,
    h=192,
    ring_type="conjugate_invariant",
)


def _make_backend():
    """Create the old Lattigo backend directly (bypassing Client)."""
    new_params = parameters.NewParameters.from_ckks_params(PARAMS)
    backend = lgo.LattigoLibrary()
    backend.setup_bindings(new_params)
    return backend


def test_serialize_deserialize_roundtrip():
    """Create a LinearTransform, serialize it, deserialize it, and verify
    the loaded transform has the same Galois elements as the original."""
    backend = _make_backend()

    try:
        max_slots = backend.GetMaxSlots()

        diag_idxs = [0, 1]
        diag_data = []
        for _ in diag_idxs:
            diag_data.extend([1.0] * max_slots)

        level = 4
        bsgs_ratio = 1.0

        orig_id = backend.GenerateLinearTransform(
            diag_idxs, diag_data, level, bsgs_ratio
        )

        orig_gal_els = backend.GetLinearTransformRotationKeys(orig_id)
        assert len(orig_gal_els) > 0

        serialized_data, c_ptr = backend.SerializeLinearTransform(orig_id)
        assert len(serialized_data) > 0

        loaded_id = backend.LoadLinearTransform(serialized_data)
        backend.FreeCArray(c_ptr)

        loaded_gal_els = backend.GetLinearTransformRotationKeys(loaded_id)

        assert sorted(orig_gal_els) == sorted(loaded_gal_els), (
            f"Galois elements mismatch: orig={sorted(orig_gal_els)}, "
            f"loaded={sorted(loaded_gal_els)}"
        )

        backend.DeleteLinearTransform(orig_id)
        backend.DeleteLinearTransform(loaded_id)

    finally:
        gc.collect()


def test_serialize_deserialize_evaluate():
    """Serialize/deserialize a LinearTransform, then verify the deserialized
    transform can be used to evaluate a ciphertext and produce correct results."""
    backend = _make_backend()

    try:
        max_slots = backend.GetMaxSlots()

        # Set up Go evaluator (needed for EvaluateLinearTransform)
        backend.NewEvaluator()

        diag_idxs = [0]
        diag_data = [2.0] * max_slots

        level = 4
        bsgs_ratio = 1.0

        orig_id = backend.GenerateLinearTransform(
            diag_idxs, diag_data, level, bsgs_ratio
        )

        serialized_data, c_ptr = backend.SerializeLinearTransform(orig_id)
        loaded_id = backend.LoadLinearTransform(serialized_data)
        backend.FreeCArray(c_ptr)

        gal_els = backend.GetLinearTransformRotationKeys(loaded_id)
        for gal_el in gal_els:
            backend.GenerateLinearTransformRotationKey(gal_el)

        moduli_chain = backend.GetModuliChain()
        input_scale = moduli_chain[level]

        input_data = [float(i) for i in range(max_slots)]
        ptxt_id = backend.Encode(input_data, level, input_scale)
        ctxt_id = backend.Encrypt(ptxt_id)

        result_ctxt_id = backend.EvaluateLinearTransform(loaded_id, ctxt_id)

        result_ptxt_id = backend.Decrypt(result_ctxt_id)
        result_data = backend.Decode(result_ptxt_id)

        for i in range(min(16, max_slots)):
            expected = input_data[i] * 2.0
            actual = result_data[i]
            assert abs(actual - expected) < 0.1, (
                f"Slot {i}: expected ~{expected}, got {actual}"
            )

        backend.DeleteLinearTransform(orig_id)
        backend.DeleteLinearTransform(loaded_id)
        backend.DeleteCiphertext(ctxt_id)
        backend.DeleteCiphertext(result_ctxt_id)
        backend.DeletePlaintext(ptxt_id)
        backend.DeletePlaintext(result_ptxt_id)

    finally:
        gc.collect()


def test_serialized_size_reasonable():
    """Verify that serialized size is reasonable."""
    backend = _make_backend()

    try:
        max_slots = backend.GetMaxSlots()

        diag_idxs = [0, 1, -1]
        diag_data = []
        for _ in diag_idxs:
            diag_data.extend([1.0] * max_slots)

        level = 4
        bsgs_ratio = 1.0

        lt_id = backend.GenerateLinearTransform(
            diag_idxs, diag_data, level, bsgs_ratio
        )

        serialized_data, c_ptr = backend.SerializeLinearTransform(lt_id)
        backend.FreeCArray(c_ptr)

        assert len(serialized_data) > 1000, (
            f"Serialized data too small: {len(serialized_data)} bytes"
        )

        backend.DeleteLinearTransform(lt_id)

    finally:
        gc.collect()
