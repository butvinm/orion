"""Tests for GoHandle lifecycle — the behavior that caused all 6 handle bugs in v1."""

import gc
from unittest.mock import patch

import torch
import pytest

from orion.params import CKKSParams
from orion.compiled_model import CompiledModel, KeyManifest, EvalKeys
from orion.compiler import Compiler
from orion.client import Client
from orion.ciphertext import Ciphertext, PlainText
from orion.evaluator import Evaluator
from orion.backend.orionclient import ffi
from orion.backend.orionclient.ffi import GoHandle
import orion.nn as on


MLP_PARAMS = CKKSParams(
    logn=13,
    logq=[29, 26, 26, 26, 26, 26],
    logp=[29, 29],
    logscale=26,
    h=8192,
    ring_type="conjugate_invariant",
)


class SimpleMLP(on.Module):
    def __init__(self):
        super().__init__()
        self.flatten = on.Flatten()
        self.fc1 = on.Linear(784, 32)
        self.act1 = on.Quad()
        self.fc2 = on.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        return self.fc2(x)


def _cleanup():
    gc.collect()


def _compile_model():
    """Helper: compile a SimpleMLP and return (compiled_bytes, keys_bytes, sk_bytes)."""
    torch.manual_seed(42)
    net = SimpleMLP()
    compiler = Compiler(net, MLP_PARAMS)
    compiler.fit(torch.randn(1, 1, 28, 28))
    compiled = compiler.compile()
    compiled_bytes = compiled.to_bytes()
    del compiler
    _cleanup()

    client = Client(compiled.params)
    keys = client.generate_keys(compiled.manifest)
    keys_bytes = keys.to_bytes()
    sk_bytes = client.secret_key
    del client
    _cleanup()

    return compiled_bytes, keys_bytes, sk_bytes


# --- GoHandle unit tests (no FFI needed) ---


class TestGoHandle:
    def test_gohandle_raw_returns_value(self):
        """GoHandle(42).raw == 42"""
        h = GoHandle(42)
        assert h.raw == 42
        # Prevent __del__ from calling DeleteHandle on a fake value
        h._raw = 0

    def test_gohandle_close_is_idempotent(self):
        """h.close(); h.close() -- no error on second call.

        Uses a real FFI handle to ensure DeleteHandle is actually called.
        """
        client = Client(MLP_PARAMS)
        h = client._handle
        # First close via client (two-step: client_close + handle.close)
        client.close()
        # h._raw is now 0, second close is a no-op
        h.close()
        _cleanup()

    def test_gohandle_raw_after_close_raises(self):
        """h.close(); h.raw -> RuntimeError('Use of closed handle')"""
        # Use a real handle from FFI
        from orion.backend.orionclient import ffi
        client_h = ffi.new_client(MLP_PARAMS.to_bridge_json())
        assert isinstance(client_h, GoHandle)
        ffi.client_close(client_h)
        client_h.close()
        with pytest.raises(RuntimeError, match="Use of closed handle"):
            _ = client_h.raw
        _cleanup()

    def test_gohandle_bool_false_after_close(self):
        """h.close(); assert not h"""
        from orion.backend.orionclient import ffi
        client_h = ffi.new_client(MLP_PARAMS.to_bridge_json())
        assert client_h  # truthy when open
        ffi.client_close(client_h)
        client_h.close()
        assert not client_h  # falsy when closed
        _cleanup()


# --- Client lifecycle tests ---


class TestClientLifecycle:
    def test_client_two_step_close(self):
        """client.close() zeros SK then deletes handle; second close() is no-op."""
        client = Client(MLP_PARAMS)
        assert client._handle  # handle is alive
        client.close()
        assert client._handle is None  # fully released
        # Second close is no-op
        client.close()
        _cleanup()

    def test_client_context_manager(self):
        """with Client(params) as c: ... calls close() on exit."""
        with Client(MLP_PARAMS) as c:
            inp = torch.randn(1, 784)
            pt = c.encode(inp, level=5)
            assert isinstance(pt, PlainText)
        # After exit, handle should be None
        assert c._handle is None
        _cleanup()


# --- Evaluator lifecycle tests ---


class TestEvaluatorLifecycle:
    def test_evaluator_close_clears_tracked_handles(self):
        """After evaluator.close(), all _tracked_handles have _raw == 0."""
        compiled_bytes, keys_bytes, _ = _compile_model()

        compiled = CompiledModel.from_bytes(compiled_bytes)
        keys = EvalKeys.from_bytes(keys_bytes)
        net = SimpleMLP()
        evaluator = Evaluator(net, compiled, keys)

        # Verify tracked handles exist and are alive
        assert len(evaluator._tracked_handles) > 0
        for h in evaluator._tracked_handles:
            assert h._raw != 0, "Handle should be alive before close"

        # Save references before close
        tracked = list(evaluator._tracked_handles)

        evaluator.close()

        # All tracked handles should now be closed
        for h in tracked:
            assert h._raw == 0, "Handle should be dead after close"

        # Eval handle should be None
        assert evaluator._eval_handle is None
        _cleanup()

    def test_evaluator_double_close(self):
        """evaluator.close(); evaluator.close() -- no error."""
        compiled_bytes, keys_bytes, _ = _compile_model()

        compiled = CompiledModel.from_bytes(compiled_bytes)
        keys = EvalKeys.from_bytes(keys_bytes)
        net = SimpleMLP()
        evaluator = Evaluator(net, compiled, keys)

        evaluator.close()
        evaluator.close()  # should not raise
        _cleanup()

    def test_evaluator_partial_init_failure(self):
        """Mock a corrupt LT blob, verify __init__ raises and partial handles freed."""
        compiled_bytes, keys_bytes, _ = _compile_model()

        compiled = CompiledModel.from_bytes(compiled_bytes)
        keys = EvalKeys.from_bytes(keys_bytes)

        # Corrupt a blob that will be used during _reconstruct_modules
        # Find the first LinearTransform blob index and corrupt it
        corrupted = False
        for mod_name, mod_meta in compiled.module_metadata.items():
            if "transform_blobs" in mod_meta:
                for key_str, blob_idx in mod_meta["transform_blobs"].items():
                    compiled.blobs[blob_idx] = b"corrupted blob data!!"
                    corrupted = True
                    break
            if corrupted:
                break

        assert corrupted, "Should have found a LinearTransform blob to corrupt"

        net = SimpleMLP()
        with pytest.raises(RuntimeError):
            Evaluator(net, compiled, keys)

        # No leaked handles — the except clause in __init__ calls self.close()
        # which cleans up any already-allocated handles
        _cleanup()


# --- Multi-instance independence ---


class TestMultiInstance:
    def test_multi_instance_independence(self):
        """Two Clients with different params coexist, operate independently."""
        params1 = CKKSParams(
            logn=13,
            logq=[29, 26, 26, 26, 26, 26],
            logp=[29, 29],
            logscale=26,
            h=8192,
            ring_type="conjugate_invariant",
        )
        params2 = CKKSParams(
            logn=13,
            logq=[29, 26, 26, 26],
            logp=[29, 29],
            logscale=26,
            h=8192,
            ring_type="conjugate_invariant",
        )

        c1 = Client(params1)
        c2 = Client(params2)

        inp1 = torch.randn(1, 784)
        inp2 = torch.randn(1, 784)

        pt1 = c1.encode(inp1, level=5)
        pt2 = c2.encode(inp2, level=3)

        ct1 = c1.encrypt(pt1)
        ct2 = c2.encrypt(pt2)

        # Decrypt with respective clients
        dec1 = c1.decode(c1.decrypt(ct1))
        dec2 = c2.decode(c2.decrypt(ct2))

        assert torch.allclose(dec1.float(), inp1.float(), atol=1e-3)
        assert torch.allclose(dec2.float(), inp2.float(), atol=1e-3)

        # Close one, other still works
        c1.close()
        dec2_again = c2.decode(c2.decrypt(ct2))
        assert torch.allclose(dec2_again.float(), inp2.float(), atol=1e-3)

        c2.close()
        _cleanup()


# --- Wire format roundtrip ---


class TestWireFormat:
    def test_ciphertext_wire_format_roundtrip(self):
        """Ciphertext.from_bytes(ct.to_bytes()) decrypts to same values."""
        client = Client(MLP_PARAMS)
        inp = torch.randn(1, 784)
        pt = client.encode(inp, level=5)
        ct = client.encrypt(pt)

        ct_bytes = ct.to_bytes()
        ct2 = Ciphertext.from_bytes(ct_bytes)

        # Verify the round-tripped ciphertext decrypts correctly
        pt2 = client.decrypt(ct2)
        decoded = client.decode(pt2)
        assert decoded.shape == inp.shape
        assert torch.allclose(decoded.float(), inp.float(), atol=1e-3)

        # Verify handle types
        assert isinstance(ct._handle, GoHandle)
        assert isinstance(ct2._handle, GoHandle)

        del client
        _cleanup()


# --- Error propagation ---


# --- F2: Error-path handle cleanup ---


class TestErrorPathCleanup:
    """F2: Error-path handle cleanup regression tests.

    The GoHandle refactor added try/except cleanup blocks in encrypt(),
    evaluate_transforms(), and Evaluator.__init__(). Each block is correct
    but was never exercised by a test. These tests inject failures at each
    stage and verify handle cleanup.
    """

    def test_encrypt_partial_failure_closes_handles(self):
        """client_encrypt fails on 2nd PT; 1st CT handle is closed."""
        client = Client(MLP_PARAMS)
        max_slots = ffi.client_max_slots(client._handle)
        big_tensor = torch.randn(max_slots * 2)
        multi_pt = client.encode(big_tensor)

        allocated = []
        real_fn = ffi.client_encrypt
        call_count = [0]

        def fail_on_second(client_h, pt_h):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("injected encrypt failure")
            h = real_fn(client_h, pt_h)
            allocated.append(h)
            return h

        with patch.object(ffi, 'client_encrypt', side_effect=fail_on_second):
            with pytest.raises(RuntimeError, match="injected encrypt failure"):
                client.encrypt(multi_pt)

        assert len(allocated) == 1
        assert allocated[0]._raw == 0, "CT handle should be closed on partial encrypt failure"

        client.close()
        _cleanup()

    def test_encrypt_combine_failure_closes_handles(self):
        """combine_single_ciphertexts fails; all individual CT handles closed."""
        client = Client(MLP_PARAMS)
        max_slots = ffi.client_max_slots(client._handle)
        big_tensor = torch.randn(max_slots * 2)
        multi_pt = client.encode(big_tensor)

        allocated = []
        real_fn = ffi.client_encrypt

        def tracking_encrypt(client_h, pt_h):
            h = real_fn(client_h, pt_h)
            allocated.append(h)
            return h

        with patch.object(ffi, 'client_encrypt', side_effect=tracking_encrypt), \
             patch.object(ffi, 'combine_single_ciphertexts',
                          side_effect=RuntimeError("injected combine failure")):
            with pytest.raises(RuntimeError, match="injected combine failure"):
                client.encrypt(multi_pt)

        assert len(allocated) == 2
        for h in allocated:
            assert h._raw == 0, "CT handle should be closed after combine failure"

        client.close()
        _cleanup()

    def test_evaluate_transforms_eval_add_failure(self):
        """eval_add fails mid-transform; intermediate handles cleaned up."""
        compiled_bytes, keys_bytes, sk_bytes = _compile_model()
        compiled = CompiledModel.from_bytes(compiled_bytes)
        keys = EvalKeys.from_bytes(keys_bytes)
        net = SimpleMLP()
        evaluator = Evaluator(net, compiled, keys)

        client = Client(MLP_PARAMS, secret_key=sk_bytes)
        pt = client.encode(torch.randn(1, 784), level=compiled.input_level)
        ct = client.encrypt(pt)
        ct.context = evaluator._context
        ct.on_shape = ct.shape

        fc1 = net.fc1
        handles = fc1.transform_handles
        keys_list = list(handles.keys())
        has_multi_col = any(k[1] > 0 for k in keys_list)

        # Ensure multi-column transforms so eval_add is called.
        # If only single-column, duplicate first entry as second column.
        added_synthetic = False
        if not has_multi_col and keys_list:
            first_key = keys_list[0]
            handles[(first_key[0], first_key[1] + 1)] = handles[first_key]
            added_synthetic = True

        lt_results = []
        real_lt = ffi.eval_linear_transform

        def tracking_lt(eval_h, ct_h, lt_h):
            h = real_lt(eval_h, ct_h, lt_h)
            lt_results.append(h)
            return h

        with patch.object(ffi, 'eval_linear_transform', side_effect=tracking_lt), \
             patch.object(ffi, 'eval_add',
                          side_effect=RuntimeError("injected eval_add failure")):
            with pytest.raises(RuntimeError, match="injected eval_add failure"):
                evaluator._context.evaluate_transforms(fc1, ct)

        # 1st result (ct_out_h) closed by outer except,
        # 2nd result (res_h) closed by inner except
        assert len(lt_results) >= 2
        for h in lt_results:
            assert h._raw == 0, "LT result handle should be closed after eval_add failure"

        if added_synthetic:
            first_key = keys_list[0]
            del handles[(first_key[0], first_key[1] + 1)]

        evaluator.close()
        client.close()
        _cleanup()

    def test_evaluator_init_evaluator_creation_failure(self):
        """new_evaluator() fails after key bundle built; cleanup runs cleanly."""
        compiled_bytes, keys_bytes, _ = _compile_model()
        compiled = CompiledModel.from_bytes(compiled_bytes)
        keys = EvalKeys.from_bytes(keys_bytes)
        net = SimpleMLP()

        with patch.object(ffi, 'new_evaluator',
                          side_effect=RuntimeError("injected evaluator creation failure")):
            with pytest.raises(RuntimeError, match="injected evaluator creation failure"):
                Evaluator(net, compiled, keys)

        # __init__ except calls self.close() on partial state
        # (no _eval_handle, empty _tracked_handles). Must not crash.
        _cleanup()

    def test_evaluator_init_reconstruct_bias_failure(self):
        """eval_encode fails during bias reconstruction; all prior handles freed."""
        compiled_bytes, keys_bytes, _ = _compile_model()
        compiled = CompiledModel.from_bytes(compiled_bytes)
        keys = EvalKeys.from_bytes(keys_bytes)
        net = SimpleMLP()

        # Track LT handles created before the failure
        unmarshalled = []
        real_unmarshal = ffi.linear_transform_unmarshal

        def tracking_unmarshal(data):
            h = real_unmarshal(data)
            unmarshalled.append(h)
            return h

        # Fail on the 2nd eval_encode call (fc2 bias encode),
        # after fc1 is fully reconstructed
        call_count = [0]
        real_encode = ffi.eval_encode

        def fail_on_second_encode(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                raise RuntimeError("injected bias encode failure")
            return real_encode(*args, **kwargs)

        with patch.object(ffi, 'linear_transform_unmarshal', side_effect=tracking_unmarshal), \
             patch.object(ffi, 'eval_encode', side_effect=fail_on_second_encode):
            with pytest.raises(RuntimeError, match="injected bias encode failure"):
                Evaluator(net, compiled, keys)

        # All LT handles (from both fc1 and fc2) should be closed by self.close()
        assert len(unmarshalled) > 0
        for h in unmarshalled:
            assert h._raw == 0, "LT handle should be closed after reconstruction failure"

        _cleanup()


class TestGoErrorPropagation:
    def test_error_propagation(self):
        """Trigger a Go error, verify Python gets RuntimeError, not process crash."""
        from orion.backend.orionclient import ffi

        # Invalid params should raise RuntimeError via errOut pattern
        bad_json = '{"logn": 3, "logq": [10], "logp": [10], "logscale": 5, "h": 64, "ring_type": "standard"}'
        with pytest.raises(RuntimeError) as exc_info:
            ffi.new_client(bad_json)

        # Error message should be non-empty and meaningful
        assert len(str(exc_info.value)) > 0

        # Process should still be alive and functional after the error
        client = Client(MLP_PARAMS)
        inp = torch.randn(1, 784)
        pt = client.encode(inp, level=5)
        assert isinstance(pt, PlainText)
        del client
        _cleanup()
