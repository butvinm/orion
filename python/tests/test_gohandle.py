"""Tests for GoHandle lifecycle and lattigo primitive object lifecycle.

Migrated from Client-based tests to use lattigo primitives directly.
GoHandle tests verify RAII semantics, close idempotency, and repr formatting.
"""

import gc

import orion_compiler.nn as on
import pytest
import torch

# Lattigo primitive imports
from lattigo.ckks import Encoder, Parameters
from lattigo.errors import FFIError, HandleClosedError
from lattigo.gohandle import GoHandle
from lattigo.rlwe import (
    Ciphertext,
    Decryptor,
    Encryptor,
    KeyGenerator,
    Plaintext,
)
from orion_compiler.params import CKKSParams

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


def _make_params():
    """Create lattigo Parameters matching MLP_PARAMS."""
    return Parameters(
        logn=13,
        logq=[29, 26, 26, 26, 26, 26],
        logp=[29, 29],
        log_default_scale=26,
        ring_type="conjugate_invariant",
        h=8192,
    )


# --- GoHandle unit tests (no FFI needed) ---


class TestGoHandle:
    def test_gohandle_raw_returns_value(self):
        """GoHandle(42).raw == 42"""
        h = GoHandle(42)
        assert h.raw == 42
        # Prevent __del__ from calling DeleteHandle on a fake value
        h._raw = 0

    def test_repr_open_with_tag(self):
        """GoHandle(42, tag="Client") -> 'GoHandle(42 Client)'"""
        h = GoHandle(42, tag="Client")
        assert repr(h) == "GoHandle(42 Client)"
        h._raw = 0

    def test_repr_open_without_tag(self):
        """GoHandle(42) -> 'GoHandle(42)'"""
        h = GoHandle(42)
        assert repr(h) == "GoHandle(42)"
        h._raw = 0

    def test_repr_closed_with_tag(self):
        """After close, repr shows 'GoHandle(closed Evaluator)'"""
        h = GoHandle(42, tag="Evaluator")
        h._raw = 0  # simulate close without calling DeleteHandle on fake value
        assert repr(h) == "GoHandle(closed Evaluator)"

    def test_repr_closed_without_tag(self):
        """After close, repr shows 'GoHandle(closed)'"""
        h = GoHandle(42)
        h._raw = 0
        assert repr(h) == "GoHandle(closed)"

    def test_repr_with_real_handle(self):
        """FFI-created handle has correct repr."""
        params = _make_params()
        h = params._handle
        r = repr(h)
        assert r.startswith("GoHandle(")
        assert "closed" not in r
        params.close()
        assert "closed" in repr(h)
        _cleanup()

    def test_gohandle_close_is_idempotent(self):
        """h.close(); h.close() -- no error on second call.

        Uses a real FFI handle to ensure DeleteHandle is actually called.
        """
        params = _make_params()
        h = params._handle
        params.close()
        # h._raw is now 0, second close is a no-op
        h.close()
        _cleanup()

    def test_gohandle_raw_after_close_raises(self):
        """h.close(); h.raw -> HandleClosedError('Use of closed handle')"""
        params = _make_params()
        h = params._handle
        params.close()
        with pytest.raises(HandleClosedError, match="Use of closed handle"):
            _ = h.raw
        _cleanup()

    def test_gohandle_bool_false_after_close(self):
        """h.close(); assert not h"""
        params = _make_params()
        h = params._handle
        assert h  # truthy when open
        params.close()
        assert not h  # falsy when closed
        _cleanup()


# --- Lattigo primitive lifecycle tests ---


class TestPrimitiveLifecycle:
    def test_parameters_close(self):
        """Parameters.close() releases handle; second close() is no-op."""
        params = _make_params()
        assert params._handle
        params.close()
        assert not params._handle
        # Second close is no-op
        params.close()
        _cleanup()

    def test_encoder_context_manager_pattern(self):
        """Lattigo objects support manual lifecycle management."""
        params = _make_params()
        encoder = Encoder(params)
        values = [1.0, 2.0, 3.0]
        pt = encoder.encode(values, params.max_level(), params.default_scale())
        assert isinstance(pt, Plaintext)
        pt.close()
        encoder.close()
        params.close()
        _cleanup()


# --- Multi-instance independence ---


class TestMultiInstance:
    def test_multi_instance_independence(self):
        """Two parameter sets coexist, operate independently."""
        params1 = Parameters(
            logn=13,
            logq=[29, 26, 26, 26, 26, 26],
            logp=[29, 29],
            log_default_scale=26,
            ring_type="conjugate_invariant",
            h=8192,
        )
        params2 = Parameters(
            logn=13,
            logq=[29, 26, 26, 26],
            logp=[29, 29],
            log_default_scale=26,
            ring_type="conjugate_invariant",
            h=8192,
        )

        encoder1 = Encoder(params1)
        encoder2 = Encoder(params2)
        kg1 = KeyGenerator(params1)
        kg2 = KeyGenerator(params2)
        sk1 = kg1.gen_secret_key()
        sk2 = kg2.gen_secret_key()
        pk1 = kg1.gen_public_key(sk1)
        pk2 = kg2.gen_public_key(sk2)
        enc1 = Encryptor(params1, pk1)
        enc2 = Encryptor(params2, pk2)
        dec1 = Decryptor(params1, sk1)
        dec2 = Decryptor(params2, sk2)

        inp1 = torch.randn(784).double().tolist()
        inp2 = torch.randn(784).double().tolist()
        slots1 = params1.max_slots()
        slots2 = params2.max_slots()
        inp1_padded = inp1 + [0.0] * (slots1 - len(inp1))
        inp2_padded = inp2 + [0.0] * (slots2 - len(inp2))

        pt1 = encoder1.encode(inp1_padded, 5, params1.default_scale())
        pt2 = encoder2.encode(inp2_padded, 3, params2.default_scale())
        ct1 = enc1.encrypt_new(pt1)
        ct2 = enc2.encrypt_new(pt2)

        # Decrypt with respective keys
        pt_dec1 = dec1.decrypt_new(ct1)
        pt_dec2 = dec2.decrypt_new(ct2)
        decoded1 = encoder1.decode(pt_dec1, slots1)[:784]
        decoded2 = encoder2.decode(pt_dec2, slots2)[:784]

        for i in range(784):
            assert abs(decoded1[i] - inp1[i]) < 1e-3
            assert abs(decoded2[i] - inp2[i]) < 1e-3

        # Close first set, second still works
        pt1.close()
        ct1.close()
        pt_dec1.close()
        enc1.close()
        dec1.close()
        encoder1.close()
        params1.close()

        pt3 = encoder2.encode(inp2_padded, 3, params2.default_scale())
        ct3 = enc2.encrypt_new(pt3)
        pt_dec3 = dec2.decrypt_new(ct3)
        decoded3 = encoder2.decode(pt_dec3, slots2)[:784]
        for i in range(784):
            assert abs(decoded3[i] - inp2[i]) < 1e-3

        pt2.close()
        ct2.close()
        pt_dec2.close()
        pt3.close()
        ct3.close()
        pt_dec3.close()
        enc2.close()
        dec2.close()
        pk1.close()
        pk2.close()
        sk1.close()
        sk2.close()
        kg1.close()
        kg2.close()
        encoder2.close()
        params2.close()
        _cleanup()


# --- Wire format roundtrip ---


class TestWireFormat:
    def test_ciphertext_wire_format_roundtrip(self):
        """Ciphertext.unmarshal_binary(ct.marshal_binary()) decrypts to same values."""
        params = _make_params()
        encoder = Encoder(params)
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        pk = kg.gen_public_key(sk)
        encryptor = Encryptor(params, pk)
        decryptor = Decryptor(params, sk)

        inp = torch.randn(784).double().tolist()
        slots = params.max_slots()
        inp_padded = inp + [0.0] * (slots - len(inp))

        pt = encoder.encode(inp_padded, 5, params.default_scale())
        ct = encryptor.encrypt_new(pt)

        ct_bytes = ct.marshal_binary()
        ct2 = Ciphertext.unmarshal_binary(ct_bytes)

        # Verify the round-tripped ciphertext decrypts correctly
        pt2 = decryptor.decrypt_new(ct2)
        decoded = encoder.decode(pt2, slots)[:784]
        for i in range(len(inp)):
            assert abs(decoded[i] - inp[i]) < 1e-3

        # Verify handle types
        assert isinstance(ct._handle, GoHandle)
        assert isinstance(ct2._handle, GoHandle)

        pt.close()
        ct.close()
        ct2.close()
        pt2.close()
        decryptor.close()
        encryptor.close()
        pk.close()
        sk.close()
        kg.close()
        encoder.close()
        params.close()
        _cleanup()


# --- Ciphertext/Plaintext close lifecycle ---


class TestCiphertextClose:
    """Verify Ciphertext.close() and Plaintext.close() are idempotent and safe."""

    def test_ciphertext_close_releases_handle(self):
        """ct.close() releases the Go handle."""
        params = _make_params()
        encoder = Encoder(params)
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        pk = kg.gen_public_key(sk)
        encryptor = Encryptor(params, pk)

        pt = encoder.encode([1.0, 2.0], params.max_level(), params.default_scale())
        ct = encryptor.encrypt_new(pt)
        assert ct._handle._raw != 0
        ct.close()
        assert ct._handle._raw == 0

        pt.close()
        encryptor.close()
        pk.close()
        sk.close()
        kg.close()
        encoder.close()
        params.close()
        _cleanup()

    def test_ciphertext_double_close(self):
        """ct.close(); ct.close() -- no error on second call."""
        params = _make_params()
        encoder = Encoder(params)
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        pk = kg.gen_public_key(sk)
        encryptor = Encryptor(params, pk)

        pt = encoder.encode([1.0, 2.0], params.max_level(), params.default_scale())
        ct = encryptor.encrypt_new(pt)
        ct.close()
        ct.close()  # should not raise

        pt.close()
        encryptor.close()
        pk.close()
        sk.close()
        kg.close()
        encoder.close()
        params.close()
        _cleanup()

    def test_plaintext_close_releases_handle(self):
        """pt.close() releases the Go handle."""
        params = _make_params()
        encoder = Encoder(params)

        pt = encoder.encode([1.0, 2.0], params.max_level(), params.default_scale())
        assert pt._handle._raw != 0
        pt.close()
        assert pt._handle._raw == 0

        encoder.close()
        params.close()
        _cleanup()

    def test_plaintext_double_close(self):
        """pt.close(); pt.close() -- no error on second call."""
        params = _make_params()
        encoder = Encoder(params)

        pt = encoder.encode([1.0, 2.0], params.max_level(), params.default_scale())
        pt.close()
        pt.close()  # should not raise

        encoder.close()
        params.close()
        _cleanup()

    def test_ciphertext_del_via_gc(self):
        """Ciphertext.__del__ triggers close() via garbage collection."""
        params = _make_params()
        encoder = Encoder(params)
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        pk = kg.gen_public_key(sk)
        encryptor = Encryptor(params, pk)

        pt = encoder.encode([1.0, 2.0], params.max_level(), params.default_scale())
        ct = encryptor.encrypt_new(pt)
        h = ct._handle
        assert h._raw != 0
        del ct
        gc.collect()
        assert h._raw == 0, "GC should have closed the handle via __del__"

        pt.close()
        encryptor.close()
        pk.close()
        sk.close()
        kg.close()
        encoder.close()
        params.close()
        _cleanup()

    def test_plaintext_del_via_gc(self):
        """Plaintext.__del__ triggers close() via garbage collection."""
        params = _make_params()
        encoder = Encoder(params)

        pt = encoder.encode([1.0, 2.0], params.max_level(), params.default_scale())
        h = pt._handle
        assert h._raw != 0
        del pt
        gc.collect()
        assert h._raw == 0, "GC should have closed the handle via __del__"

        encoder.close()
        params.close()
        _cleanup()


# --- Moduli chain tests using new lattigo API ---


class TestModuliChain:
    """Test CKKSParams moduli chain via new lattigo API."""

    def test_moduli_chain_returns_list(self):
        """moduli_chain() returns a non-empty list of uint64 moduli."""
        params = _make_params()
        chain = params.moduli_chain()
        assert isinstance(chain, list)
        assert len(chain) > 0
        assert all(isinstance(v, int) and v > 0 for v in chain)
        params.close()
        _cleanup()

    def test_aux_moduli_chain_returns_list(self):
        """aux_moduli_chain() returns a list of uint64 moduli."""
        params = _make_params()
        chain = params.aux_moduli_chain()
        assert isinstance(chain, list)
        assert all(isinstance(v, int) and v > 0 for v in chain)
        params.close()
        _cleanup()

    def test_moduli_chain_matches_params(self):
        """Moduli chain length should correspond to the number of Q primes in params."""
        params = _make_params()
        chain = params.moduli_chain()
        assert len(chain) == len(MLP_PARAMS.logq)
        params.close()
        _cleanup()

    def test_moduli_chain_different_params(self):
        """Moduli chain length matches logq for different parameter sets."""
        params = Parameters(
            logn=13,
            logq=[29, 26, 26, 26],
            logp=[29, 29],
            log_default_scale=26,
            ring_type="conjugate_invariant",
            h=8192,
        )
        chain = params.moduli_chain()
        assert len(chain) == 4
        params.close()
        _cleanup()


class TestContextManager:
    """Verify __enter__/__exit__ context manager protocol."""

    def test_parameters_with_statement(self):
        """with Parameters(...) as p: works, p is closed after block."""
        with Parameters(
            logn=13,
            logq=[29, 26, 26, 26, 26, 26],
            logp=[29, 29],
            log_default_scale=26,
            ring_type="conjugate_invariant",
            h=8192,
        ) as p:
            assert p._handle
            slots = p.max_slots()
            assert slots > 0
        assert not p._handle
        _cleanup()

    def test_encoder_with_statement(self):
        """with Encoder(params) as enc: works, enc is closed after block."""
        params = _make_params()
        with Encoder(params) as enc:
            pt = enc.encode([1.0, 2.0], params.max_level(), params.default_scale())
            assert isinstance(pt, Plaintext)
            pt.close()
        assert not enc._handle
        params.close()
        _cleanup()

    def test_nested_with_statements(self):
        """Nested with statements work correctly."""
        with _make_params() as params:
            with Encoder(params) as enc:
                with KeyGenerator(params) as kg:
                    sk = kg.gen_secret_key()
                    pk = kg.gen_public_key(sk)
                    with Encryptor(params, pk) as encryptor:
                        pt = enc.encode([1.0, 2.0], params.max_level(), params.default_scale())
                        ct = encryptor.encrypt_new(pt)
                        assert ct._handle
                        ct.close()
                        pt.close()
                    assert not encryptor._handle
                    pk.close()
                    sk.close()
                assert not kg._handle
            assert not enc._handle
        assert not params._handle
        _cleanup()

    def test_exit_called_on_exception(self):
        """__exit__ is called even when an exception occurs inside the with block."""
        params = _make_params()
        encoder = None
        with pytest.raises(ValueError, match="test error"), Encoder(params) as enc:
            encoder = enc
            assert enc._handle
            raise ValueError("test error")
        assert encoder is not None
        assert not encoder._handle
        params.close()
        _cleanup()

    def test_gohandle_context_manager(self):
        """GoHandle itself supports with statement with a live handle."""
        params = _make_params()
        # Use the live handle from params to test GoHandle context manager
        handle_raw = params._handle._raw
        assert handle_raw != 0
        with params._handle as gh:
            assert gh._raw == handle_raw
        # After exiting the with block, the handle should be closed
        assert params._handle._raw == 0
        _cleanup()

    def test_gohandle_context_manager_already_closed(self):
        """GoHandle with block on already-closed handle does not error."""
        with GoHandle(0) as gh:
            assert gh._raw == 0
        _cleanup()

    def test_keygen_with_statement(self):
        """KeyGenerator works as context manager."""
        params = _make_params()
        with KeyGenerator(params) as kg:
            sk = kg.gen_secret_key()
            assert sk._handle
            sk.close()
        assert not kg._handle
        params.close()
        _cleanup()

    def test_decryptor_with_statement(self):
        """Decryptor works as context manager."""
        params = _make_params()
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        with Decryptor(params, sk) as dec:
            assert dec._handle
        assert not dec._handle
        sk.close()
        kg.close()
        params.close()
        _cleanup()

    def test_ciphertext_with_statement(self):
        """Ciphertext works as context manager."""
        params = _make_params()
        encoder = Encoder(params)
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        pk = kg.gen_public_key(sk)
        encryptor = Encryptor(params, pk)
        pt = encoder.encode([1.0, 2.0], params.max_level(), params.default_scale())
        with encryptor.encrypt_new(pt) as ct:
            assert ct._handle
            assert ct.level() >= 0
        assert not ct._handle
        pt.close()
        encryptor.close()
        pk.close()
        sk.close()
        kg.close()
        encoder.close()
        params.close()
        _cleanup()


class TestGoErrorPropagation:
    def test_error_propagation(self):
        """Trigger a Go error, verify Python gets FFIError, not process crash."""
        with pytest.raises(FFIError) as exc_info:
            Parameters(
                logn=3, logq=[10], logp=[10], log_default_scale=5, ring_type="standard", h=64
            )

        assert len(str(exc_info.value)) > 0

        # Process should still be alive and functional after the error
        params = _make_params()
        encoder = Encoder(params)
        pt = encoder.encode([1.0, 2.0], params.max_level(), params.default_scale())
        assert isinstance(pt, Plaintext)
        pt.close()
        encoder.close()
        params.close()
        _cleanup()
