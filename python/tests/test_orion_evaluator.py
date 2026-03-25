"""Tests for the orion-evaluator Python package.

Unit tests for Model and Evaluator lifecycle, plus E2E test:
compile (orion-compiler) -> keygen+encrypt (lattigo) ->
forward (orion-evaluator) -> decrypt (lattigo).
"""

import gc
import json
import os

import orion_compiler.nn as on
import pytest
import torch
from lattigo.ckks import Encoder, Parameters
from lattigo.rlwe import (
    Ciphertext as RLWECiphertext,
)
from lattigo.rlwe import (
    Decryptor,
    Encryptor,
    KeyGenerator,
    MemEvaluationKeySet,
)
from orion_compiler.compiler import Compiler
from orion_compiler.params import CKKSParams
from orion_evaluator import Evaluator, EvaluatorError, Model
from orion_evaluator.errors import ModelLoadError

# Path to pre-compiled test models
TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "evaluator", "testdata")
MLP_ORION = os.path.join(TESTDATA_DIR, "mlp.orion")
MLP_INPUT = os.path.join(TESTDATA_DIR, "mlp.input.json")
MLP_EXPECTED = os.path.join(TESTDATA_DIR, "mlp.expected.json")


def _cleanup():
    gc.collect()


def _params_from_dict(params_dict: dict) -> Parameters:
    """Convert a client_params() dict to a Parameters object."""
    return Parameters.from_dict(params_dict)


# -----------------------------------------------------------------------
# Model lifecycle tests
# -----------------------------------------------------------------------


class TestModelLifecycle:
    def test_load_valid_model(self):
        """Model.load succeeds on valid .orion file."""
        with open(MLP_ORION, "rb") as f:
            data = f.read()
        model = Model.load(data)
        assert model._handle != 0
        model.close()
        _cleanup()

    def test_client_params(self):
        """Model.client_params returns valid params, manifest, input_level."""
        with open(MLP_ORION, "rb") as f:
            data = f.read()
        model = Model.load(data)
        params, manifest, input_level = model.client_params()

        assert isinstance(params, dict)
        assert "logn" in params
        assert "logq" in params
        assert "logp" in params
        assert "log_default_scale" in params

        assert isinstance(manifest, dict)
        assert "galois_elements" in manifest
        assert "needs_rlk" in manifest
        assert len(manifest["galois_elements"]) > 0

        assert isinstance(input_level, int)
        assert input_level > 0

        model.close()
        _cleanup()

    def test_close_is_idempotent(self):
        """Closing a model twice does not crash."""
        with open(MLP_ORION, "rb") as f:
            data = f.read()
        model = Model.load(data)
        model.close()
        model.close()  # should not raise
        _cleanup()

    def test_closed_model_raises(self):
        """Operations on closed model raise."""
        with open(MLP_ORION, "rb") as f:
            data = f.read()
        model = Model.load(data)
        model.close()
        with pytest.raises(EvaluatorError, match="closed"):
            model.client_params()
        _cleanup()

    def test_load_invalid_data_raises(self):
        """Model.load with invalid data raises ModelLoadError (subclass of EvaluatorError)."""
        with pytest.raises(ModelLoadError):
            Model.load(b"not a valid orion file")
        _cleanup()


# -----------------------------------------------------------------------
# Evaluator lifecycle tests
# -----------------------------------------------------------------------


def _make_evaluator_from_model(model):
    """Helper: create an Evaluator from a loaded Model using lattigo keygen."""
    params_dict, manifest, _input_level = model.client_params()
    params = _params_from_dict(params_dict)

    kg = KeyGenerator(params)
    sk = kg.gen_secret_key()

    rlk = kg.gen_relin_key(sk) if manifest["needs_rlk"] else None
    gks = [kg.gen_galois_key(sk, int(ge)) for ge in manifest["galois_elements"]]

    evk = MemEvaluationKeySet(rlk=rlk, galois_keys=gks)
    keys_bytes = evk.marshal_binary()

    evaluator = Evaluator(params_dict, keys_bytes)

    # Cleanup keygen objects
    evk.close()
    for gk in gks:
        gk.close()
    if rlk:
        rlk.close()

    return evaluator, sk, params, kg


class TestEvaluatorLifecycle:
    def test_create_and_close(self):
        """Evaluator can be created and closed."""
        with open(MLP_ORION, "rb") as f:
            data = f.read()
        model = Model.load(data)
        evaluator, sk, params, kg = _make_evaluator_from_model(model)

        assert evaluator._handle is not None
        evaluator.close()
        assert evaluator._handle is None

        sk.close()
        kg.close()
        params.close()
        model.close()
        _cleanup()

    def test_close_is_idempotent(self):
        """Closing evaluator twice does not crash."""
        with open(MLP_ORION, "rb") as f:
            data = f.read()
        model = Model.load(data)
        evaluator, sk, params, kg = _make_evaluator_from_model(model)

        evaluator.close()
        evaluator.close()  # should not raise

        sk.close()
        kg.close()
        params.close()
        model.close()
        _cleanup()

    def test_closed_evaluator_raises(self):
        """Operations on closed evaluator raise."""
        with open(MLP_ORION, "rb") as f:
            data = f.read()
        model = Model.load(data)
        evaluator, sk, params, kg = _make_evaluator_from_model(model)
        evaluator.close()

        with pytest.raises(EvaluatorError, match="closed"):
            evaluator.forward(model, [b"dummy"])

        sk.close()
        kg.close()
        params.close()
        model.close()
        _cleanup()


# -----------------------------------------------------------------------
# Bootstrap key parameter tests
# -----------------------------------------------------------------------


class TestBootstrapKeyParameter:
    def test_evaluator_without_btp_keys_on_non_bootstrap_model(self):
        """Evaluator(params, keys, btp_keys_bytes=None) works for non-bootstrap model."""
        with open(MLP_ORION, "rb") as f:
            data = f.read()
        model = Model.load(data)
        params_dict, manifest, _input_level = model.client_params()
        params = _params_from_dict(params_dict)

        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        rlk = kg.gen_relin_key(sk) if manifest["needs_rlk"] else None
        gks = [kg.gen_galois_key(sk, int(ge)) for ge in manifest["galois_elements"]]
        evk = MemEvaluationKeySet(rlk=rlk, galois_keys=gks)
        keys_bytes = evk.marshal_binary()

        # Explicitly pass btp_keys_bytes=None
        evaluator = Evaluator(params_dict, keys_bytes, btp_keys_bytes=None)
        assert evaluator._handle != 0

        evaluator.close()
        evk.close()
        for gk in gks:
            gk.close()
        if rlk:
            rlk.close()
        sk.close()
        kg.close()
        params.close()
        model.close()
        _cleanup()

    def test_evaluator_positional_btp_keys_none(self):
        """Evaluator(params, keys, None) works same as omitting btp_keys_bytes."""
        with open(MLP_ORION, "rb") as f:
            data = f.read()
        model = Model.load(data)
        params_dict, manifest, _input_level = model.client_params()
        params = _params_from_dict(params_dict)

        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        rlk = kg.gen_relin_key(sk) if manifest["needs_rlk"] else None
        gks = [kg.gen_galois_key(sk, int(ge)) for ge in manifest["galois_elements"]]
        evk = MemEvaluationKeySet(rlk=rlk, galois_keys=gks)
        keys_bytes = evk.marshal_binary()

        # Pass None positionally (not keyword)
        evaluator = Evaluator(params_dict, keys_bytes, None)
        assert evaluator._handle != 0

        evaluator.close()
        evk.close()
        for gk in gks:
            gk.close()
        if rlk:
            rlk.close()
        sk.close()
        kg.close()
        params.close()
        model.close()
        _cleanup()

    def test_forward_without_btp_keys_e2e(self):
        """E2E forward pass with explicit btp_keys_bytes=None works on non-bootstrap model."""
        with open(MLP_ORION, "rb") as f:
            model_data = f.read()
        model = Model.load(model_data)
        params_dict, manifest, input_level = model.client_params()

        params = _params_from_dict(params_dict)
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        pk = kg.gen_public_key(sk)
        encoder = Encoder(params)
        encryptor = Encryptor(params, pk)
        decryptor = Decryptor(params, sk)

        rlk = kg.gen_relin_key(sk) if manifest["needs_rlk"] else None
        gks = [kg.gen_galois_key(sk, int(ge)) for ge in manifest["galois_elements"]]
        evk = MemEvaluationKeySet(rlk=rlk, galois_keys=gks)
        keys_bytes = evk.marshal_binary()

        evaluator = Evaluator(params_dict, keys_bytes, btp_keys_bytes=None)

        with open(MLP_INPUT) as f:
            input_values = json.load(f)
        max_slots = params.max_slots()
        padded = input_values + [0.0] * (max_slots - len(input_values))
        scale = params.default_scale()

        pt = encoder.encode(padded, input_level, scale)
        ct = encryptor.encrypt_new(pt)
        ct_bytes = ct.marshal_binary()

        result_bytes_list = evaluator.forward(model, [ct_bytes])
        assert isinstance(result_bytes_list, list)
        assert len(result_bytes_list) == 1
        result_bytes = result_bytes_list[0]
        assert isinstance(result_bytes, bytes)
        assert len(result_bytes) > 0

        result_ct = RLWECiphertext.unmarshal_binary(result_bytes)
        result_pt = decryptor.decrypt_new(result_ct)
        decoded = encoder.decode(result_pt, max_slots)

        with open(MLP_EXPECTED) as f:
            expected = json.load(f)

        tolerance = 0.025
        for i, v in enumerate(expected):
            assert abs(v - decoded[i]) < tolerance

        result_pt.close()
        result_ct.close()
        evaluator.close()
        evk.close()
        for gk in gks:
            gk.close()
        if rlk:
            rlk.close()
        ct.close()
        pt.close()
        decryptor.close()
        encryptor.close()
        pk.close()
        sk.close()
        kg.close()
        encoder.close()
        params.close()
        model.close()
        _cleanup()


# -----------------------------------------------------------------------
# E2E test: compile -> keygen+encrypt -> forward -> decrypt
# -----------------------------------------------------------------------


class TestE2EForward:
    def test_forward_mlp_precompiled(self):
        """E2E: load precompiled MLP -> keygen -> encrypt -> forward -> decrypt -> correct output.

        Uses the pre-compiled mlp.orion test model with known input/expected output.
        """
        # Load model
        with open(MLP_ORION, "rb") as f:
            model_data = f.read()
        model = Model.load(model_data)
        params_dict, manifest, input_level = model.client_params()

        # Create lattigo objects for keygen + encryption
        params = _params_from_dict(params_dict)
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        pk = kg.gen_public_key(sk)
        encoder = Encoder(params)
        encryptor = Encryptor(params, pk)
        decryptor = Decryptor(params, sk)

        # Generate evaluation keys
        rlk = kg.gen_relin_key(sk) if manifest["needs_rlk"] else None
        gks = [kg.gen_galois_key(sk, int(ge)) for ge in manifest["galois_elements"]]
        evk = MemEvaluationKeySet(rlk=rlk, galois_keys=gks)
        keys_bytes = evk.marshal_binary()

        # Create evaluator
        evaluator = Evaluator(params_dict, keys_bytes)

        # Encode and encrypt input
        with open(MLP_INPUT) as f:
            input_values = json.load(f)
        max_slots = params.max_slots()
        padded = input_values + [0.0] * (max_slots - len(input_values))
        scale = params.default_scale()

        pt = encoder.encode(padded, input_level, scale)
        ct = encryptor.encrypt_new(pt)
        ct_bytes = ct.marshal_binary()

        # Forward pass
        result_bytes_list = evaluator.forward(model, [ct_bytes])
        assert isinstance(result_bytes_list, list)
        assert len(result_bytes_list) == 1
        result_bytes = result_bytes_list[0]
        assert isinstance(result_bytes, bytes)
        assert len(result_bytes) > 0

        # Decrypt result
        result_ct = RLWECiphertext.unmarshal_binary(result_bytes)
        result_pt = decryptor.decrypt_new(result_ct)
        decoded = encoder.decode(result_pt, max_slots)

        # Compare with expected
        with open(MLP_EXPECTED) as f:
            expected = json.load(f)

        # Calibrated tolerance from Go evaluator tests: max_observed * 1.5 = 0.025
        tolerance = 0.025
        for i, v in enumerate(expected):
            assert abs(v - decoded[i]) < tolerance, (
                f"slot {i}: expected {v}, got {decoded[i]}, diff {abs(v - decoded[i])}"
            )

        # Cleanup
        result_pt.close()
        result_ct.close()
        evaluator.close()
        evk.close()
        for gk in gks:
            gk.close()
        if rlk:
            rlk.close()
        ct.close()
        pt.close()
        decryptor.close()
        encryptor.close()
        pk.close()
        sk.close()
        kg.close()
        encoder.close()
        params.close()
        model.close()
        _cleanup()

    def test_forward_mlp_from_compiler(self):
        """E2E: compile fresh MLP -> keygen -> encrypt -> forward -> decrypt -> reasonable output.

        This test exercises the full pipeline from PyTorch model to encrypted inference.
        """

        class TinyMLP(on.Module):
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

        # Compile
        torch.manual_seed(42)
        net = TinyMLP()
        mlp_params = CKKSParams(
            logn=13,
            logq=[29, 26, 26, 26, 26, 26],
            logp=[29, 29],
            log_default_scale=26,
            h=8192,
            ring_type="conjugate_invariant",
        )
        compiler = Compiler(net, mlp_params)
        compiler.fit(torch.randn(1, 1, 28, 28))
        compiled = compiler.compile()
        model_bytes = compiled.to_bytes()

        # Load in evaluator
        model = Model.load(model_bytes)
        params_dict, manifest, input_level = model.client_params()

        # Create lattigo objects
        params = _params_from_dict(params_dict)
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        pk = kg.gen_public_key(sk)
        encoder = Encoder(params)
        encryptor = Encryptor(params, pk)
        decryptor = Decryptor(params, sk)

        rlk = kg.gen_relin_key(sk) if manifest["needs_rlk"] else None
        gks = [kg.gen_galois_key(sk, int(ge)) for ge in manifest["galois_elements"]]
        evk = MemEvaluationKeySet(rlk=rlk, galois_keys=gks)
        keys_bytes = evk.marshal_binary()

        evaluator = Evaluator(params_dict, keys_bytes)

        # Encode, encrypt, forward
        torch.manual_seed(123)
        test_input = torch.randn(1, 1, 28, 28)
        max_slots = params.max_slots()
        flat = test_input.flatten().double().tolist()
        padded = flat + [0.0] * (max_slots - len(flat))
        scale = params.default_scale()

        pt = encoder.encode(padded, input_level, scale)
        ct = encryptor.encrypt_new(pt)
        ct_bytes = ct.marshal_binary()

        result_bytes_list = evaluator.forward(model, [ct_bytes])
        result_bytes = result_bytes_list[0]

        # Decrypt
        result_ct = RLWECiphertext.unmarshal_binary(result_bytes)
        result_pt = decryptor.decrypt_new(result_ct)
        decoded = encoder.decode(result_pt, max_slots)

        # Compare with cleartext
        net.eval()
        with torch.no_grad():
            cleartext = net(test_input).flatten().tolist()

        # Tolerance for freshly compiled model (larger because of random weights)
        tolerance = 0.1
        for i in range(len(cleartext)):
            assert abs(cleartext[i] - decoded[i]) < tolerance, (
                f"slot {i}: cleartext {cleartext[i]}, fhe {decoded[i]}, "
                f"diff {abs(cleartext[i] - decoded[i])}"
            )

        # Cleanup
        result_pt.close()
        result_ct.close()
        evaluator.close()
        evk.close()
        for gk in gks:
            gk.close()
        if rlk:
            rlk.close()
        ct.close()
        pt.close()
        decryptor.close()
        encryptor.close()
        pk.close()
        sk.close()
        kg.close()
        encoder.close()
        params.close()
        model.close()
        del compiler
        _cleanup()

    def test_forward_conv2d_small_channels(self):
        """E2E: Conv2d(1,5,k=2,s=2,p=0) with small channel count and output rotations."""

        class SmallConvNet(on.Module):
            """Minimal Conv2d model with small channel count and output rotations."""

            def __init__(self):
                super().__init__()
                self.conv1 = on.Conv2d(1, 5, kernel_size=2, stride=2, padding=0)
                self.act1 = on.Quad()
                self.flatten = on.Flatten()
                self.fc1 = on.Linear(5 * 14 * 14, 10)

            def forward(self, x):
                x = self.act1(self.conv1(x))
                x = self.flatten(x)
                return self.fc1(x)

        # Compile
        torch.manual_seed(42)
        net = SmallConvNet()
        conv_params = CKKSParams(
            logn=13,
            logq=[29, 26, 26, 26, 26, 26, 26, 26, 26, 26],
            logp=[29, 29],
            log_default_scale=26,
            h=8192,
            ring_type="conjugate_invariant",
        )
        compiler = Compiler(net, conv_params)
        compiler.fit(torch.randn(1, 1, 28, 28))
        compiled = compiler.compile()
        model_bytes = compiled.to_bytes()

        # Load in evaluator
        model = Model.load(model_bytes)
        params_dict, manifest, input_level = model.client_params()

        # Create lattigo objects
        params = _params_from_dict(params_dict)
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        pk = kg.gen_public_key(sk)
        encoder = Encoder(params)
        encryptor = Encryptor(params, pk)
        decryptor = Decryptor(params, sk)

        rlk = kg.gen_relin_key(sk) if manifest["needs_rlk"] else None
        gks = [kg.gen_galois_key(sk, int(ge)) for ge in manifest["galois_elements"]]
        evk = MemEvaluationKeySet(rlk=rlk, galois_keys=gks)
        keys_bytes = evk.marshal_binary()

        evaluator = Evaluator(params_dict, keys_bytes)

        # Encode, encrypt, forward
        torch.manual_seed(123)
        test_input = torch.randn(1, 1, 28, 28)
        max_slots = params.max_slots()
        flat = test_input.flatten().double().tolist()
        padded = flat + [0.0] * (max_slots - len(flat))
        scale = params.default_scale()

        pt = encoder.encode(padded, input_level, scale)
        ct = encryptor.encrypt_new(pt)
        ct_bytes = ct.marshal_binary()

        result_bytes_list = evaluator.forward(model, [ct_bytes])
        result_bytes = result_bytes_list[0]

        # Decrypt
        result_ct = RLWECiphertext.unmarshal_binary(result_bytes)
        result_pt = decryptor.decrypt_new(result_ct)
        decoded = encoder.decode(result_pt, max_slots)

        # Compare with cleartext
        net.eval()
        with torch.no_grad():
            cleartext = net(test_input).flatten().tolist()

        tolerance = 0.1
        max_diff = max(abs(cleartext[i] - decoded[i]) for i in range(len(cleartext)))
        assert max_diff < tolerance, (
            f"Conv2d small channels: max diff {max_diff:.6f} exceeds tolerance {tolerance}"
        )

        # Cleanup
        result_pt.close()
        result_ct.close()
        evaluator.close()
        evk.close()
        for gk in gks:
            gk.close()
        if rlk:
            rlk.close()
        ct.close()
        pt.close()
        decryptor.close()
        encryptor.close()
        pk.close()
        sk.close()
        kg.close()
        encoder.close()
        params.close()
        model.close()
        del compiler
        _cleanup()
