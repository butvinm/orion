"""Integration tests for Compiler, Client, Evaluator."""

import gc

import torch
import pytest

from orion.params import CKKSParams, CompilerConfig
from orion.compiled_model import CompiledModel, KeyManifest, EvalKeys
from orion.compiler import Compiler
from orion.client import Client, PlainText, CipherText
from orion.evaluator import Evaluator
import orion.nn as on


# -----------------------------------------------------------------------
# Test params matching the MLP test config
# -----------------------------------------------------------------------

MLP_PARAMS = CKKSParams(
    logn=13,
    logq=[29, 26, 26, 26, 26, 26],
    logp=[29, 29],
    logscale=26,
    h=8192,
    ring_type="conjugate_invariant",
)


# -----------------------------------------------------------------------
# Helper: simple model for faster tests
# -----------------------------------------------------------------------


class SimpleMLP(on.Module):
    """Tiny MLP for fast testing."""

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


def _cleanup_backend():
    """Force garbage collection to clean up Go backend singletons."""
    gc.collect()


# -----------------------------------------------------------------------
# Compiler tests
# -----------------------------------------------------------------------


class TestCompiler:
    def test_compiler_fit_requires_data(self):
        """Compiler.fit() requires tensor or dataloader."""
        net = SimpleMLP()
        compiler = Compiler(net, MLP_PARAMS)
        with pytest.raises(ValueError, match="torch.Tensor or DataLoader"):
            compiler.fit("invalid")
        del compiler
        _cleanup_backend()

    def test_compiler_compile_requires_fit(self):
        """Compiler.compile() raises if fit() not called."""
        net = SimpleMLP()
        compiler = Compiler(net, MLP_PARAMS)
        with pytest.raises(ValueError, match="not been fit"):
            compiler.compile()
        del compiler
        _cleanup_backend()

    def test_compiler_produces_compiled_model(self):
        """Full Compiler flow: fit + compile -> CompiledModel."""
        torch.manual_seed(42)
        net = SimpleMLP()
        compiler = Compiler(net, MLP_PARAMS)

        # Fit with a single sample
        inp = torch.randn(1, 1, 28, 28)
        compiler.fit(inp)

        # Compile
        compiled = compiler.compile()

        assert isinstance(compiled, CompiledModel)
        assert compiled.params == MLP_PARAMS
        assert isinstance(compiled.manifest, KeyManifest)
        assert len(compiled.manifest.galois_elements) > 0
        assert compiled.manifest.needs_rlk is True
        assert compiled.input_level > 0
        assert len(compiled.blobs) > 0
        assert len(compiled.topology) > 0
        assert len(compiled.module_metadata) > 0

        # Check that LinearTransform modules are in metadata
        has_linear = any(
            m.get("type") in ("Linear", "Conv2d")
            for m in compiled.module_metadata.values()
        )
        assert has_linear

        del compiler
        _cleanup_backend()


# -----------------------------------------------------------------------
# Client tests
# -----------------------------------------------------------------------


class TestClient:
    def test_client_encode_decode(self):
        """Client encode/decode roundtrip."""
        client = Client(MLP_PARAMS)
        inp = torch.randn(1, 784)
        pt = client.encode(inp, level=5)
        assert isinstance(pt, PlainText)
        assert len(pt.ids) > 0

        decoded = client.decode(pt)
        assert decoded.shape == inp.shape
        # Allow small FHE encoding error
        assert torch.allclose(decoded.float(), inp.float(), atol=1e-4)

        del client
        _cleanup_backend()

    def test_client_encrypt_decrypt(self):
        """Client encrypt/decrypt roundtrip."""
        client = Client(MLP_PARAMS)
        inp = torch.randn(1, 784)
        pt = client.encode(inp, level=5)
        ct = client.encrypt(pt)
        assert isinstance(ct, CipherText)
        assert len(ct.ids) > 0

        pt2 = client.decrypt(ct)
        decoded = client.decode(pt2)
        assert decoded.shape == inp.shape
        assert torch.allclose(decoded.float(), inp.float(), atol=1e-3)

        del client
        _cleanup_backend()

    def test_ciphertext_serialization(self):
        """CipherText to_bytes/from_bytes roundtrip."""
        client = Client(MLP_PARAMS)
        inp = torch.randn(1, 784)
        pt = client.encode(inp, level=5)
        ct = client.encrypt(pt)

        ct_bytes = ct.to_bytes()
        assert isinstance(ct_bytes, bytes)
        assert len(ct_bytes) > 0

        ct2 = CipherText.from_bytes(ct_bytes, client.backend)
        pt2 = client.decrypt(ct2)
        decoded = client.decode(pt2)
        assert decoded.shape == inp.shape
        assert torch.allclose(decoded.float(), inp.float(), atol=1e-3)

        del client
        _cleanup_backend()

    def test_client_generate_keys(self):
        """Client generates EvalKeys from a KeyManifest."""
        client = Client(MLP_PARAMS)
        manifest = KeyManifest(
            galois_elements=frozenset([5, 25, 125]),
            bootstrap_slots=(),
            boot_logp=None,
            needs_rlk=True,
        )
        keys = client.generate_keys(manifest)
        assert isinstance(keys, EvalKeys)
        assert keys.has_rlk
        assert len(keys.galois_keys) == 3
        assert 5 in keys.galois_keys
        assert 25 in keys.galois_keys
        assert 125 in keys.galois_keys

        del client
        _cleanup_backend()

    def test_evalkeys_serialization(self):
        """EvalKeys to_bytes/from_bytes roundtrip."""
        client = Client(MLP_PARAMS)
        manifest = KeyManifest(
            galois_elements=frozenset([5, 25]),
            bootstrap_slots=(),
            boot_logp=None,
            needs_rlk=True,
        )
        keys = client.generate_keys(manifest)

        keys_bytes = keys.to_bytes()
        assert isinstance(keys_bytes, bytes)

        keys2 = EvalKeys.from_bytes(keys_bytes)
        assert keys2.has_rlk
        assert keys2.galois_elements == {5, 25}

        del client
        _cleanup_backend()


# -----------------------------------------------------------------------
# Evaluator tests
# -----------------------------------------------------------------------


class TestEvaluator:
    def test_compiled_model_serialization_roundtrip(self):
        """CompiledModel to_bytes -> from_bytes -> Evaluator works."""
        torch.manual_seed(42)

        net = SimpleMLP()
        compiler = Compiler(net, MLP_PARAMS)
        compiler.fit(torch.randn(1, 1, 28, 28))
        compiled = compiler.compile()

        # Roundtrip
        data = compiled.to_bytes()
        compiled2 = CompiledModel.from_bytes(data)

        assert compiled2.params == compiled.params
        assert compiled2.config.margin == compiled.config.margin
        assert compiled2.input_level == compiled.input_level
        assert len(compiled2.blobs) == len(compiled.blobs)
        assert compiled2.topology == compiled.topology

        del compiler
        _cleanup_backend()

    def test_evaluator_modules_have_levels(self):
        """After Evaluator construction, modules have correct levels/depths."""
        torch.manual_seed(42)

        net = SimpleMLP()
        compiler = Compiler(net, MLP_PARAMS)
        compiler.fit(torch.randn(1, 1, 28, 28))
        compiled = compiler.compile()

        compiled_bytes = compiled.to_bytes()
        del compiler
        _cleanup_backend()

        # Create client and keys
        compiled = CompiledModel.from_bytes(compiled_bytes)
        client = Client(compiled.params)
        keys = client.generate_keys(compiled.manifest)
        keys_bytes = keys.to_bytes()
        del client
        _cleanup_backend()

        # Create evaluator
        compiled = CompiledModel.from_bytes(compiled_bytes)
        keys = EvalKeys.from_bytes(keys_bytes)
        net_eval = SimpleMLP()
        evaluator = Evaluator(net_eval, compiled, keys)

        # Check that modules have levels set
        for name, module in net_eval.named_modules():
            if name in compiled.module_metadata:
                meta = compiled.module_metadata[name]
                if "level" in meta:
                    assert module.level == meta["level"], (
                        f"Module {name}: expected level {meta['level']}, "
                        f"got {module.level}"
                    )

        del evaluator
        _cleanup_backend()


class TestClientSecretKey:
    def test_secret_key_roundtrip(self):
        """Client secret key can be serialized and restored."""
        client = Client(MLP_PARAMS)
        inp = torch.randn(1, 784)
        pt = client.encode(inp, level=5)
        ct = client.encrypt(pt)

        # Serialize secret key and ciphertext
        sk_bytes = client.secret_key
        ct_bytes = ct.to_bytes()
        assert isinstance(sk_bytes, bytes)
        assert len(sk_bytes) > 0

        del client
        _cleanup_backend()

        # Restore client with same secret key
        client2 = Client(MLP_PARAMS, secret_key=sk_bytes)
        ct2 = CipherText.from_bytes(ct_bytes, client2.backend)
        pt2 = client2.decrypt(ct2)
        decoded = client2.decode(pt2)

        assert decoded.shape == inp.shape
        assert torch.allclose(decoded.float(), inp.float(), atol=1e-3)

        del client2
        _cleanup_backend()
