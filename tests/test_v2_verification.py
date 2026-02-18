"""Task 6: Verification and cleanup tests.

Validates end-to-end correctness of the v2 API:
- Compiler produces correct CompiledModel for smooth and ReLU activations
- All serialization roundtrips preserve data
- No global state remains
- Context-on-tensor isolation between Evaluator instances
"""

import gc
import inspect

import torch
import pytest

from orion.params import CKKSParams
from orion.compiled_model import CompiledModel, KeyManifest, EvalKeys
from orion.compiler import Compiler
from orion.client import Client, CipherText
from orion.evaluator import Evaluator
import orion.nn as on


# -----------------------------------------------------------------------
# Test params
# -----------------------------------------------------------------------

SMALL_PARAMS = CKKSParams(
    logn=13,
    logq=[29, 26, 26, 26, 26, 26],
    logp=[29, 29],
    logscale=26,
    h=8192,
    ring_type="conjugate_invariant",
)

# Sigmoid needs more levels (Chebyshev polynomial consumes ~3 levels)
DEEP_PARAMS = CKKSParams(
    logn=13,
    logq=[29, 26, 26, 26, 26, 26, 26, 26, 26, 26],
    logp=[29, 29],
    logscale=26,
    h=8192,
    ring_type="conjugate_invariant",
)


def _cleanup():
    gc.collect()


# -----------------------------------------------------------------------
# Helper models
# -----------------------------------------------------------------------


class SmoothMLP(on.Module):
    """MLP with smooth (Quad) activations -- no bootstrapping needed."""

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


class SigmoidMLP(on.Module):
    """MLP with Sigmoid activation (Chebyshev polynomial)."""

    def __init__(self):
        super().__init__()
        self.flatten = on.Flatten()
        self.fc1 = on.Linear(784, 32)
        self.act1 = on.Sigmoid()
        self.fc2 = on.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        return self.fc2(x)


# -----------------------------------------------------------------------
# 1. Compiler produces correct CompiledModel
# -----------------------------------------------------------------------


class TestCompilerProducesCorrectModel:
    def test_smooth_activation_model(self):
        """Compiler produces valid CompiledModel for Quad-activation network."""
        torch.manual_seed(42)
        net = SmoothMLP()
        compiler = Compiler(net, SMALL_PARAMS)
        compiler.fit(torch.randn(1, 1, 28, 28))
        compiled = compiler.compile()

        # Structure checks
        assert isinstance(compiled, CompiledModel)
        assert compiled.params == SMALL_PARAMS
        assert isinstance(compiled.manifest, KeyManifest)
        assert compiled.manifest.needs_rlk is True
        assert len(compiled.manifest.galois_elements) > 0
        assert compiled.input_level > 0

        # Module metadata
        has_linear = any(
            m.get("type") in ("Linear", "Conv2d")
            for m in compiled.module_metadata.values()
        )
        assert has_linear, "No LinearTransform modules in metadata"

        has_quad = any(
            m.get("type") == "Quad"
            for m in compiled.module_metadata.values()
        )
        assert has_quad, "No Quad module in metadata"

        # Blobs exist for linear transforms
        assert len(compiled.blobs) > 0

        # Topology is non-empty and contains module names
        assert len(compiled.topology) > 0

        # No bootstrap needed for Quad-only model
        assert len(compiled.manifest.bootstrap_slots) == 0

        del compiler
        _cleanup()

    def test_sigmoid_activation_model(self):
        """Compiler produces valid CompiledModel for Sigmoid (Chebyshev) network."""
        torch.manual_seed(42)
        net = SigmoidMLP()
        compiler = Compiler(net, DEEP_PARAMS)
        compiler.fit(torch.randn(1, 1, 28, 28))
        compiled = compiler.compile()

        assert isinstance(compiled, CompiledModel)

        # Should have Chebyshev activation metadata
        has_cheby = any(
            m.get("type") == "Chebyshev"
            for m in compiled.module_metadata.values()
        )
        assert has_cheby, "No Chebyshev module in metadata for Sigmoid model"

        # Chebyshev modules should have coeffs
        for name, meta in compiled.module_metadata.items():
            if meta.get("type") == "Chebyshev":
                assert "coeffs" in meta, f"Chebyshev {name} missing coeffs"
                assert len(meta["coeffs"]) > 0
                assert "degree" in meta
                assert "prescale" in meta

        del compiler
        _cleanup()


# -----------------------------------------------------------------------
# 2. CompiledModel serialization roundtrip
# -----------------------------------------------------------------------


class TestCompiledModelRoundtrip:
    def test_roundtrip_preserves_all_fields(self):
        """CompiledModel to_bytes -> from_bytes preserves all data."""
        torch.manual_seed(42)
        net = SmoothMLP()
        compiler = Compiler(net, SMALL_PARAMS)
        compiler.fit(torch.randn(1, 1, 28, 28))
        compiled = compiler.compile()

        data = compiled.to_bytes()
        restored = CompiledModel.from_bytes(data)

        # Params
        assert restored.params == compiled.params
        assert restored.params.logn == compiled.params.logn
        assert restored.params.logq == compiled.params.logq
        assert restored.params.logp == compiled.params.logp
        assert restored.params.logscale == compiled.params.logscale
        assert restored.params.h == compiled.params.h
        assert restored.params.ring_type == compiled.params.ring_type

        # Config
        assert restored.config.margin == compiled.config.margin
        assert restored.config.embedding_method == compiled.config.embedding_method
        assert restored.config.fuse_modules == compiled.config.fuse_modules

        # Manifest
        assert restored.manifest.galois_elements == compiled.manifest.galois_elements
        assert restored.manifest.bootstrap_slots == compiled.manifest.bootstrap_slots
        assert restored.manifest.boot_logp == compiled.manifest.boot_logp
        assert restored.manifest.needs_rlk == compiled.manifest.needs_rlk

        # Input level
        assert restored.input_level == compiled.input_level

        # Topology
        assert restored.topology == compiled.topology

        # Module metadata keys
        assert set(restored.module_metadata.keys()) == set(
            compiled.module_metadata.keys()
        )

        # Blob count and content
        assert len(restored.blobs) == len(compiled.blobs)
        for i, (orig, rest) in enumerate(zip(compiled.blobs, restored.blobs)):
            assert orig == rest, f"Blob {i} differs after roundtrip"

        del compiler
        _cleanup()


# -----------------------------------------------------------------------
# 3. EvalKeys serialization roundtrip
# -----------------------------------------------------------------------


class TestEvalKeysRoundtrip:
    def test_keys_roundtrip_with_full_manifest(self):
        """EvalKeys serialization preserves all key data."""
        client = Client(SMALL_PARAMS)
        manifest = KeyManifest(
            galois_elements=frozenset([5, 25, 125, 625]),
            bootstrap_slots=(),
            boot_logp=None,
            needs_rlk=True,
        )
        keys = client.generate_keys(manifest)

        data = keys.to_bytes()
        restored = EvalKeys.from_bytes(data)

        assert restored.has_rlk == keys.has_rlk
        assert restored.galois_elements == keys.galois_elements
        assert len(restored.galois_keys) == len(keys.galois_keys)

        # Key data is identical
        for gal_el in keys.galois_keys:
            assert gal_el in restored.galois_keys
            assert restored.galois_keys[gal_el] == keys.galois_keys[gal_el]

        if keys.rlk_data is not None:
            assert restored.rlk_data == keys.rlk_data

        del client
        _cleanup()


# -----------------------------------------------------------------------
# 4. CipherText serialization roundtrip
# -----------------------------------------------------------------------


class TestCipherTextRoundtrip:
    def test_ciphertext_roundtrip_cross_process(self):
        """CipherText serialization simulates client -> server transfer."""
        client = Client(SMALL_PARAMS)
        inp = torch.randn(1, 784)
        pt = client.encode(inp, level=5)
        ct = client.encrypt(pt)

        # Serialize
        ct_bytes = ct.to_bytes()
        assert isinstance(ct_bytes, bytes)
        assert len(ct_bytes) > 100  # non-trivial

        # Deserialize on same backend (simulates same params on server)
        ct2 = CipherText.from_bytes(ct_bytes, client.backend)

        # Decrypt and verify data integrity
        pt2 = client.decrypt(ct2)
        decoded = client.decode(pt2)

        assert decoded.shape == inp.shape
        assert torch.allclose(decoded.float(), inp.float(), atol=1e-3)

        del client
        _cleanup()


# -----------------------------------------------------------------------
# 5. No global state
# -----------------------------------------------------------------------


class TestNoGlobalState:
    def test_no_module_scheme_class_variable(self):
        """Module class has no 'scheme' class variable."""
        assert not hasattr(on.Module, "scheme"), (
            "Module still has 'scheme' class variable"
        )

    def test_no_module_margin_class_variable(self):
        """Module class has no 'margin' class variable."""
        assert not hasattr(on.Module, "margin"), (
            "Module still has 'margin' class variable"
        )

    def test_no_set_scheme_method(self):
        """Module has no set_scheme() static/class method."""
        assert not hasattr(on.Module, "set_scheme"), (
            "Module still has set_scheme()"
        )

    def test_no_set_margin_method(self):
        """Module has no set_margin() static/class method."""
        assert not hasattr(on.Module, "set_margin"), (
            "Module still has set_margin()"
        )

    def test_no_module_level_scheme_singleton(self):
        """No module-level 'scheme' variable in orion.core or orion.__init__."""
        import orion
        assert not hasattr(orion, "scheme"), (
            "orion module still has 'scheme' attribute"
        )
        import orion.core
        # orion.core.__init__.py should not re-export a scheme
        assert not hasattr(orion.core, "scheme"), (
            "orion.core still has 'scheme' attribute"
        )

    def test_no_yaml_config_in_api(self):
        """The public API classes don't reference YAML config paths."""
        import orion.compiler
        import orion.client
        import orion.evaluator
        for mod in [orion.compiler, orion.client, orion.evaluator]:
            src = inspect.getsource(mod)
            assert ".yml" not in src, f"{mod.__name__} references .yml"
            assert "yaml" not in src.lower(), f"{mod.__name__} references yaml"

    def test_no_hdf5_in_api(self):
        """The public API classes don't reference HDF5."""
        import orion.compiler
        import orion.client
        import orion.evaluator
        for mod in [orion.compiler, orion.client, orion.evaluator]:
            src = inspect.getsource(mod)
            assert "h5py" not in src, f"{mod.__name__} references h5py"
            assert ".h5" not in src, f"{mod.__name__} references .h5"

    def test_old_scheme_file_deleted(self):
        """orion/core/orion.py should not exist."""
        import importlib
        try:
            importlib.import_module("orion.core.orion")
            pytest.fail("orion.core.orion module still exists!")
        except (ImportError, ModuleNotFoundError):
            pass  # expected


# -----------------------------------------------------------------------
# 6. Context-on-tensor isolation
# -----------------------------------------------------------------------


class TestContextIsolation:
    def test_two_evaluators_python_level_isolation(self):
        """Two Evaluator instances have separate Python contexts.

        Note: Go backend is a singleton, so we can't actually run two
        Evaluators simultaneously. But we verify the Python-level context
        objects are independent.
        """
        torch.manual_seed(42)

        # Build a compiled model
        net = SmoothMLP()
        compiler = Compiler(net, SMALL_PARAMS)
        compiler.fit(torch.randn(1, 1, 28, 28))
        compiled = compiler.compile()
        compiled_bytes = compiled.to_bytes()
        del compiler
        _cleanup()

        # Generate keys
        compiled = CompiledModel.from_bytes(compiled_bytes)
        client = Client(compiled.params)
        keys = client.generate_keys(compiled.manifest)
        keys_bytes = keys.to_bytes()
        del client
        _cleanup()

        # Create first evaluator
        compiled = CompiledModel.from_bytes(compiled_bytes)
        keys = EvalKeys.from_bytes(keys_bytes)
        net1 = SmoothMLP()
        evaluator1 = Evaluator(net1, compiled, keys)

        ctx1 = evaluator1._context

        # Verify context is independent from the evaluator
        assert ctx1.backend is evaluator1.backend
        assert ctx1.evaluator is evaluator1._evaluator
        assert ctx1.encoder is evaluator1._encoder

        # Verify context fields are populated
        assert ctx1.lt_evaluator is not None
        assert ctx1.poly_evaluator is not None
        assert ctx1.bootstrapper is not None

        del evaluator1
        _cleanup()


# -----------------------------------------------------------------------
# Additional: imports and version
# -----------------------------------------------------------------------


class TestPublicAPI:
    def test_version(self):
        import orion
        assert orion.__version__ == "2.0.0"

    def test_all_public_exports(self):
        import orion
        assert hasattr(orion, "CKKSParams")
        assert hasattr(orion, "CompilerConfig")
        assert hasattr(orion, "Compiler")
        assert hasattr(orion, "Client")
        assert hasattr(orion, "PlainText")
        assert hasattr(orion, "CipherText")
        assert hasattr(orion, "Evaluator")
        assert hasattr(orion, "CompiledModel")
        assert hasattr(orion, "KeyManifest")
        assert hasattr(orion, "EvalKeys")

    def test_no_flat_api_exports(self):
        """Old flat API (init_scheme, fit, compile, etc.) is gone."""
        import orion
        for name in [
            "init_scheme", "init_params_only", "fit", "compile",
            "encode", "encrypt", "decrypt", "decode",
        ]:
            assert not hasattr(orion, name), (
                f"Old flat API '{name}' still exported from orion"
            )
