"""Integration tests for Compiler, Client, and v2 CompiledModel format."""

import gc
import json
import struct

import torch
import pytest

from orion.params import CKKSParams, CostProfile
from orion.compiled_model import (
    CompiledModel,
    KeyManifest,
    EvalKeys,
    Graph,
    unpack_raw_diagonals,
)
from orion.compiler import Compiler
from orion.client import Client
from orion.ciphertext import Ciphertext, PlainText
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
# Shared fixture: compile SimpleMLP once for multiple tests
# -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_mlp():
    """Compile SimpleMLP once and return (compiled, compiler) for tests."""
    torch.manual_seed(42)
    net = SimpleMLP()
    compiler = Compiler(net, MLP_PARAMS)
    compiler.fit(torch.randn(1, 1, 28, 28))
    compiled = compiler.compile()
    yield compiled
    del compiler
    _cleanup_backend()


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

    def test_compiler_produces_compiled_model(self, compiled_mlp):
        """Full Compiler flow: fit + compile -> CompiledModel with graph."""
        compiled = compiled_mlp

        assert isinstance(compiled, CompiledModel)
        assert compiled.params == MLP_PARAMS
        assert isinstance(compiled.manifest, KeyManifest)
        assert len(compiled.manifest.galois_elements) > 0
        assert compiled.manifest.needs_rlk is True
        assert compiled.input_level > 0
        assert len(compiled.blobs) > 0

        # v2: graph has nodes and edges
        assert isinstance(compiled.graph, Graph)
        assert len(compiled.graph.nodes) > 0
        assert len(compiled.graph.edges) > 0

        # Check that linear_transform nodes exist
        has_lt = any(
            n.op == "linear_transform" for n in compiled.graph.nodes
        )
        assert has_lt

    def test_compiled_model_serialization_roundtrip(self, compiled_mlp):
        """CompiledModel to_bytes -> from_bytes roundtrip."""
        compiled = compiled_mlp

        data = compiled.to_bytes()
        compiled2 = CompiledModel.from_bytes(data)

        assert compiled2.params == compiled.params
        assert compiled2.config.margin == compiled.config.margin
        assert compiled2.input_level == compiled.input_level
        assert len(compiled2.blobs) == len(compiled.blobs)

        # v2: graph roundtrips
        assert compiled2.graph.input == compiled.graph.input
        assert compiled2.graph.output == compiled.graph.output
        assert len(compiled2.graph.nodes) == len(compiled.graph.nodes)
        assert len(compiled2.graph.edges) == len(compiled.graph.edges)

        # v2: cost roundtrips
        assert compiled2.cost.bootstrap_count == compiled.cost.bootstrap_count
        assert compiled2.cost.galois_key_count == compiled.cost.galois_key_count

    def test_magic_bytes_v2(self, compiled_mlp):
        """Magic bytes are ORION\\x00\\x02\\x00."""
        data = compiled_mlp.to_bytes()
        assert data[:8] == b"ORION\x00\x02\x00"

    def test_json_header_has_graph(self, compiled_mlp):
        """JSON header contains 'graph' key, not 'topology' or 'modules'."""
        data = compiled_mlp.to_bytes()
        # Parse header: skip 8 magic, read 4-byte header length
        header_len = struct.unpack_from("<I", data, 8)[0]
        header_json = data[12 : 12 + header_len].decode("utf-8")
        header = json.loads(header_json)

        assert "graph" in header
        assert "topology" not in header
        assert "modules" not in header
        assert header["version"] == 2

    def test_blob_refs_point_to_valid_indices(self, compiled_mlp):
        """All blob_refs in nodes point to valid blob indices."""
        compiled = compiled_mlp
        num_blobs = len(compiled.blobs)

        for node in compiled.graph.nodes:
            if node.blob_refs:
                for ref_name, idx in node.blob_refs.items():
                    assert 0 <= idx < num_blobs, (
                        f"Node {node.name} blob_ref '{ref_name}' = {idx} "
                        f"out of range [0, {num_blobs})"
                    )

    def test_edge_refs_exist_in_nodes(self, compiled_mlp):
        """All edge src/dst reference existing node names."""
        compiled = compiled_mlp
        node_names = {n.name for n in compiled.graph.nodes}

        for edge in compiled.graph.edges:
            assert edge.src in node_names, (
                f"Edge src '{edge.src}' not in node names"
            )
            assert edge.dst in node_names, (
                f"Edge dst '{edge.dst}' not in node names"
            )

    def test_linear_transform_blobs_are_raw_float64(self, compiled_mlp):
        """All linear_transform blobs unpack with unpack_raw_diagonals."""
        compiled = compiled_mlp
        max_slots = MLP_PARAMS.max_slots

        for node in compiled.graph.nodes:
            if node.op == "linear_transform" and node.blob_refs:
                for ref_name, idx in node.blob_refs.items():
                    if ref_name.startswith("diag_"):
                        blob = compiled.blobs[idx]
                        diags = unpack_raw_diagonals(blob, max_slots)
                        assert isinstance(diags, dict)
                        assert len(diags) > 0
                        for diag_idx, vals in diags.items():
                            assert len(vals) == max_slots

    def test_bias_blobs_correct_length(self, compiled_mlp):
        """Bias blobs are raw float64 of length max_slots * 8."""
        compiled = compiled_mlp
        max_slots = MLP_PARAMS.max_slots

        for node in compiled.graph.nodes:
            if node.op == "linear_transform" and node.blob_refs:
                if "bias" in node.blob_refs:
                    blob = compiled.blobs[node.blob_refs["bias"]]
                    assert len(blob) == max_slots * 8, (
                        f"Bias blob for {node.name}: expected "
                        f"{max_slots * 8} bytes, got {len(blob)}"
                    )

    def test_polynomial_coeffs_inline(self, compiled_mlp):
        """Polynomial coefficients are inline in node config (not blobs)."""
        compiled = compiled_mlp

        for node in compiled.graph.nodes:
            if node.op == "polynomial":
                assert "coeffs" in node.config, (
                    f"Node {node.name}: polynomial has no 'coeffs' in config"
                )
                assert isinstance(node.config["coeffs"], list)
                assert node.blob_refs is None

    def test_edges_form_valid_dag(self, compiled_mlp):
        """Edges form a valid DAG (acyclic)."""
        import networkx as nx

        compiled = compiled_mlp
        g = nx.DiGraph()
        for node in compiled.graph.nodes:
            g.add_node(node.name)
        for edge in compiled.graph.edges:
            g.add_edge(edge.src, edge.dst)

        assert nx.is_directed_acyclic_graph(g)

    def test_cost_profile_populated(self, compiled_mlp):
        """CostProfile fields are populated and reasonable."""
        compiled = compiled_mlp

        assert isinstance(compiled.cost, CostProfile)
        assert compiled.cost.bootstrap_count >= 0
        assert compiled.cost.galois_key_count > 0
        assert compiled.cost.galois_key_count == len(
            compiled.manifest.galois_elements
        )
        assert compiled.cost.bootstrap_key_count >= 0


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
        assert isinstance(ct, Ciphertext)

        pt2 = client.decrypt(ct)
        decoded = client.decode(pt2)
        assert decoded.shape == inp.shape
        assert torch.allclose(decoded.float(), inp.float(), atol=1e-3)

        del client
        _cleanup_backend()

    def test_ciphertext_serialization(self):
        """Ciphertext to_bytes/from_bytes roundtrip."""
        client = Client(MLP_PARAMS)
        inp = torch.randn(1, 784)
        pt = client.encode(inp, level=5)
        ct = client.encrypt(pt)

        ct_bytes = ct.to_bytes()
        assert isinstance(ct_bytes, bytes)
        assert len(ct_bytes) > 0

        ct2 = Ciphertext.from_bytes(ct_bytes)
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
# Evaluator tests (skipped — Python evaluator removed in v2)
# -----------------------------------------------------------------------


@pytest.mark.skip(
    reason="Python evaluator removed — Phase 2 provides Go evaluator"
)
class TestEvaluator:
    def test_evaluator_modules_have_levels(self):
        pass


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
        ct2 = Ciphertext.from_bytes(ct_bytes)
        pt2 = client2.decrypt(ct2)
        decoded = client2.decode(pt2)

        assert decoded.shape == inp.shape
        assert torch.allclose(decoded.float(), inp.float(), atol=1e-3)

        del client2
        _cleanup_backend()
