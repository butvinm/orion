"""Integration tests for Compiler and v2 CompiledModel format.

Client tests migrated to use lattigo primitives directly.
"""

import gc
import json
import struct

import orion_compiler.nn as on
import pytest
import torch
from lattigo.ckks import Encoder, Parameters
from lattigo.rlwe import (
    Ciphertext,
    Decryptor,
    Encryptor,
    KeyGenerator,
    MemEvaluationKeySet,
    Plaintext,
    SecretKey,
)
from orion_compiler.compiled_model import (
    CompiledModel,
    Graph,
    KeyManifest,
    unpack_raw_diagonals,
)
from orion_compiler.compiler import Compiler
from orion_compiler.errors import CompilationError
from orion_compiler.params import CKKSParams, CostProfile

# -----------------------------------------------------------------------
# Test params matching the MLP test config
# -----------------------------------------------------------------------

MLP_PARAMS = CKKSParams(
    logn=13,
    logq=[29, 26, 26, 26, 26, 26],
    logp=[29, 29],
    log_default_scale=26,
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
        with pytest.raises(CompilationError, match=r"torch\.Tensor or DataLoader"):
            compiler.fit("invalid")
        del compiler
        _cleanup_backend()

    def test_compiler_compile_requires_fit(self):
        """Compiler.compile() raises if fit() not called."""
        net = SimpleMLP()
        compiler = Compiler(net, MLP_PARAMS)
        with pytest.raises(CompilationError, match="not been fit"):
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
        has_lt = any(n.op == "linear_transform" for n in compiled.graph.nodes)
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
            assert edge.src in node_names, f"Edge src '{edge.src}' not in node names"
            assert edge.dst in node_names, f"Edge dst '{edge.dst}' not in node names"

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
                        for _diag_idx, vals in diags.items():
                            assert len(vals) == max_slots

    def test_bias_blobs_correct_length(self, compiled_mlp):
        """Bias blobs are raw float64 of length max_slots * 8."""
        compiled = compiled_mlp
        max_slots = MLP_PARAMS.max_slots

        for node in compiled.graph.nodes:
            if node.op == "linear_transform" and node.blob_refs and "bias" in node.blob_refs:
                blob = compiled.blobs[node.blob_refs["bias"]]
                assert len(blob) == max_slots * 8, (
                    f"Bias blob for {node.name}: expected {max_slots * 8} bytes, got {len(blob)}"
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
        assert compiled.cost.galois_key_count == len(compiled.manifest.galois_elements)
        assert compiled.cost.bootstrap_key_count >= 0


# -----------------------------------------------------------------------
# Lattigo primitive tests (replacing old Client tests)
# -----------------------------------------------------------------------


def _make_params():
    """Create lattigo Parameters matching MLP_PARAMS."""
    return Parameters(
        logn=13,
        logq=[29, 26, 26, 26, 26, 26],
        logp=[29, 29],
        log_default_scale=26,
        h=8192,
        ring_type="conjugate_invariant",
    )


def _encode_tensor(encoder, params, tensor, level=None):
    """Encode a torch.Tensor into a lattigo rlwe.Plaintext."""
    if level is None:
        level = params.max_level()
    scale = params.default_scale()
    max_slots = params.max_slots()
    flat = tensor.flatten().double().tolist()
    if len(flat) < max_slots:
        flat.extend([0.0] * (max_slots - len(flat)))
    return encoder.encode(flat, level, scale)


def _decode_tensor(encoder, params, pt, shape):
    """Decode a lattigo rlwe.Plaintext back to a torch.Tensor."""
    values = encoder.decode(pt, params.max_slots())
    numel = 1
    for s in shape:
        numel *= s
    return torch.tensor(values[:numel]).reshape(shape)


class TestLattigoPrimitives:
    def test_encode_decode(self):
        """Lattigo encode/decode roundtrip with torch tensor."""
        params = _make_params()
        encoder = Encoder(params)
        inp = torch.randn(1, 784)
        pt = _encode_tensor(encoder, params, inp, level=5)
        assert isinstance(pt, Plaintext)

        decoded = _decode_tensor(encoder, params, pt, inp.shape)
        assert decoded.shape == inp.shape
        assert torch.allclose(decoded.float(), inp.float(), atol=1e-4)

        pt.close()
        encoder.close()
        params.close()
        _cleanup_backend()

    def test_encrypt_decrypt(self):
        """Lattigo encrypt/decrypt roundtrip."""
        params = _make_params()
        encoder = Encoder(params)
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        pk = kg.gen_public_key(sk)
        encryptor = Encryptor(params, pk)
        decryptor = Decryptor(params, sk)

        inp = torch.randn(1, 784)
        pt = _encode_tensor(encoder, params, inp, level=5)
        ct = encryptor.encrypt_new(pt)
        assert isinstance(ct, Ciphertext)

        pt2 = decryptor.decrypt_new(ct)
        decoded = _decode_tensor(encoder, params, pt2, inp.shape)
        assert decoded.shape == inp.shape
        assert torch.allclose(decoded.float(), inp.float(), atol=1e-3)

        pt.close()
        ct.close()
        pt2.close()
        decryptor.close()
        encryptor.close()
        pk.close()
        sk.close()
        kg.close()
        encoder.close()
        params.close()
        _cleanup_backend()

    def test_ciphertext_serialization(self):
        """Ciphertext marshal/unmarshal roundtrip."""
        params = _make_params()
        encoder = Encoder(params)
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        pk = kg.gen_public_key(sk)
        encryptor = Encryptor(params, pk)
        decryptor = Decryptor(params, sk)

        inp = torch.randn(1, 784)
        pt = _encode_tensor(encoder, params, inp, level=5)
        ct = encryptor.encrypt_new(pt)

        ct_bytes = ct.marshal_binary()
        assert isinstance(ct_bytes, bytes)
        assert len(ct_bytes) > 0

        ct2 = Ciphertext.unmarshal_binary(ct_bytes)
        pt2 = decryptor.decrypt_new(ct2)
        decoded = _decode_tensor(encoder, params, pt2, inp.shape)
        assert decoded.shape == inp.shape
        assert torch.allclose(decoded.float(), inp.float(), atol=1e-3)

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
        _cleanup_backend()

    def test_key_generation(self):
        """Generate individual keys via Lattigo KeyGenerator."""
        params = _make_params()
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()

        # Relinearization key
        rlk = kg.gen_relin_key(sk)
        assert rlk._handle

        # Galois keys
        galois_elements = [5, 25, 125]
        gks = []
        for gel in galois_elements:
            gk = kg.gen_galois_key(sk, gel)
            assert gk._handle
            gks.append(gk)

        # Build MemEvaluationKeySet
        evk = MemEvaluationKeySet(rlk=rlk, galois_keys=gks)
        assert evk._handle

        evk.close()
        for gk in gks:
            gk.close()
        rlk.close()
        sk.close()
        kg.close()
        params.close()
        _cleanup_backend()

    def test_eval_key_set_serialization(self):
        """MemEvaluationKeySet marshal/unmarshal roundtrip."""
        params = _make_params()
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()

        rlk = kg.gen_relin_key(sk)
        gk1 = kg.gen_galois_key(sk, 5)
        gk2 = kg.gen_galois_key(sk, 25)

        evk = MemEvaluationKeySet(rlk=rlk, galois_keys=[gk1, gk2])
        keys_bytes = evk.marshal_binary()
        assert isinstance(keys_bytes, bytes)
        assert len(keys_bytes) > 0

        evk2 = MemEvaluationKeySet.unmarshal_binary(keys_bytes)
        assert evk2._handle
        # Re-marshal should produce identical bytes
        keys_bytes2 = evk2.marshal_binary()
        assert keys_bytes == keys_bytes2

        evk2.close()
        evk.close()
        gk2.close()
        gk1.close()
        rlk.close()
        sk.close()
        kg.close()
        params.close()
        _cleanup_backend()


class TestSecretKeyRoundtrip:
    def test_secret_key_roundtrip(self):
        """Secret key can be serialized and used to decrypt with a new Decryptor."""
        params = _make_params()
        encoder = Encoder(params)
        kg = KeyGenerator(params)
        sk = kg.gen_secret_key()
        pk = kg.gen_public_key(sk)
        encryptor = Encryptor(params, pk)

        inp = torch.randn(1, 784)
        pt = _encode_tensor(encoder, params, inp, level=5)
        ct = encryptor.encrypt_new(pt)

        # Serialize secret key and ciphertext
        sk_bytes = sk.marshal_binary()
        ct_bytes = ct.marshal_binary()
        assert isinstance(sk_bytes, bytes)
        assert len(sk_bytes) > 0

        # Destroy original objects
        pt.close()
        ct.close()
        encryptor.close()
        pk.close()
        sk.close()
        kg.close()
        _cleanup_backend()

        # Restore secret key and decrypt
        sk2 = SecretKey.unmarshal_binary(sk_bytes)
        decryptor = Decryptor(params, sk2)
        ct2 = Ciphertext.unmarshal_binary(ct_bytes)
        pt2 = decryptor.decrypt_new(ct2)
        decoded = _decode_tensor(encoder, params, pt2, inp.shape)

        assert decoded.shape == inp.shape
        assert torch.allclose(decoded.float(), inp.float(), atol=1e-3)

        pt2.close()
        ct2.close()
        decryptor.close()
        sk2.close()
        encoder.close()
        params.close()
        _cleanup_backend()
