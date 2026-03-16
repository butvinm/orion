"""Cleartext graph validator for .orion v2 compiled format.

Validates the CompiledModel format without any CKKS operations:
structural correctness, blob format, and numerical accuracy.
"""

import gc

import networkx as nx
import numpy as np
import orion_compiler.nn as on
import pytest
import torch
from orion_compiler.compiled_model import (
    CompiledModel,
    pack_raw_diagonals,
    unpack_raw_bias,
    unpack_raw_diagonals,
)
from orion_compiler.compiler import Compiler
from orion_compiler.params import CKKSParams

# -----------------------------------------------------------------------
# Test params
# -----------------------------------------------------------------------

MLP_PARAMS = CKKSParams(
    logn=13,
    logq=[29, 26, 26, 26, 26, 26],
    logp=[29, 29],
    log_default_scale=26,
    h=8192,
    ring_type="conjugate_invariant",
)


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


def _cleanup():
    gc.collect()


# -----------------------------------------------------------------------
# Shared fixture
# -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_result():
    """Compile SimpleMLP and return (compiled, net, input_tensor)."""
    torch.manual_seed(42)
    net = SimpleMLP()
    net.eval()
    x = torch.randn(1, 1, 28, 28)
    compiler = Compiler(net, MLP_PARAMS)
    compiler.fit(x)
    compiled = compiler.compile()
    yield compiled, net, x
    del compiler
    _cleanup()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def reconstruct_dense_matrix(diags, max_slots, block_height):
    """Reconstruct a dense matrix from packed diagonals.

    Inverse of packing.diagonalize() for a single block.
    The diagonalize() function extracts diagonals as:
        diagonal[i][j] = matrix[j % block_height, (i + j) % max_slots]
    This function inverts that mapping.

    Args:
        diags: dict mapping diagonal index -> list of float64 values
        max_slots: number of CKKS slots (column dimension)
        block_height: height of the block (may be < max_slots for hybrid)

    Returns:
        numpy array of shape (block_height, max_slots)
    """
    matrix = np.zeros((block_height, max_slots))
    j = np.arange(max_slots)
    rows = j % block_height
    for diag_idx, vals in diags.items():
        cols = (diag_idx + j) % max_slots
        matrix[rows, cols] = vals
    return matrix


# -----------------------------------------------------------------------
# Structural tests
# -----------------------------------------------------------------------


class TestStructural:
    def test_roundtrip_preserves_all_fields(self, compiled_result):
        """to_bytes -> from_bytes roundtrip preserves all fields."""
        compiled, _, _ = compiled_result
        data = compiled.to_bytes()
        restored = CompiledModel.from_bytes(data)

        assert restored.params == compiled.params
        assert restored.config.margin == compiled.config.margin
        assert restored.config.embedding_method == compiled.config.embedding_method
        assert restored.config.fuse_modules == compiled.config.fuse_modules
        assert restored.input_level == compiled.input_level
        assert restored.cost == compiled.cost
        assert restored.graph.input == compiled.graph.input
        assert restored.graph.output == compiled.graph.output
        assert len(restored.graph.nodes) == len(compiled.graph.nodes)
        assert len(restored.graph.edges) == len(compiled.graph.edges)
        assert len(restored.blobs) == len(compiled.blobs)
        for i in range(len(compiled.blobs)):
            assert restored.blobs[i] == compiled.blobs[i]

    def test_blob_refs_point_to_valid_indices(self, compiled_result):
        """All blob_refs point to valid blob indices."""
        compiled, _, _ = compiled_result
        for node in compiled.graph.nodes:
            if node.blob_refs:
                for ref_name, idx in node.blob_refs.items():
                    assert 0 <= idx < len(compiled.blobs), (
                        f"Node {node.name}: blob_ref '{ref_name}' = {idx} "
                        f"out of range [0, {len(compiled.blobs)})"
                    )

    def test_edge_refs_exist_in_nodes(self, compiled_result):
        """All edge src/dst reference existing node names."""
        compiled, _, _ = compiled_result
        node_names = {n.name for n in compiled.graph.nodes}
        for edge in compiled.graph.edges:
            assert edge.src in node_names, f"Edge src '{edge.src}' not in node names"
            assert edge.dst in node_names, f"Edge dst '{edge.dst}' not in node names"

    def test_graph_input_output_exist(self, compiled_result):
        """graph.input and graph.output exist in node list."""
        compiled, _, _ = compiled_result
        node_names = {n.name for n in compiled.graph.nodes}
        assert compiled.graph.input in node_names
        assert compiled.graph.output in node_names

    def test_topological_sort_acyclic(self, compiled_result):
        """Topological sort of edges is acyclic."""
        compiled, _, _ = compiled_result
        g = nx.DiGraph()
        for node in compiled.graph.nodes:
            g.add_node(node.name)
        for edge in compiled.graph.edges:
            g.add_edge(edge.src, edge.dst)
        assert nx.is_directed_acyclic_graph(g)

    def test_non_input_nodes_have_incoming_edge(self, compiled_result):
        """Every non-input node has at least one incoming edge."""
        compiled, _, _ = compiled_result
        dst_set = {e.dst for e in compiled.graph.edges}
        for node in compiled.graph.nodes:
            if node.name != compiled.graph.input:
                assert node.name in dst_set, f"Non-input node '{node.name}' has no incoming edge"

    def test_add_mult_have_two_incoming_edges(self, compiled_result):
        """add/mult nodes have exactly two incoming edges."""
        compiled, _, _ = compiled_result
        in_count = {}
        for edge in compiled.graph.edges:
            in_count[edge.dst] = in_count.get(edge.dst, 0) + 1
        for node in compiled.graph.nodes:
            if node.op in ("add", "mult"):
                assert in_count.get(node.name, 0) == 2, (
                    f"Node '{node.name}' (op={node.op}) has "
                    f"{in_count.get(node.name, 0)} incoming edges, expected 2"
                )


# -----------------------------------------------------------------------
# Numerical tests
# -----------------------------------------------------------------------


class TestNumerical:
    def test_reconstruct_identity(self):
        """Reconstruct identity matrix from diagonal 0."""
        diags = {0: [1.0, 1.0, 1.0, 1.0]}
        W = reconstruct_dense_matrix(diags, max_slots=4, block_height=4)
        np.testing.assert_array_equal(W, np.eye(4))

    def test_reconstruct_off_diagonal(self):
        """Reconstruct matrix with off-diagonal entries."""
        diags = {1: [1.0, 2.0, 3.0, 4.0]}
        W = reconstruct_dense_matrix(diags, max_slots=4, block_height=4)
        expected = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 3],
                [4, 0, 0, 0],
            ],
            dtype=float,
        )
        np.testing.assert_array_equal(W, expected)

    def test_reconstruct_hybrid_embedding(self):
        """Reconstruct with block_height < max_slots (hybrid embedding)."""
        max_slots = 8
        block_height = 2
        # Diagonal 0: vals[j] maps to (j%2, j%8)
        # j=0: row 0, col 0 = 1
        # j=1: row 1, col 1 = 2
        # j=2: row 0, col 2 = 3
        # j=3: row 1, col 3 = 4
        # etc.
        diags = {0: [1, 2, 3, 4, 5, 6, 7, 8]}
        W = reconstruct_dense_matrix(diags, max_slots, block_height)
        expected = np.array(
            [
                [1, 0, 3, 0, 5, 0, 7, 0],
                [0, 2, 0, 4, 0, 6, 0, 8],
            ],
            dtype=float,
        )
        np.testing.assert_array_equal(W, expected)

    def test_reconstruct_multiple_diagonals(self):
        """Reconstruct with multiple diagonals."""
        max_slots = 4
        block_height = 4
        diags = {
            0: [1.0, 5.0, 9.0, 13.0],
            1: [2.0, 6.0, 10.0, 14.0],
            2: [3.0, 7.0, 11.0, 15.0],
            3: [4.0, 8.0, 12.0, 16.0],
        }
        W = reconstruct_dense_matrix(diags, max_slots, block_height)
        # diag 0: (0,0)=1, (1,1)=5, (2,2)=9, (3,3)=13
        # diag 1: (0,1)=2, (1,2)=6, (2,3)=10, (3,0)=14
        # diag 2: (0,2)=3, (1,3)=7, (2,0)=11, (3,1)=15
        # diag 3: (0,3)=4, (1,0)=8, (2,1)=12, (3,2)=16
        expected = np.array(
            [
                [1, 2, 3, 4],
                [8, 5, 6, 7],
                [11, 12, 9, 10],
                [14, 15, 16, 13],
            ],
            dtype=float,
        )
        np.testing.assert_array_equal(W, expected)

    def test_cleartext_graph_walk(self, compiled_result):
        """Walk compiled graph in cleartext numpy, compare to PyTorch."""
        compiled, net, x = compiled_result
        max_slots = compiled.params.max_slots

        # PyTorch reference output in float64
        net_f64 = SimpleMLP()
        net_f64.load_state_dict(net.state_dict())
        net_f64.double()
        net_f64.eval()
        x_f64 = x.double()
        with torch.no_grad():
            y_expected = net_f64(x_f64).detach().numpy().flatten()

        # Build networkx graph for topological order
        g = nx.DiGraph()
        node_map = {}
        for node in compiled.graph.nodes:
            g.add_node(node.name)
            node_map[node.name] = node
        for edge in compiled.graph.edges:
            g.add_edge(edge.src, edge.dst)
        topo_order = list(nx.topological_sort(g))

        # Prepare input: flatten, pad to max_slots
        x_flat = x_f64.detach().numpy().flatten()
        x_padded = np.zeros(max_slots)
        x_padded[: len(x_flat)] = x_flat

        # Walk the graph
        values = {}
        for name in topo_order:
            node = node_map[name]
            preds = list(g.predecessors(name))

            if node.op == "flatten":
                if preds:
                    values[name] = values[preds[0]]
                else:
                    # Input node
                    values[name] = x_padded

            elif node.op == "linear_transform":
                inp = values[preds[0]]
                output_rotations = node.config.get("output_rotations", 0)
                block_height = (
                    max_slots // (2**output_rotations) if output_rotations > 0 else max_slots
                )

                # Unpack diagonals for each block
                diags_by_block = {}
                for ref_name, idx in node.blob_refs.items():
                    if ref_name.startswith("diag_"):
                        parts = ref_name.split("_")
                        row, col = int(parts[1]), int(parts[2])
                        diags_by_block[(row, col)] = unpack_raw_diagonals(
                            compiled.blobs[idx], max_slots
                        )

                assert len(diags_by_block) == 1, f"Multi-block not supported in test: {node.name}"
                block_key = next(iter(diags_by_block.keys()))
                W = reconstruct_dense_matrix(diags_by_block[block_key], max_slots, block_height)

                # Dense matrix-vector product
                y = np.matmul(W, inp)
                y_padded = np.zeros(max_slots)
                y_padded[:block_height] = y

                # Add bias
                bias_blob = compiled.blobs[node.blob_refs["bias"]]
                bias = np.array(unpack_raw_bias(bias_blob, max_slots))
                y_padded += bias

                values[name] = y_padded

            elif node.op == "quad":
                inp = values[preds[0]]
                values[name] = inp * inp

            elif node.op == "polynomial":
                inp = values[preds[0]]
                coeffs = node.config["coeffs"]
                basis = node.config.get("basis", "monomial")
                prescale = node.config.get("prescale", 1)
                postscale = node.config.get("postscale", 1)
                constant = node.config.get("constant", 0)
                x_scaled = inp * prescale + constant
                if basis == "chebyshev":
                    result = np.polynomial.chebyshev.chebval(x_scaled, coeffs)
                else:
                    # Horner's method matching Activation.forward
                    result = np.zeros_like(x_scaled)
                    for c in coeffs:
                        result = c + x_scaled * result
                values[name] = result * postscale

            elif node.op in ("add", "mult"):
                a, b = values[preds[0]], values[preds[1]]
                values[name] = a + b if node.op == "add" else a * b

            elif node.op == "bootstrap":
                # In cleartext, bootstrap is identity
                values[name] = values[preds[0]]

            else:
                raise ValueError(f"Unknown op: {node.op}")

        output = values[compiled.graph.output]
        y_computed = output[: len(y_expected)]

        np.testing.assert_allclose(
            y_computed,
            y_expected,
            atol=1e-10,
            err_msg="Cleartext graph walk does not match PyTorch forward",
        )


# -----------------------------------------------------------------------
# Blob format tests
# -----------------------------------------------------------------------


class TestBlobFormat:
    def test_diagonal_roundtrip_with_compiled_data(self, compiled_result):
        """pack_raw_diagonals -> unpack_raw_diagonals roundtrip."""
        compiled, _, _ = compiled_result
        max_slots = compiled.params.max_slots

        for node in compiled.graph.nodes:
            if node.op == "linear_transform" and node.blob_refs:
                for ref_name, idx in node.blob_refs.items():
                    if ref_name.startswith("diag_"):
                        blob = compiled.blobs[idx]
                        diags = unpack_raw_diagonals(blob, max_slots)
                        repacked = pack_raw_diagonals(diags, max_slots)
                        diags2 = unpack_raw_diagonals(repacked, max_slots)
                        assert diags.keys() == diags2.keys()
                        for k in diags:
                            np.testing.assert_array_equal(diags[k], diags2[k])

    def test_diagonal_indices_sorted(self, compiled_result):
        """Diagonal indices sorted ascending in packed output."""
        compiled, _, _ = compiled_result
        max_slots = compiled.params.max_slots

        for node in compiled.graph.nodes:
            if node.op == "linear_transform" and node.blob_refs:
                for ref_name, idx in node.blob_refs.items():
                    if ref_name.startswith("diag_"):
                        blob = compiled.blobs[idx]
                        diags = unpack_raw_diagonals(blob, max_slots)
                        indices = list(diags.keys())
                        assert indices == sorted(indices), f"Not sorted: {node.name}/{ref_name}"

    def test_diagonal_values_have_max_slots(self, compiled_result):
        """Each diagonal has exactly max_slots values."""
        compiled, _, _ = compiled_result
        max_slots = compiled.params.max_slots

        for node in compiled.graph.nodes:
            if node.op == "linear_transform" and node.blob_refs:
                for ref_name, idx in node.blob_refs.items():
                    if ref_name.startswith("diag_"):
                        blob = compiled.blobs[idx]
                        diags = unpack_raw_diagonals(blob, max_slots)
                        for diag_idx, vals in diags.items():
                            assert len(vals) == max_slots, (
                                f"Diagonal {diag_idx} of {node.name} has "
                                f"{len(vals)} values, expected {max_slots}"
                            )

    def test_bias_blob_length(self, compiled_result):
        """Bias blob length = max_slots * 8 bytes."""
        compiled, _, _ = compiled_result
        max_slots = compiled.params.max_slots

        for node in compiled.graph.nodes:
            if node.op == "linear_transform" and node.blob_refs and "bias" in node.blob_refs:
                blob = compiled.blobs[node.blob_refs["bias"]]
                assert len(blob) == max_slots * 8, (
                    f"Bias blob for {node.name}: expected {max_slots * 8} bytes, got {len(blob)}"
                )
