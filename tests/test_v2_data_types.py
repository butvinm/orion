"""Tests for v2 data types: CostProfile, GraphNode, GraphEdge, Graph."""

import pytest

from orion.params import CostProfile
from orion.compiled_model import GraphNode, GraphEdge, Graph


# -----------------------------------------------------------------------
# CostProfile
# -----------------------------------------------------------------------


class TestCostProfile:
    def test_creation(self):
        cp = CostProfile(bootstrap_count=3, galois_key_count=42, bootstrap_key_count=2)
        assert cp.bootstrap_count == 3
        assert cp.galois_key_count == 42
        assert cp.bootstrap_key_count == 2

    def test_frozen(self):
        cp = CostProfile(bootstrap_count=0, galois_key_count=0, bootstrap_key_count=0)
        with pytest.raises(AttributeError):
            cp.bootstrap_count = 5

    def test_negative_bootstrap_count(self):
        with pytest.raises(ValueError, match="bootstrap_count"):
            CostProfile(bootstrap_count=-1, galois_key_count=0, bootstrap_key_count=0)

    def test_negative_galois_key_count(self):
        with pytest.raises(ValueError, match="galois_key_count"):
            CostProfile(bootstrap_count=0, galois_key_count=-1, bootstrap_key_count=0)

    def test_negative_bootstrap_key_count(self):
        with pytest.raises(ValueError, match="bootstrap_key_count"):
            CostProfile(bootstrap_count=0, galois_key_count=0, bootstrap_key_count=-1)

    def test_to_dict(self):
        cp = CostProfile(bootstrap_count=3, galois_key_count=42, bootstrap_key_count=2)
        d = cp.to_dict()
        assert d == {
            "bootstrap_count": 3,
            "galois_key_count": 42,
            "bootstrap_key_count": 2,
        }

    def test_from_dict(self):
        d = {
            "bootstrap_count": 5,
            "galois_key_count": 100,
            "bootstrap_key_count": 4,
        }
        cp = CostProfile.from_dict(d)
        assert cp.bootstrap_count == 5
        assert cp.galois_key_count == 100
        assert cp.bootstrap_key_count == 4

    def test_roundtrip(self):
        original = CostProfile(
            bootstrap_count=7, galois_key_count=55, bootstrap_key_count=3
        )
        restored = CostProfile.from_dict(original.to_dict())
        assert restored == original

    def test_zero_counts(self):
        cp = CostProfile(bootstrap_count=0, galois_key_count=0, bootstrap_key_count=0)
        restored = CostProfile.from_dict(cp.to_dict())
        assert restored == cp


# -----------------------------------------------------------------------
# GraphNode
# -----------------------------------------------------------------------


class TestGraphNode:
    def test_creation_minimal(self):
        node = GraphNode(name="fc1", op="linear_transform", level=5, depth=1)
        assert node.name == "fc1"
        assert node.op == "linear_transform"
        assert node.level == 5
        assert node.depth == 1
        assert node.shape is None
        assert node.config == {}
        assert node.blob_refs is None

    def test_creation_full(self):
        node = GraphNode(
            name="fc1",
            op="linear_transform",
            level=5,
            depth=1,
            shape={"fhe_input_shape": [1, 784], "fhe_output_shape": [1, 32]},
            config={"bsgs_ratio": 2.0, "output_rotations": 1},
            blob_refs={"diag_0_0": 0, "bias": 1},
        )
        assert node.shape == {"fhe_input_shape": [1, 784], "fhe_output_shape": [1, 32]}
        assert node.config["bsgs_ratio"] == 2.0
        assert node.blob_refs == {"diag_0_0": 0, "bias": 1}

    def test_to_dict_minimal(self):
        node = GraphNode(name="act1", op="quad", level=4, depth=1)
        d = node.to_dict()
        assert d == {
            "name": "act1",
            "op": "quad",
            "level": 4,
            "depth": 1,
            "config": {},
        }
        assert "shape" not in d
        assert "blob_refs" not in d

    def test_to_dict_full(self):
        node = GraphNode(
            name="fc1",
            op="linear_transform",
            level=5,
            depth=1,
            shape={"fhe_input_shape": [1, 784]},
            config={"bsgs_ratio": 2.0},
            blob_refs={"diag_0_0": 0},
        )
        d = node.to_dict()
        assert d["shape"] == {"fhe_input_shape": [1, 784]}
        assert d["config"] == {"bsgs_ratio": 2.0}
        assert d["blob_refs"] == {"diag_0_0": 0}

    def test_from_dict_minimal(self):
        d = {"name": "act1", "op": "quad", "level": 4, "depth": 1}
        node = GraphNode.from_dict(d)
        assert node.name == "act1"
        assert node.op == "quad"
        assert node.shape is None
        assert node.config == {}
        assert node.blob_refs is None

    def test_roundtrip(self):
        original = GraphNode(
            name="fc1",
            op="linear_transform",
            level=5,
            depth=1,
            shape={"fhe_input_shape": [1, 784], "fhe_output_shape": [1, 32]},
            config={"bsgs_ratio": 2.0, "output_rotations": 1},
            blob_refs={"diag_0_0": 0, "bias": 1},
        )
        restored = GraphNode.from_dict(original.to_dict())
        assert restored == original

    def test_polynomial_node_config(self):
        node = GraphNode(
            name="sig1",
            op="polynomial",
            level=3,
            depth=2,
            config={
                "coeffs": [0.5, 0.25, -0.01],
                "basis": "chebyshev",
                "prescale": 1.0,
                "postscale": 1.0,
                "constant": 0.0,
            },
        )
        d = node.to_dict()
        restored = GraphNode.from_dict(d)
        assert restored.config["coeffs"] == [0.5, 0.25, -0.01]
        assert restored.config["basis"] == "chebyshev"

    def test_bootstrap_node_config(self):
        node = GraphNode(
            name="boot_0",
            op="bootstrap",
            level=1,
            depth=0,
            shape={"slots": 8192},
            config={
                "input_level": 1,
                "input_min": -10.0,
                "input_max": 10.0,
                "prescale": 1.5,
                "postscale": 0.8,
                "constant": 0.0,
                "slots": 8192,
            },
        )
        restored = GraphNode.from_dict(node.to_dict())
        assert restored == node


# -----------------------------------------------------------------------
# GraphEdge
# -----------------------------------------------------------------------


class TestGraphEdge:
    def test_creation(self):
        edge = GraphEdge(src="fc1", dst="act1")
        assert edge.src == "fc1"
        assert edge.dst == "act1"

    def test_frozen(self):
        edge = GraphEdge(src="a", dst="b")
        with pytest.raises(AttributeError):
            edge.src = "c"

    def test_to_dict(self):
        edge = GraphEdge(src="fc1", dst="act1")
        assert edge.to_dict() == {"src": "fc1", "dst": "act1"}

    def test_from_dict(self):
        edge = GraphEdge.from_dict({"src": "fc1", "dst": "act1"})
        assert edge.src == "fc1"
        assert edge.dst == "act1"

    def test_roundtrip(self):
        original = GraphEdge(src="fc1", dst="act1")
        restored = GraphEdge.from_dict(original.to_dict())
        assert restored == original


# -----------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------


class TestGraph:
    def _make_simple_graph(self):
        """A -> B -> C linear graph."""
        nodes = [
            GraphNode(name="A", op="flatten", level=5, depth=0),
            GraphNode(name="B", op="linear_transform", level=5, depth=1),
            GraphNode(name="C", op="quad", level=4, depth=1),
        ]
        edges = [
            GraphEdge(src="A", dst="B"),
            GraphEdge(src="B", dst="C"),
        ]
        return Graph(input="A", output="C", nodes=nodes, edges=edges)

    def test_creation(self):
        g = self._make_simple_graph()
        assert g.input == "A"
        assert g.output == "C"
        assert len(g.nodes) == 3
        assert len(g.edges) == 2

    def test_frozen(self):
        g = self._make_simple_graph()
        with pytest.raises(AttributeError):
            g.input = "X"

    def test_list_coercion(self):
        """Lists passed as nodes/edges should be coerced to tuples."""
        nodes = [
            GraphNode(name="A", op="flatten", level=5, depth=0),
            GraphNode(name="B", op="quad", level=4, depth=1),
        ]
        edges = [GraphEdge(src="A", dst="B")]
        g = Graph(input="A", output="B", nodes=nodes, edges=edges)
        assert isinstance(g.nodes, tuple)
        assert isinstance(g.edges, tuple)

    def test_invalid_input(self):
        nodes = [GraphNode(name="A", op="flatten", level=5, depth=0)]
        edges = []
        with pytest.raises(ValueError, match="input 'X' not found"):
            Graph(input="X", output="A", nodes=nodes, edges=edges)

    def test_invalid_output(self):
        nodes = [GraphNode(name="A", op="flatten", level=5, depth=0)]
        edges = []
        with pytest.raises(ValueError, match="output 'Y' not found"):
            Graph(input="A", output="Y", nodes=nodes, edges=edges)

    def test_to_dict(self):
        g = self._make_simple_graph()
        d = g.to_dict()
        assert d["input"] == "A"
        assert d["output"] == "C"
        assert len(d["nodes"]) == 3
        assert len(d["edges"]) == 2
        assert d["nodes"][0]["name"] == "A"
        assert d["edges"][0] == {"src": "A", "dst": "B"}

    def test_from_dict(self):
        d = {
            "input": "A",
            "output": "B",
            "nodes": [
                {"name": "A", "op": "flatten", "level": 5, "depth": 0},
                {"name": "B", "op": "quad", "level": 4, "depth": 1},
            ],
            "edges": [{"src": "A", "dst": "B"}],
        }
        g = Graph.from_dict(d)
        assert g.input == "A"
        assert g.output == "B"
        assert len(g.nodes) == 2
        assert g.nodes[0].name == "A"

    def test_roundtrip(self):
        original = self._make_simple_graph()
        restored = Graph.from_dict(original.to_dict())
        assert restored == original

    def test_diamond_graph(self):
        """Test a graph with two paths (diamond shape): A -> B, A -> C, B -> D, C -> D."""
        nodes = [
            GraphNode(name="A", op="flatten", level=5, depth=0),
            GraphNode(name="B", op="linear_transform", level=5, depth=1),
            GraphNode(name="C", op="linear_transform", level=5, depth=1),
            GraphNode(name="D", op="add", level=4, depth=0),
        ]
        edges = [
            GraphEdge(src="A", dst="B"),
            GraphEdge(src="A", dst="C"),
            GraphEdge(src="B", dst="D"),
            GraphEdge(src="C", dst="D"),
        ]
        g = Graph(input="A", output="D", nodes=nodes, edges=edges)
        restored = Graph.from_dict(g.to_dict())
        assert restored == g

    def test_single_node_graph(self):
        """A graph with one node and no edges."""
        nodes = [GraphNode(name="X", op="flatten", level=5, depth=0)]
        g = Graph(input="X", output="X", nodes=nodes, edges=[])
        restored = Graph.from_dict(g.to_dict())
        assert restored == g

    def test_empty_edges(self):
        """Graph nodes with no edges — valid (input == output single-node case)."""
        nodes = [GraphNode(name="only", op="flatten", level=0, depth=0)]
        g = Graph(input="only", output="only", nodes=nodes, edges=[])
        assert len(g.edges) == 0
