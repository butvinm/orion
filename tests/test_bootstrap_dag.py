"""Tests for bootstrap node insertion into the NetworkDAG.

Task 4: Bootstrap operations become explicit DAG nodes instead of forward hooks.
"""

import math
import types
from unittest.mock import MagicMock

import networkx as nx
import pytest
import torch

from orion.core.auto_bootstrap import BootstrapPlacer, BootstrapSolver
from orion.core.network_dag import NetworkDAG
from orion.nn.operations import Bootstrap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_module(level, depth, output_min=-1.0, output_max=1.0,
                      fhe_output_shape=None):
    """Create a mock Orion module with the attributes BootstrapPlacer needs."""
    m = MagicMock()
    m.level = level
    m.depth = depth
    # Bootstrap.fit() calls .item() on these, so they must be tensors
    m.output_min = torch.tensor(output_min)
    m.output_max = torch.tensor(output_max)
    m.fhe_output_shape = fhe_output_shape or torch.Size([1, 32])
    return m


def _make_mock_context(margin=1.2):
    """Create a minimal context namespace for bootstrapper.fit()."""
    ctx = types.SimpleNamespace()
    ctx.margin = margin
    return ctx


def _build_simple_dag():
    """Build a simple 3-node DAG: A -> B -> C.

    Returns a plain nx.DiGraph (not NetworkDAG, since NetworkDAG requires
    a trace). This is fine because BootstrapPlacer only uses the graph's
    node/edge methods inherited from nx.DiGraph.
    """
    dag = nx.DiGraph()

    mod_a = _make_mock_module(level=5, depth=1)
    mod_b = _make_mock_module(level=4, depth=1)
    mod_c = _make_mock_module(level=3, depth=1)

    dag.add_node("A", op="linear_transform", module=mod_a, bootstrap=False)
    dag.add_node("B", op="quad", module=mod_b, bootstrap=True)
    dag.add_node("C", op="linear_transform", module=mod_c, bootstrap=False)

    dag.add_edge("A", "B")
    dag.add_edge("B", "C")

    return dag


def _build_forking_dag():
    """Build a DAG where a bootstrapped node has two children:

        A -> B -> C
                -> D
    B is marked for bootstrap.
    """
    dag = nx.DiGraph()

    mod_a = _make_mock_module(level=5, depth=1)
    mod_b = _make_mock_module(level=4, depth=1)
    mod_c = _make_mock_module(level=3, depth=1)
    mod_d = _make_mock_module(level=3, depth=1)

    dag.add_node("A", op="linear_transform", module=mod_a, bootstrap=False)
    dag.add_node("B", op="quad", module=mod_b, bootstrap=True)
    dag.add_node("C", op="linear_transform", module=mod_c, bootstrap=False)
    dag.add_node("D", op="linear_transform", module=mod_d, bootstrap=False)

    dag.add_edge("A", "B")
    dag.add_edge("B", "C")
    dag.add_edge("B", "D")

    return dag


# ---------------------------------------------------------------------------
# Unit tests: bootstrap DAG insertion
# ---------------------------------------------------------------------------


class TestBootstrapDagInsertion:
    def test_bootstrap_node_inserted(self):
        """A bootstrap node is inserted after the marked node."""
        dag = _build_simple_dag()
        ctx = _make_mock_context()

        placer = BootstrapPlacer(net=None, network_dag=dag, context=ctx)
        placer.place_bootstraps()

        assert "boot_0" in dag.nodes
        assert dag.nodes["boot_0"]["op"] == "bootstrap"
        assert isinstance(dag.nodes["boot_0"]["module"], Bootstrap)

    def test_bootstrap_edges_correct(self):
        """Edges are re-linked: B -> boot_0 -> C (not B -> C)."""
        dag = _build_simple_dag()
        ctx = _make_mock_context()

        placer = BootstrapPlacer(net=None, network_dag=dag, context=ctx)
        placer.place_bootstraps()

        # B -> boot_0
        assert dag.has_edge("B", "boot_0")
        # boot_0 -> C
        assert dag.has_edge("boot_0", "C")
        # B -> C should no longer exist
        assert not dag.has_edge("B", "C")
        # A -> B unchanged
        assert dag.has_edge("A", "B")

    def test_bootstrap_level_set(self):
        """Bootstrap node level = predecessor.level - predecessor.depth."""
        dag = _build_simple_dag()
        ctx = _make_mock_context()

        placer = BootstrapPlacer(net=None, network_dag=dag, context=ctx)
        placer.place_bootstraps()

        # B has level=4, depth=1, so boot level = 4 - 1 = 3
        assert dag.nodes["boot_0"]["level"] == 3

    def test_bootstrap_module_attributes(self):
        """Bootstrap module has correct input_level, prescale, etc."""
        dag = _build_simple_dag()
        ctx = _make_mock_context()

        placer = BootstrapPlacer(net=None, network_dag=dag, context=ctx)
        placer.place_bootstraps()

        boot_mod = dag.nodes["boot_0"]["module"]
        assert isinstance(boot_mod, Bootstrap)
        # input_level = predecessor.level - predecessor.depth = 4 - 1 = 3
        assert boot_mod.input_level == 3
        assert boot_mod.fhe_input_shape == torch.Size([1, 32])

    def test_forking_node_bootstrap(self):
        """Bootstrap with multiple children: boot_0 sits between B and all children."""
        dag = _build_forking_dag()
        ctx = _make_mock_context()

        placer = BootstrapPlacer(net=None, network_dag=dag, context=ctx)
        placer.place_bootstraps()

        assert "boot_0" in dag.nodes
        # B -> boot_0
        assert dag.has_edge("B", "boot_0")
        # boot_0 -> C and boot_0 -> D
        assert dag.has_edge("boot_0", "C")
        assert dag.has_edge("boot_0", "D")
        # B -> C and B -> D should no longer exist
        assert not dag.has_edge("B", "C")
        assert not dag.has_edge("B", "D")

    def test_multiple_bootstraps(self):
        """Two nodes marked for bootstrap get separate boot nodes."""
        dag = nx.DiGraph()
        mod_a = _make_mock_module(level=5, depth=1)
        mod_b = _make_mock_module(level=4, depth=1)
        mod_c = _make_mock_module(level=3, depth=1)
        mod_d = _make_mock_module(level=2, depth=1)

        dag.add_node("A", op="linear_transform", module=mod_a, bootstrap=True)
        dag.add_node("B", op="quad", module=mod_b, bootstrap=False)
        dag.add_node("C", op="quad", module=mod_c, bootstrap=True)
        dag.add_node("D", op="linear_transform", module=mod_d, bootstrap=False)

        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        dag.add_edge("C", "D")

        ctx = _make_mock_context()
        placer = BootstrapPlacer(net=None, network_dag=dag, context=ctx)
        placer.place_bootstraps()

        assert "boot_0" in dag.nodes
        assert "boot_1" in dag.nodes

        # A -> boot_0 -> B -> C -> boot_1 -> D
        assert dag.has_edge("A", "boot_0")
        assert dag.has_edge("boot_0", "B")
        assert dag.has_edge("B", "C")
        assert dag.has_edge("C", "boot_1")
        assert dag.has_edge("boot_1", "D")

    def test_no_bootstrap_when_none_marked(self):
        """No bootstrap nodes inserted when no node is marked."""
        dag = _build_simple_dag()
        # Unmark B
        dag.nodes["B"]["bootstrap"] = False
        ctx = _make_mock_context()

        placer = BootstrapPlacer(net=None, network_dag=dag, context=ctx)
        placer.place_bootstraps()

        boot_nodes = [n for n in dag.nodes if n.startswith("boot_")]
        assert len(boot_nodes) == 0

    def test_bootstrap_node_depth_is_zero(self):
        """Bootstrap node has depth=0 (it's a single-level operation in the graph)."""
        dag = _build_simple_dag()
        ctx = _make_mock_context()

        placer = BootstrapPlacer(net=None, network_dag=dag, context=ctx)
        placer.place_bootstraps()

        assert dag.nodes["boot_0"]["depth"] == 0

    def test_bootstrap_node_not_marked_for_bootstrap(self):
        """The boot node itself should not be marked for bootstrap."""
        dag = _build_simple_dag()
        ctx = _make_mock_context()

        placer = BootstrapPlacer(net=None, network_dag=dag, context=ctx)
        placer.place_bootstraps()

        assert dag.nodes["boot_0"]["bootstrap"] is False


class TestTopologicalSortAfterInsertion:
    def test_topological_sort_valid(self):
        """Topological sort succeeds after bootstrap insertion."""
        dag = _build_simple_dag()
        ctx = _make_mock_context()

        placer = BootstrapPlacer(net=None, network_dag=dag, context=ctx)
        placer.place_bootstraps()

        topo = list(nx.topological_sort(dag))
        assert len(topo) == 4  # A, B, boot_0, C

    def test_topological_order_preserved(self):
        """Bootstrap node appears between its parent and children in topo order."""
        dag = _build_simple_dag()
        ctx = _make_mock_context()

        placer = BootstrapPlacer(net=None, network_dag=dag, context=ctx)
        placer.place_bootstraps()

        topo = list(nx.topological_sort(dag))
        assert topo.index("B") < topo.index("boot_0")
        assert topo.index("boot_0") < topo.index("C")

    def test_dag_remains_acyclic(self):
        """Graph is still a DAG after insertion."""
        dag = _build_simple_dag()
        ctx = _make_mock_context()

        placer = BootstrapPlacer(net=None, network_dag=dag, context=ctx)
        placer.place_bootstraps()

        assert nx.is_directed_acyclic_graph(dag)

    def test_forking_dag_acyclic(self):
        """Forking DAG remains acyclic after bootstrap insertion."""
        dag = _build_forking_dag()
        ctx = _make_mock_context()

        placer = BootstrapPlacer(net=None, network_dag=dag, context=ctx)
        placer.place_bootstraps()

        assert nx.is_directed_acyclic_graph(dag)
        topo = list(nx.topological_sort(dag))
        assert len(topo) == 5  # A, B, boot_0, C, D


# ---------------------------------------------------------------------------
# Integration test: compile a model that requires bootstraps
# ---------------------------------------------------------------------------


class TestBootstrapIntegration:
    def test_bootstrap_nodes_in_compiled_dag(self):
        """Compile a model with a short logq chain that triggers bootstraps.

        Verifies that boot_* nodes appear in the DAG after compilation.
        """
        import orion.nn as on
        from orion.params import CKKSParams
        from orion.compiler import Compiler

        # Short logq chain: l_eff = 2 (only 2 usable levels).
        # MLP with 2 linear layers + 1 quad = 3 depth -> bootstraps needed.
        short_params = CKKSParams(
            logn=13,
            logq=[29, 26, 26],
            logp=[29, 29],
            logscale=26,
            h=8192,
            ring_type="conjugate_invariant",
        )

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

        torch.manual_seed(42)
        net = TinyMLP()
        compiler = Compiler(net, short_params)
        compiler.fit(torch.randn(1, 1, 28, 28))

        # Access the internal DAG to verify bootstrap insertion.
        # We need to replicate the compile() steps up to bootstrap placement.
        from orion.core.network_dag import NetworkDAG
        from orion.core.fuser import Fuser

        network_dag = NetworkDAG(compiler._traced)
        network_dag.build_dag()

        for module in net.modules():
            if hasattr(module, "init_orion_params") and callable(
                module.init_orion_params
            ):
                module.init_orion_params()

        for module in net.modules():
            if hasattr(module, "update_params") and callable(
                module.update_params
            ):
                module.update_params()

        for module in net.modules():
            if hasattr(module, "he_mode"):
                module.scheme = compiler

        if compiler.config.fuse_modules:
            fuser = Fuser(network_dag)
            fuser.fuse_modules()
            network_dag.remove_fused_batchnorms()

        topo_sort = list(network_dag.topological_sort())
        from orion.nn.linear import LinearTransform
        last_linear = None
        for node in reversed(topo_sort):
            module = network_dag.nodes[node]["module"]
            if isinstance(module, LinearTransform):
                last_linear = node
                break

        for node in topo_sort:
            module = network_dag.nodes[node]["module"]
            if isinstance(module, LinearTransform):
                module.generate_diagonals(last=(node == last_linear))

        network_dag.find_residuals()

        l_eff = len(compiler.params.get_logq()) - 1
        btp_solver = BootstrapSolver(
            net, network_dag, l_eff=l_eff, context=compiler._context
        )
        input_level, num_bootstraps, bootstrapper_slots = btp_solver.solve()

        # With l_eff=2 and 3 depth, we should need at least 1 bootstrap
        assert num_bootstraps > 0, (
            f"Expected bootstraps with l_eff={l_eff}, got {num_bootstraps}"
        )

        btp_placer = BootstrapPlacer(net, network_dag, compiler._context)
        btp_placer.place_bootstraps()

        # Verify bootstrap nodes exist in the DAG
        boot_nodes = [
            n for n in network_dag.nodes
            if network_dag.nodes[n].get("op") == "bootstrap"
        ]
        assert len(boot_nodes) == num_bootstraps, (
            f"Expected {num_bootstraps} boot nodes, found {len(boot_nodes)}: "
            f"{boot_nodes}"
        )

        # Verify each boot node has correct attributes
        for bn in boot_nodes:
            attrs = network_dag.nodes[bn]
            assert attrs["op"] == "bootstrap"
            assert isinstance(attrs["module"], Bootstrap)
            assert attrs["level"] is not None
            assert attrs["depth"] == 0
            assert attrs["bootstrap"] is False

        # Verify DAG is still valid
        assert nx.is_directed_acyclic_graph(network_dag)
        topo = list(nx.topological_sort(network_dag))
        assert len(topo) > 0

        # Verify each boot node has at least one predecessor and one successor
        for bn in boot_nodes:
            preds = list(network_dag.predecessors(bn))
            succs = list(network_dag.successors(bn))
            assert len(preds) >= 1, f"{bn} has no predecessors"
            assert len(succs) >= 1, f"{bn} has no successors"

        del compiler
        import gc
        gc.collect()
