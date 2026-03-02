"""Unit tests for orion.core.galois — pure Python Galois element computation."""

import gc

import torch
import pytest

from orion.core.galois import (
    GALOIS_GEN,
    galois_element,
    bsgs_index,
    compute_galois_elements,
    nth_root_for_ring,
    compute_galois_elements_for_linear_transform,
)


# -----------------------------------------------------------------------
# galois_element
# -----------------------------------------------------------------------


class TestGaloisElement:
    def test_rotation_1(self):
        """galois_element(1, 2^15) = GaloisGen^1 mod 2^15 = 5."""
        nth_root = 2**15
        assert galois_element(1, nth_root) == 5

    def test_rotation_0(self):
        """galois_element(0, ...) = GaloisGen^0 = 1 for any nth_root."""
        for logn in [10, 13, 15]:
            nth_root = 2 ** (logn + 2)
            assert galois_element(0, nth_root) == 1

    def test_rotation_2(self):
        """galois_element(2, nth_root) = GaloisGen^2 mod nth_root = 25."""
        nth_root = 2**15
        assert galois_element(2, nth_root) == GALOIS_GEN**2 % nth_root

    def test_negative_rotation_wraps(self):
        """Negative k is masked by (nth_root - 1), producing a valid result."""
        nth_root = 2**15
        result = galois_element(-1, nth_root)
        # -1 & (2^15 - 1) = 2^15 - 1 = 32767
        expected = pow(GALOIS_GEN, 32767, nth_root)
        assert result == expected

    def test_result_always_odd(self):
        """Galois elements are always odd (GaloisGen=5 is odd, odd^k is odd)."""
        nth_root = 2**15
        for k in range(20):
            assert galois_element(k, nth_root) % 2 == 1


# -----------------------------------------------------------------------
# nth_root_for_ring
# -----------------------------------------------------------------------


class TestNthRootForRing:
    def test_standard_ring(self):
        """Standard ring: NthRoot = 2^(logn+1)."""
        assert nth_root_for_ring(13, "standard") == 2**14
        assert nth_root_for_ring(14, "standard") == 2**15

    def test_conjugate_invariant_ring(self):
        """ConjugateInvariant ring: NthRoot = 2^(logn+2)."""
        assert nth_root_for_ring(13, "conjugate_invariant") == 2**15
        assert nth_root_for_ring(14, "conjugate_invariant") == 2**16

    def test_default_is_conjugate_invariant(self):
        """Default ring_type is conjugate_invariant."""
        assert nth_root_for_ring(13) == nth_root_for_ring(13, "conjugate_invariant")


# -----------------------------------------------------------------------
# bsgs_index
# -----------------------------------------------------------------------


class TestBsgsIndex:
    def test_single_diagonal(self):
        """Single diagonal index decomposes correctly."""
        slots = 16
        n1 = 4
        index_map, rot_n1, rot_n2 = bsgs_index([5], slots, n1)
        # rot=5, idx_n1 = (5//4)*4 = 4, idx_n2 = 5 & 3 = 1
        assert rot_n1 == [4]
        assert rot_n2 == [1]
        assert index_map == {4: [1]}

    def test_multiple_diagonals(self):
        """Multiple diagonals decompose into baby/giant steps."""
        slots = 16
        n1 = 4
        # diags: 0, 1, 4, 5
        index_map, rot_n1, rot_n2 = bsgs_index([0, 1, 4, 5], slots, n1)
        # rot=0: n1=0, n2=0
        # rot=1: n1=0, n2=1
        # rot=4: n1=4, n2=0
        # rot=5: n1=4, n2=1
        assert rot_n1 == [0, 4]
        assert rot_n2 == [0, 1]
        assert index_map[0] == [0, 1]
        assert index_map[4] == [0, 1]

    def test_negative_diag_wraps(self):
        """Negative diagonal index wraps via & (slots - 1)."""
        slots = 16
        n1 = 4
        # -1 & 15 = 15
        index_map, rot_n1, rot_n2 = bsgs_index([-1], slots, n1)
        # rot=15: n1 = (15//4)*4 = 12, n2 = 15 & 3 = 3
        assert rot_n1 == [12]
        assert rot_n2 == [3]

    def test_baby_steps_sorted(self):
        """Baby steps within each giant-step group are sorted."""
        slots = 64
        n1 = 8
        diags = [3, 1, 7, 2, 10, 15]
        index_map, _, _ = bsgs_index(diags, slots, n1)
        for group in index_map.values():
            assert group == sorted(group)


# -----------------------------------------------------------------------
# compute_galois_elements
# -----------------------------------------------------------------------


class TestComputeGaloisElements:
    def test_bsgs_disabled(self):
        """With log_bsgs_ratio < 0, BSGS is disabled (naive approach)."""
        diags = [0, 1, 2, 3]
        slots = 16
        nth_root = 2**10  # arbitrary
        result = compute_galois_elements(diags, slots, -1, nth_root)
        # Should produce sorted unique Galois elements
        assert result == sorted(result)
        assert len(result) == len(set(result))

    def test_bsgs_enabled(self):
        """With log_bsgs_ratio >= 0, BSGS is used."""
        diags = [0, 1, 2, 3, 4, 5, 6, 7]
        slots = 64
        nth_root = 2**10
        result_bsgs = compute_galois_elements(diags, slots, 1, nth_root)
        result_naive = compute_galois_elements(diags, slots, -1, nth_root)
        # BSGS may produce more elements (giant-step rotations added)
        # but both sets should be valid (BSGS is a superset of naive in general)
        assert len(result_bsgs) > 0
        assert len(result_naive) > 0

    def test_sorted_and_unique(self):
        """Output is always sorted with unique elements."""
        diags = [0, 3, 7, 15, 31]
        slots = 64
        nth_root = 2**10
        for log_bsgs in [-1, 0, 1, 2]:
            result = compute_galois_elements(diags, slots, log_bsgs, nth_root)
            assert result == sorted(set(result))

    def test_single_diagonal_zero(self):
        """Diagonal 0 (identity) still produces a Galois element for rotation 0."""
        diags = [0]
        slots = 16
        nth_root = 2**10
        result = compute_galois_elements(diags, slots, 1, nth_root)
        # rotation 0 -> galois_element(0, nth_root) = 1
        assert 1 in result


# -----------------------------------------------------------------------
# compute_galois_elements_for_linear_transform
# -----------------------------------------------------------------------


class TestComputeGaloisElementsForLinearTransform:
    def test_single_block(self):
        """Single block produces valid Galois elements."""
        diag_indices = {(0, 0): [0, 1, 2, 3]}
        slots = 64
        result = compute_galois_elements_for_linear_transform(
            diag_indices, slots, 2.0, 13, "conjugate_invariant"
        )
        assert isinstance(result, set)
        assert len(result) > 0

    def test_multiple_blocks_union(self):
        """Multiple blocks' Galois elements are unioned."""
        block_a = {(0, 0): [0, 1, 2]}
        block_b = {(0, 0): [0, 1, 2], (0, 1): [3, 4, 5]}
        result_a = compute_galois_elements_for_linear_transform(
            block_a, 64, 2.0, 13, "conjugate_invariant"
        )
        result_b = compute_galois_elements_for_linear_transform(
            block_b, 64, 2.0, 13, "conjugate_invariant"
        )
        # block_b is a superset of block_a's diags, so its elements should include a's
        assert result_a.issubset(result_b)

    def test_bsgs_disabled_via_zero_ratio(self):
        """bsgs_ratio=0 disables BSGS (log2(0) is undefined, uses -1)."""
        diag_indices = {(0, 0): [0, 1, 2]}
        # bsgs_ratio=0 -> log_bsgs = -1 (disabled)
        result = compute_galois_elements_for_linear_transform(
            diag_indices, 64, 0, 13, "conjugate_invariant"
        )
        assert isinstance(result, set)
        assert len(result) > 0


# -----------------------------------------------------------------------
# Integration test: compare against Lattigo via TransformEncoder
# -----------------------------------------------------------------------


class TestGaloisVsLattigo:
    """Compare Python Galois element computation against Lattigo's output.

    Uses the same approach as experiments/07_galois_elements_python/compare.py:
    compile a SimpleMLP, then compare per-LT Galois elements.
    """

    @pytest.fixture(scope="class")
    def compiled_model_data(self):
        """Compile a SimpleMLP and collect Galois data from both Python and Lattigo."""
        import orion.nn as on
        from orion.params import CKKSParams
        from orion.compiler import Compiler
        from orion.nn.module import Module
        from orion.nn.linear import LinearTransform
        from orion.core.network_dag import NetworkDAG
        from orion.core.fuser import Fuser
        from orion.core.auto_bootstrap import BootstrapSolver

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

        params = CKKSParams(
            logn=13,
            logq=[29, 26, 26, 26, 26, 26],
            logp=[29, 29],
            logscale=26,
            h=8192,
            ring_type="conjugate_invariant",
        )

        torch.manual_seed(42)
        net = SimpleMLP()
        compiler = Compiler(net, params)
        compiler.fit(torch.randn(1, 1, 28, 28))

        # Build DAG and generate diagonals (mirroring compile() internals)
        network_dag = NetworkDAG(compiler._traced)
        network_dag.build_dag()

        for module in net.modules():
            if hasattr(module, "init_orion_params") and callable(
                module.init_orion_params
            ):
                module.init_orion_params()
        for module in net.modules():
            if hasattr(module, "update_params") and callable(module.update_params):
                module.update_params()
        for module in net.modules():
            if isinstance(module, Module):
                module.scheme = compiler

        if compiler.config.fuse_modules:
            fuser = Fuser(network_dag)
            fuser.fuse_modules()
            network_dag.remove_fused_batchnorms()

        topo_sort = list(network_dag.topological_sort())

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

        # Run bootstrap solver to assign levels
        network_dag.find_residuals()
        l_eff = len(compiler.params.get_logq()) - 1
        BootstrapSolver(
            net, network_dag, l_eff=l_eff, context=compiler._context
        ).solve()

        # Collect Lattigo and Python Galois elements per LT
        slots = compiler.params.get_slots()
        logn = params.logn
        ring_type = params.ring_type
        nth_root = nth_root_for_ring(logn, ring_type)

        lt_comparisons = []
        for node in topo_sort:
            module = network_dag.nodes[node]["module"]
            if not isinstance(module, LinearTransform):
                continue

            # Lattigo path
            lt_ids = compiler._lt_evaluator.generate_transforms(module)
            lattigo_galois = set()
            for tid in lt_ids.values():
                keys = compiler._lt_evaluator.get_galois_elements(tid)
                lattigo_galois.update(keys)
            compiler._lt_evaluator.delete_transforms(lt_ids)

            # Python path
            diag_indices_per_block = {}
            for (row, col), diags_dict in module.diagonals.items():
                diag_indices_per_block[(row, col)] = list(diags_dict.keys())

            python_galois = compute_galois_elements_for_linear_transform(
                diag_indices_per_block,
                slots,
                module.bsgs_ratio,
                logn,
                ring_type,
            )

            lt_comparisons.append(
                {
                    "name": node,
                    "lattigo": lattigo_galois,
                    "python": python_galois,
                }
            )

        # Power-of-2 rotations
        lattigo_po2 = set()
        python_po2 = set()
        i = 1
        while i < slots:
            lattigo_po2.add(compiler.backend.GetGaloisElement(i))
            python_po2.add(galois_element(i, nth_root))
            i *= 2

        # Output rotations
        lattigo_out = set()
        python_out = set()
        for node in topo_sort:
            module = network_dag.nodes[node]["module"]
            if isinstance(module, LinearTransform) and module.output_rotations > 0:
                for i in range(1, module.output_rotations + 1):
                    rotation = slots // (2**i)
                    lattigo_out.add(compiler.backend.GetGaloisElement(rotation))
                    python_out.add(galois_element(rotation, nth_root))

        compiler.close()
        gc.collect()

        return {
            "lt_comparisons": lt_comparisons,
            "lattigo_po2": lattigo_po2,
            "python_po2": python_po2,
            "lattigo_out": lattigo_out,
            "python_out": python_out,
        }

    def test_per_lt_galois_elements_match(self, compiled_model_data):
        """Per-LinearTransform BSGS Galois elements match Lattigo."""
        for comp in compiled_model_data["lt_comparisons"]:
            assert comp["lattigo"] == comp["python"], (
                f"Mismatch for {comp['name']}: "
                f"only_lattigo={comp['lattigo'] - comp['python']}, "
                f"only_python={comp['python'] - comp['lattigo']}"
            )

    def test_power_of_2_rotations_match(self, compiled_model_data):
        """Power-of-2 rotation Galois elements match Lattigo."""
        assert compiled_model_data["lattigo_po2"] == compiled_model_data["python_po2"]

    def test_output_rotations_match(self, compiled_model_data):
        """Output rotation Galois elements match Lattigo."""
        assert compiled_model_data["lattigo_out"] == compiled_model_data["python_out"]
