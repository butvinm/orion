"""Unit tests for orion.core.galois — pure Python Galois element computation."""

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


