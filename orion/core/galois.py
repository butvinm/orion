"""Pure Python implementation of Lattigo's BSGS Galois element computation.

Reimplements the algorithm from:
  lattigo/v6/circuits/common/lintrans/lintrans.go

Computes required Galois elements from diagonal indices + BSGS ratio
without creating any Lattigo objects.

Validated against Lattigo output in experiments/07_galois_elements_python/.
"""

import math

# Lattigo constant: ring/ring.go line 19
GALOIS_GEN = 5


def mod_exp(base: int, exp: int, mod: int) -> int:
    """Modular exponentiation: base^exp mod mod."""
    return pow(base, exp, mod)


def galois_element(k: int, nth_root: int) -> int:
    """Compute GaloisGen^k mod NthRoot.

    Reimplements rlwe.Parameters.GaloisElement(k).
    """
    k_masked = k & (nth_root - 1)
    return mod_exp(GALOIS_GEN, k_masked, nth_root)


def galois_elements(rotations: list[int], nth_root: int) -> list[int]:
    """Convert rotation indices to Galois elements.

    Reimplements rlwe.Parameters.GaloisElements(k).
    """
    return [galois_element(k, nth_root) for k in rotations]


def bsgs_index(
    non_zero_diags: list[int], slots: int, n1: int
) -> tuple[dict[int, list[int]], list[int], list[int]]:
    """Baby-step giant-step decomposition of diagonal indices.

    Returns (index_map, rot_n1, rot_n2) where:
      - index_map: giant_step -> [baby_steps]
      - rot_n1: sorted unique giant-step rotations
      - rot_n2: sorted unique baby-step rotations

    Reimplements lintrans.BSGSIndex().
    """
    index = {}
    rot_n1_set = set()
    rot_n2_set = set()

    for rot in non_zero_diags:
        rot = rot & (slots - 1)  # normalize to [0, slots)
        idx_n1 = ((rot // n1) * n1) & (slots - 1)  # giant-step
        idx_n2 = rot & (n1 - 1)  # baby-step

        if idx_n1 not in index:
            index[idx_n1] = []
        index[idx_n1].append(idx_n2)

        rot_n1_set.add(idx_n1)
        rot_n2_set.add(idx_n2)

    # Sort baby-steps within each giant-step group
    for k in index:
        index[k].sort()

    return index, sorted(rot_n1_set), sorted(rot_n2_set)


def find_best_bsgs_ratio(
    non_zero_diags: list[int], max_n: int, log_max_ratio: int
) -> int:
    """Find optimal N1 for baby-step giant-step.

    Reimplements lintrans.FindBestBSGSRatio().
    """
    max_ratio = float(1 << log_max_ratio)

    n1 = 1
    while n1 < max_n:
        _, rot_n1, rot_n2 = bsgs_index(non_zero_diags, max_n, n1)

        nb_n1 = len(rot_n1) - 1
        nb_n2 = len(rot_n2) - 1

        if nb_n1 == 0:
            n1 <<= 1
            continue

        ratio = nb_n2 / nb_n1

        if ratio == max_ratio:
            return n1

        if ratio > max_ratio:
            return max(1, n1 // 2)

        n1 <<= 1

    return 1


def compute_galois_elements(
    diag_indices: list[int],
    slots: int,
    log_bsgs_ratio: int,
    nth_root: int,
) -> list[int]:
    """Compute required Galois elements for a linear transform.

    Reimplements lintrans.LinearTransformation.GaloisElements().

    Args:
        diag_indices: Non-zero diagonal indices of the matrix
        slots: Number of CKKS slots (2^logslots)
        log_bsgs_ratio: Log2 of BSGS ratio (from CKKSParams). If < 0, BSGS disabled.
        nth_root: Ring NthRoot = 2 * ring_degree = 2 * 2^logn

    Returns:
        Sorted list of unique Galois elements needed for evaluation.
    """
    if log_bsgs_ratio < 0:
        # BSGS disabled: naive approach, N1 = slots
        _, _, rot_n2 = bsgs_index(diag_indices, slots, slots)
        return sorted(set(galois_element(r, nth_root) for r in rot_n2))

    # BSGS enabled
    n1 = find_best_bsgs_ratio(diag_indices, slots, log_bsgs_ratio)
    _, rot_n1, rot_n2 = bsgs_index(diag_indices, slots, n1)

    all_rotations = list(set(rot_n1 + rot_n2))
    return sorted(set(galois_element(r, nth_root) for r in all_rotations))


def nth_root_for_ring(logn: int, ring_type: str = "conjugate_invariant") -> int:
    """Compute the NthRoot for the given ring parameters.

    Standard ring:            NthRoot = 2^(logn+1) = 2N
    ConjugateInvariant ring:  NthRoot = 2^(logn+2) = 4N
    """
    if ring_type == "standard":
        return 1 << (logn + 1)
    else:  # conjugate_invariant
        return 1 << (logn + 2)


def compute_galois_elements_for_linear_transform(
    diag_indices_per_block: dict[tuple[int, int], list[int]],
    slots: int,
    bsgs_ratio: float,
    logn: int,
    ring_type: str = "conjugate_invariant",
) -> set[int]:
    """Compute Galois elements for all blocks of a LinearTransform module.

    This is the high-level function matching what the compiler needs.

    Args:
        diag_indices_per_block: {(row, col): [diag_indices]} from module.diagonals
        slots: Number of CKKS slots
        bsgs_ratio: BSGS ratio (e.g., 2.0)
        logn: Ring logN parameter
        ring_type: "standard" or "conjugate_invariant"

    Returns:
        Set of all required Galois elements across all blocks.
    """
    nth_root = nth_root_for_ring(logn, ring_type)
    log_bsgs = int(math.log2(bsgs_ratio)) if bsgs_ratio > 0 else -1

    all_galois = set()
    for (_row, _col), diags in diag_indices_per_block.items():
        gels = compute_galois_elements(list(diags), slots, log_bsgs, nth_root)
        all_galois.update(gels)

    return all_galois
