"""Tests for auxiliary moduli chain from the old Lattigo backend.

Skipped: old Lattigo backend cannot coexist with orionclient library.
Will be removed in Phase 6.
"""

import gc

import pytest

from orion import CKKSParams
from orion.backend.lattigo import bindings as lgo
from orion.backend.python import parameters

pytestmark = pytest.mark.skip(reason="old backend tests, removed in Phase 6")


def test_aux_moduli_chain():
    """
    Test that the GetAuxModuliChain function correctly returns auxiliary primes
    (P primes) with the expected bit sizes as specified in LogP.
    """
    params = CKKSParams(
        logn=14,
        logq=(45, 30, 30, 30, 30, 45),
        logp=(50, 51, 52),
        logscale=30,
        h=192,
        ring_type="standard",
    )

    new_params = parameters.NewParameters.from_ckks_params(params)
    backend = lgo.LattigoLibrary()
    backend.setup_bindings(new_params)

    try:
        aux_moduli = backend.GetAuxModuliChain()

        expected_count = len(params.logp)
        assert len(aux_moduli) == expected_count, (
            f"Expected {expected_count} auxiliary primes, got {len(aux_moduli)}"
        )

        for i, (prime, expected_bits) in enumerate(zip(aux_moduli, params.logp)):
            actual_bits = prime.bit_length()
            assert actual_bits == expected_bits or actual_bits == expected_bits + 1, (
                f"Auxiliary prime {i} has {actual_bits} bits, expected {expected_bits} "
                f"(prime value: {prime})"
            )

    finally:
        gc.collect()
