import gc

from orion import CKKSParams, Client


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

    client = Client(params)

    try:
        aux_moduli = client.backend.GetAuxModuliChain()

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
        del client
        gc.collect()
