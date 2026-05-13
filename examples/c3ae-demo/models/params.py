"""CKKS parameter sets for the C3AE FHE experiments.

Two no-bootstrap configurations with **identical multiplicative depth (15
levels)** so the experiment isolates the cost of doubling the ring degree from
``logn=15`` to ``logn=16``.

Security bound caveat: the bounds quoted below are the **dense / uniform-ternary**
HE Standard 128-bit limits, which are the *more conservative* of the two common
choices (dense vs. sparse-ternary with Hamming weight ``h=192``). The Lattigo
``CKKSParams`` default at the time of writing uses ``h=192`` (sparse), whose
permitted ``LogQP`` is **strictly larger** at the same logn (e.g. ~1232 vs. 881
at logn=15 in the HE Standard tables). Reporting against the dense bound is
therefore a safe upper-envelope: any config that satisfies the dense bound also
satisfies the sparse bound at ``h=192``. We report the dense bound for
documentation simplicity; the actual encryption uses the Lattigo default
(sparse, ``h=192``).

Arithmetic per config:

``logn15`` (verbatim copy of the demo params at
``examples/c3ae-demo/generate_model.py:23-29``):

* ``logq = [51] + [40] * 15``  →  ``sum(logq) = 51 + 15 * 40 = 651``
* ``logp = [50] * 4``           →  ``sum(logp) = 4 * 50 = 200``
* ``LogQP = 651 + 200 = 851``  ≤  ``881`` (128-bit *dense* bound at logn=15;
  sparse ``h=192`` bound is larger and also satisfied)
* No bootstrap (``boot_logp = None``).

``logn16`` (new — same multiplicative depth as ``logn15`` for a clean
ring-degree comparison):

* ``logq = [55] + [40] * 15``  →  ``sum(logq) = 55 + 15 * 40 = 655``
* ``logp = [55] * 6``           →  ``sum(logp) = 6 * 55 = 330``
* ``LogQP = 655 + 330 = 985``  ≤  ``1770`` (128-bit *dense* bound at logn=16;
  sparse ``h=192`` bound is larger and also satisfied)
* No bootstrap (``boot_logp = None``).

Both configs share ``log_default_scale = 40`` and ``ring_type = "standard"``.
"""

from orion_compiler import CKKSParams

PARAMS: dict[str, CKKSParams] = {
    "logn15": CKKSParams(
        logn=15,
        logq=[51, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
        logp=[50, 50, 50, 50],
        log_default_scale=40,
        ring_type="standard",
    ),
    "logn16": CKKSParams(
        logn=16,
        logq=[55] + [40] * 15,
        logp=[55] * 6,
        log_default_scale=40,
        ring_type="standard",
    ),
}
