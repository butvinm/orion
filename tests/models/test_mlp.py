import gc

import torch
import pytest
import orion
import orion.models as models
from orion.core.utils import get_mnist_datasets, mae


MLP_PARAMS = orion.CKKSParams(
    logn=13,
    logq=(29, 26, 26, 26, 26, 26),
    logp=(29, 29),
    logscale=26,
    h=8192,
    ring_type="conjugate_invariant",
)


def _cleanup():
    gc.collect()


@pytest.mark.skip(reason="Python evaluator removed — Phase 2 provides Go evaluator")
def test_mlp():
    pass
