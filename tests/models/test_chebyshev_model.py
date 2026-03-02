"""Integration test for a model using Chebyshev-based activation (Sigmoid).

This exercises code paths that Quad-only models skip:
- Chebyshev.fit() (polynomial fitting with minimax)
- Chebyshev.set_depth() compatibility with Evaluator._reconstruct_modules
- Chebyshev.compile() (polynomial generation)
- Chebyshev polynomial evaluation during FHE inference
"""

import gc

import torch
import pytest

from orion_compiler.params import CKKSParams
from orion_compiler.compiler import Compiler
import orion_compiler.nn as on


# Deeper moduli chain to accommodate Chebyshev polynomial depth.
# Sigmoid(degree=7) consumes ceil(log2(8))=3 multiplicative levels,
# plus 1 per Linear layer = 5 total minimum.
SIGMOID_PARAMS = CKKSParams(
    logn=13,
    logq=[29, 26, 26, 26, 26, 26, 26, 26],
    logp=[29, 29],
    logscale=26,
    h=8192,
    ring_type="conjugate_invariant",
)


def _cleanup():
    gc.collect()


class SigmoidMLP(on.Module):
    """Tiny MLP with Sigmoid activation (Chebyshev polynomial)."""

    def __init__(self):
        super().__init__()
        self.flatten = on.Flatten()
        self.fc1 = on.Linear(784, 32)
        self.act1 = on.Sigmoid(degree=7)
        self.fc2 = on.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        return self.fc2(x)


def test_sigmoid_compile_produces_polynomial_nodes():
    """Compiler produces polynomial nodes for Sigmoid activation."""
    torch.manual_seed(42)
    net = SigmoidMLP()
    net.eval()

    compiler = Compiler(net, SIGMOID_PARAMS)
    compiler.fit(torch.randn(1, 1, 28, 28))
    compiled = compiler.compile()

    # Verify polynomial node is present in graph
    has_poly = any(
        n.op == "polynomial" for n in compiled.graph.nodes
    )
    assert has_poly, "Expected polynomial node in graph"

    # Verify polynomial coefficients are inline
    poly_node = next(
        n for n in compiled.graph.nodes if n.op == "polynomial"
    )
    assert "coeffs" in poly_node.config
    assert "basis" in poly_node.config
    assert poly_node.config["basis"] == "chebyshev"

    del compiler
    _cleanup()


@pytest.mark.skip(reason="Python evaluator removed — Phase 2 provides Go evaluator")
def test_sigmoid_full_roundtrip():
    """Full Compiler -> Client -> Evaluator -> Client roundtrip with Sigmoid."""
    pass
