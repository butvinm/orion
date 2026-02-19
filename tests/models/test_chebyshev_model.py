"""Integration test for a model using Chebyshev-based activation (Sigmoid).

This exercises code paths that Quad-only models skip:
- Chebyshev.fit() (polynomial fitting with minimax)
- Chebyshev.set_depth() compatibility with Evaluator._reconstruct_modules
- Chebyshev.compile() (polynomial generation)
- Chebyshev polynomial evaluation during FHE inference
"""

import gc

import torch

from orion.params import CKKSParams
from orion.compiled_model import CompiledModel, EvalKeys
from orion.compiler import Compiler
from orion.client import Client, CipherText
from orion.evaluator import Evaluator
import orion.nn as on


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


def test_sigmoid_compile_and_reconstruct():
    """Compiler -> CompiledModel -> Evaluator roundtrip with Sigmoid.

    This specifically tests that Evaluator._reconstruct_modules can
    call set_depth() on Chebyshev subclasses without crashing.
    """
    torch.manual_seed(42)
    net = SigmoidMLP()
    net.eval()

    compiler = Compiler(net, SIGMOID_PARAMS)
    compiler.fit(torch.randn(1, 1, 28, 28))
    compiled = compiler.compile()

    # Verify Chebyshev metadata is present
    has_chebyshev = any(
        m.get("type") == "Chebyshev"
        for m in compiled.module_metadata.values()
    )
    assert has_chebyshev, "Expected Chebyshev module in metadata"

    compiled_bytes = compiled.to_bytes()
    del compiler
    _cleanup()

    # Client: generate keys
    compiled = CompiledModel.from_bytes(compiled_bytes)
    client = Client(compiled.params)
    keys = client.generate_keys(compiled.manifest)
    keys_bytes = keys.to_bytes()
    del client
    _cleanup()

    # Evaluator: reconstruct modules — this is the critical test.
    # Before the fix, set_depth(depth) on Sigmoid would crash with
    # TypeError: set_depth() takes 1 positional argument but 2 were given.
    compiled = CompiledModel.from_bytes(compiled_bytes)
    keys = EvalKeys.from_bytes(keys_bytes)
    net_eval = SigmoidMLP()
    evaluator = Evaluator(net_eval, compiled, keys)

    # Verify the Sigmoid module has the correct depth from metadata
    sigmoid_meta = next(
        m for m in compiled.module_metadata.values()
        if m["type"] == "Chebyshev"
    )
    assert net_eval.act1.depth == sigmoid_meta["depth"]
    assert net_eval.act1.level == sigmoid_meta["level"]

    del evaluator
    _cleanup()


def test_sigmoid_full_roundtrip():
    """Full Compiler -> Client -> Evaluator -> Client roundtrip with Sigmoid."""
    torch.manual_seed(42)
    net = SigmoidMLP()
    net.eval()
    inp = torch.randn(1, 1, 28, 28)
    out_clear = net(inp)

    # Compile
    compiler = Compiler(net, SIGMOID_PARAMS)
    compiler.fit(inp)
    compiled = compiler.compile()
    compiled_bytes = compiled.to_bytes()
    del compiler
    _cleanup()

    # Client: keys + encrypt
    compiled = CompiledModel.from_bytes(compiled_bytes)
    client = Client(compiled.params)
    keys = client.generate_keys(compiled.manifest)
    pt = client.encode(inp, level=compiled.input_level)
    ct = client.encrypt(pt)
    ct_bytes = ct.to_bytes()
    keys_bytes = keys.to_bytes()
    sk_bytes = client.secret_key
    del client
    _cleanup()

    # Evaluator: run
    compiled = CompiledModel.from_bytes(compiled_bytes)
    keys = EvalKeys.from_bytes(keys_bytes)
    net_eval = SigmoidMLP()
    evaluator = Evaluator(net_eval, compiled, keys)
    ct_in = CipherText.from_bytes(ct_bytes, evaluator.backend)
    ct_out = evaluator.run(ct_in)
    ct_out_bytes = ct_out.to_bytes()
    del evaluator
    _cleanup()

    # Client: decrypt
    client = Client(compiled.params, secret_key=sk_bytes)
    ct_result = CipherText.from_bytes(ct_out_bytes, client.backend)
    pt_result = client.decrypt(ct_result)
    out_fhe = client.decode(pt_result)

    dist = (out_clear.detach() - out_fhe[:, :10].float()).abs().mean()
    del client
    _cleanup()

    assert dist < 1.0, f"MAE {dist:.6f} exceeds threshold"
