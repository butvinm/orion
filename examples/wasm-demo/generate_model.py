#!/usr/bin/env python3
"""Generate a demo .orion model file for the WASM browser demo.

Usage:
    python generate_model.py [output_path]

Compiles a SimpleMLP (784 -> 32 -> 10) with random weights.
Accuracy doesn't matter — this validates the end-to-end pipeline.
"""

import sys
import torch

from orion_compiler import Compiler, CKKSParams
import orion_compiler.nn as on


class SimpleMLP(on.Module):
    """Tiny MLP for fast demo. Input: 784 floats, output: 10 floats."""

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


PARAMS = CKKSParams(
    logn=13,
    logq=[29, 26, 26, 26, 26, 26],
    logp=[29, 29],
    logscale=26,
    h=8192,
    ring_type="conjugate_invariant",
)


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "model.orion"

    torch.manual_seed(42)
    net = SimpleMLP()
    compiler = Compiler(net, PARAMS)
    compiler.fit(torch.randn(32, 1, 28, 28))
    compiled = compiler.compile()

    model_bytes = compiled.to_bytes()
    with open(output_path, "wb") as f:
        f.write(model_bytes)

    print(f"Model written to {output_path} ({len(model_bytes)} bytes)")
    print(f"Input level: {compiled.input_level}")
    print(f"Galois elements: {len(compiled.manifest.galois_elements)}")
    print(f"Needs RLK: {compiled.manifest.needs_rlk}")
    print(f"Bootstrap slots: {compiled.manifest.bootstrap_slots}")
    print(f"Graph nodes: {len(compiled.graph.nodes)}")
    print(f"Graph edges: {len(compiled.graph.edges)}")


if __name__ == "__main__":
    main()
