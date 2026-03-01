"""Generate test fixtures for the Go evaluator package.

Produces .orion compiled model files and corresponding input/expected JSON
files for end-to-end testing.

Usage:
    cd evaluator/testdata
    python generate.py
"""

import json
import sys
import os

# Add project root to path so we can import orion
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import orion
import orion.nn as on


# ---------------------------------------------------------------------------
# Model definitions (small models for fast test fixtures)
# ---------------------------------------------------------------------------


class SimpleMLP(on.Module):
    """Flatten -> Linear(784,32) -> Quad -> Linear(32,10)"""

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


class SigmoidMLP(on.Module):
    """Flatten -> Linear(784,32) -> Sigmoid(degree=7) -> Linear(32,10)"""

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


def generate_fixture(name, net_cls, params):
    """Compile a model and write .orion + input/expected JSON files."""
    torch.manual_seed(42)
    net = net_cls()
    net.eval()

    # Compile
    compiler = orion.Compiler(net, params)
    compiler.fit(torch.randn(1, 1, 28, 28))
    compiled = compiler.compile()

    # Write compiled model
    outdir = os.path.dirname(__file__)
    orion_path = os.path.join(outdir, f"{name}.orion")
    with open(orion_path, "wb") as f:
        f.write(compiled.to_bytes())
    print(f"  wrote {orion_path} ({os.path.getsize(orion_path)} bytes)")

    # Generate input (deterministic)
    torch.manual_seed(42)
    input_tensor = torch.randn(1, 1, 28, 28)
    input_flat = input_tensor.flatten().tolist()

    input_path = os.path.join(outdir, f"{name}.input.json")
    with open(input_path, "w") as f:
        json.dump(input_flat, f)
    print(f"  wrote {input_path}")

    # Compute cleartext expected output
    torch.manual_seed(42)
    net2 = net_cls()
    net2.eval()
    with torch.no_grad():
        expected = net2(input_tensor)
    expected_flat = expected.flatten().tolist()

    expected_path = os.path.join(outdir, f"{name}.expected.json")
    with open(expected_path, "w") as f:
        json.dump(expected_flat, f)
    print(f"  wrote {expected_path}")

    # Cleanup
    del compiler
    import gc
    gc.collect()


def main():
    # SimpleMLP: logn=13, logq=[29,26,26,26,26,26], h=8192, conjugate_invariant
    mlp_params = orion.CKKSParams(
        logn=13,
        logq=[29, 26, 26, 26, 26, 26],
        logp=[29, 29],
        logscale=26,
        h=8192,
        ring_type="conjugate_invariant",
    )

    # SigmoidMLP: needs more levels for degree-7 polynomial
    sigmoid_params = orion.CKKSParams(
        logn=13,
        logq=[29, 26, 26, 26, 26, 26, 26, 26],
        logp=[29, 29],
        logscale=26,
        h=8192,
        ring_type="conjugate_invariant",
    )

    print("Generating SimpleMLP fixture...")
    generate_fixture("mlp", SimpleMLP, mlp_params)

    print("Generating SigmoidMLP fixture...")
    generate_fixture("sigmoid", SigmoidMLP, sigmoid_params)

    print("Done!")


if __name__ == "__main__":
    main()
