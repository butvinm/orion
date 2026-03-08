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

import torch
import orion_compiler
import orion_compiler.nn as on


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


class DeepMLP(on.Module):
    """Flatten -> Linear(784,32) -> Quad -> Linear(32,32) -> Quad -> Linear(32,10)

    Two hidden layers force 1 bootstrap when compiled with a short logq chain
    (e.g. logq=[55,40,40,40] — only 3 levels, but 5 are needed).
    """

    def __init__(self):
        super().__init__()
        self.flatten = on.Flatten()
        self.fc1 = on.Linear(784, 32)
        self.act1 = on.Quad()
        self.fc2 = on.Linear(32, 32)
        self.act2 = on.Quad()
        self.fc3 = on.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


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


def eval_chebyshev_torch(x, coeffs):
    """Evaluate Chebyshev polynomial: sum(c_k * T_k(x)) element-wise."""
    coeffs_t = [torch.tensor(c, dtype=x.dtype) for c in coeffs]
    if len(coeffs) == 0:
        return torch.zeros_like(x)
    if len(coeffs) == 1:
        return torch.full_like(x, coeffs[0])

    T_prev = torch.ones_like(x)  # T_0(x) = 1
    T_curr = x.clone()           # T_1(x) = x
    result = coeffs_t[0] * T_prev + coeffs_t[1] * T_curr

    for k in range(2, len(coeffs)):
        T_next = 2 * x * T_curr - T_prev
        result = result + coeffs_t[k] * T_next
        T_prev = T_curr
        T_curr = T_next

    return result


def compute_expected_fhe(net, input_tensor, fused=True):
    """Compute expected output matching what the Go evaluator produces in exact arithmetic.

    Uses the compiled model's weights and polynomial approximations instead of
    exact activation functions. This accounts for:
    - fuse_modules: when True, prescale/constant absorbed into linear layer weights
    - When False, prescale/constant applied before polynomial evaluation
    - Chebyshev polynomial approximation instead of exact sigmoid/SiLU/etc.
    """
    import torch.nn.functional as F

    with torch.no_grad():
        x = input_tensor.flatten()

        for name, module in net.named_children():
            if isinstance(module, on.Flatten):
                pass  # already flattened
            elif isinstance(module, on.Linear):
                # Use on_weight/on_bias (fused if fuser ran, same as original otherwise)
                x = F.linear(x, module.on_weight, module.on_bias)
            elif isinstance(module, on.Quad):
                x = x * x
            elif hasattr(module, 'coeffs') and module.coeffs is not None:
                # Chebyshev activation (Sigmoid, SiLU, GELU, etc.)
                if not fused:
                    # When not fused, apply prescale then constant before polynomial
                    prescale = getattr(module, 'prescale', 1)
                    constant = getattr(module, 'constant', 0)
                    if prescale != 1:
                        x = x * prescale
                    if constant != 0:
                        x = x + constant
                x = eval_chebyshev_torch(x, module.coeffs)
            else:
                raise ValueError(f"Unsupported module type: {type(module).__name__}")

    return x


def generate_fixture(name, net_cls, params, config=None):
    """Compile a model and write .orion + input/expected JSON files."""
    torch.manual_seed(42)
    net = net_cls()
    net.eval()

    # Compile
    compiler = orion_compiler.Compiler(net, params, config=config)
    compiler.fit(torch.randn(1, 1, 28, 28))
    compiled = compiler.compile()
    fused = config.fuse_modules if config else True

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

    # Compute expected output matching FHE evaluator's exact-arithmetic computation.
    # Uses fused on_weight/on_bias for linear layers, Chebyshev polynomial for activations.
    # This differs from cleartext forward (which uses original weight + exact sigmoid).
    expected = compute_expected_fhe(net, input_tensor, fused=fused)
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
    mlp_params = orion_compiler.CKKSParams(
        logn=13,
        logq=[29, 26, 26, 26, 26, 26],
        logp=[29, 29],
        logscale=26,
        h=8192,
        ring_type="conjugate_invariant",
    )

    # SigmoidMLP: needs more levels for degree-7 polynomial
    sigmoid_params = orion_compiler.CKKSParams(
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

    # SigmoidMLP unfused: same params but fuse_modules=False
    # Needs extra level because prescale adds 1 to depth
    sigmoid_unfused_params = orion_compiler.CKKSParams(
        logn=13,
        logq=[29, 26, 26, 26, 26, 26, 26, 26, 26],
        logp=[29, 29],
        logscale=26,
        h=8192,
        ring_type="conjugate_invariant",
    )
    unfused_config = orion_compiler.CompilerConfig(fuse_modules=False)

    print("Generating SigmoidMLP unfused fixture...")
    generate_fixture("sigmoid_unfused", SigmoidMLP, sigmoid_unfused_params, config=unfused_config)

    # DeepMLP with bootstrap: logn=14, short logq chain forces 1 bootstrap.
    # Uses standard ring (required for bootstrap) and h=192.
    bootstrap_params = orion_compiler.CKKSParams(
        logn=14,
        logq=[55, 40, 40, 40],
        logp=[61, 61],
        logscale=40,
        h=192,
        ring_type="standard",
        boot_logp=[61, 61, 61, 61, 61, 61],
    )

    print("Generating DeepMLP with bootstrap fixture...")
    generate_fixture("bootstrap_mlp", DeepMLP, bootstrap_params)

    print("Done!")


if __name__ == "__main__":
    main()
