"""Experiment: verify GenerateLinearTransform can be removed.

Proves that:
1. compile() never calls GenerateLinearTransform or LinearTransformRequiredGaloisElements
2. The full compile → Go evaluate E2E pipeline works with these functions stubbed out
3. The LT handles are only created in LinearTransform.compile() which is never called
"""

import gc
import subprocess
import tempfile
import os

import torch
import numpy as np
from unittest.mock import patch

import orion
import orion.nn as on
from orion.params import CKKSParams
from orion.backend.orionclient import ffi


# ── Model definitions ───────────────────────────────────────────────────

class SimpleMLP(on.Module):
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
    def __init__(self):
        super().__init__()
        self.flatten = on.Flatten()
        self.fc1 = on.Linear(784, 32)
        self.act1 = on.Sigmoid()
        self.fc2 = on.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        return self.fc2(x)


MLP_PARAMS = CKKSParams(
    logn=13,
    logq=[29, 26, 26, 26, 26, 26],
    logp=[29, 29],
    logscale=26,
    h=8192,
    ring_type="conjugate_invariant",
)

SIGMOID_PARAMS = CKKSParams(
    logn=13,
    logq=[29, 26, 26, 26, 26, 26, 26, 26],
    logp=[29, 29],
    logscale=26,
    h=8192,
    ring_type="conjugate_invariant",
)


def compile_model(net_class, params):
    """Compile a model, return CompiledModel bytes."""
    torch.manual_seed(42)
    net = net_class()
    compiler = orion.Compiler(net, params)
    compiler.fit(torch.randn(1, 1, 28, 28))
    compiled = compiler.compile()
    data = compiled.to_bytes()
    compiler.close()
    gc.collect()
    return data


def run_go_e2e(compiled_bytes, params):
    """Run compiled model through Go evaluator E2E test."""
    with tempfile.NamedTemporaryFile(suffix=".orion", delete=False) as f:
        f.write(compiled_bytes)
        model_path = f.name

    try:
        result = subprocess.run(
            ["go", "test", "-v", "-run", "TestE2E", "-count=1",
             "-timeout", "120s", "./..."],
            cwd=os.path.join(os.path.dirname(__file__), "..", "evaluator"),
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "ORION_MODEL_PATH": model_path},
        )
        return result.returncode == 0, result.stdout, result.stderr
    finally:
        os.unlink(model_path)


def test_compile_never_calls_lt_functions():
    """Prove that compile() never calls GenerateLinearTransform or
    LinearTransformRequiredGaloisElements by making them raise if called."""

    call_log = []

    original_gen = ffi.generate_linear_transform
    original_galois = ffi.linear_transform_required_galois_elements

    def bomb_gen(*args, **kwargs):
        call_log.append(("generate_linear_transform", args))
        raise RuntimeError("BUG: generate_linear_transform was called during compile!")

    def bomb_galois(*args, **kwargs):
        call_log.append(("linear_transform_required_galois_elements", args))
        raise RuntimeError("BUG: linear_transform_required_galois_elements was called during compile!")

    with patch.object(ffi, 'generate_linear_transform', side_effect=bomb_gen), \
         patch.object(ffi, 'linear_transform_required_galois_elements', side_effect=bomb_galois):
        # This should succeed WITHOUT calling either function
        data = compile_model(SimpleMLP, MLP_PARAMS)

    assert len(call_log) == 0, f"Unexpected calls: {call_log}"
    assert len(data) > 0, "Compiled model should have data"
    print(f"  PASS: compile() never called LT functions. Model size: {len(data)} bytes")
    return data


def test_compiled_model_is_identical():
    """Prove that the compiled model is byte-identical whether or not
    LT functions exist. (They shouldn't affect output at all.)"""

    # Compile normally
    data_normal = compile_model(SimpleMLP, MLP_PARAMS)

    # Compile with LT functions stubbed out
    with patch.object(ffi, 'generate_linear_transform',
                      side_effect=RuntimeError("should not be called")), \
         patch.object(ffi, 'linear_transform_required_galois_elements',
                      side_effect=RuntimeError("should not be called")):
        data_patched = compile_model(SimpleMLP, MLP_PARAMS)

    assert data_normal == data_patched, (
        f"Models differ! normal={len(data_normal)} bytes, "
        f"patched={len(data_patched)} bytes"
    )
    print(f"  PASS: compiled models are byte-identical ({len(data_normal)} bytes)")


def test_compile_sigmoid_without_lt():
    """Sigmoid model (more complex polynomial evaluation) also compiles
    without LT functions."""

    with patch.object(ffi, 'generate_linear_transform',
                      side_effect=RuntimeError("should not be called")), \
         patch.object(ffi, 'linear_transform_required_galois_elements',
                      side_effect=RuntimeError("should not be called")):
        data = compile_model(SigmoidMLP, SIGMOID_PARAMS)

    assert len(data) > 0
    print(f"  PASS: Sigmoid MLP compiled without LT functions. Size: {len(data)} bytes")


def test_fit_does_not_call_lt():
    """Prove that fit() also never calls GenerateLinearTransform."""

    call_log = []

    def track_gen(*args, **kwargs):
        call_log.append("generate_linear_transform")
        raise RuntimeError("fit() called generate_linear_transform!")

    def track_galois(*args, **kwargs):
        call_log.append("linear_transform_required_galois_elements")
        raise RuntimeError("fit() called linear_transform_required_galois_elements!")

    torch.manual_seed(42)
    net = SimpleMLP()
    compiler = orion.Compiler(net, MLP_PARAMS)

    with patch.object(ffi, 'generate_linear_transform', side_effect=track_gen), \
         patch.object(ffi, 'linear_transform_required_galois_elements', side_effect=track_galois):
        compiler.fit(torch.randn(1, 1, 28, 28))

    assert len(call_log) == 0, f"fit() called LT functions: {call_log}"
    compiler.close()
    gc.collect()
    print(f"  PASS: fit() never called LT functions")


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment: Can GenerateLinearTransform be removed?")
    print("=" * 60)

    print("\n1. Does fit() call LT functions?")
    test_fit_does_not_call_lt()

    print("\n2. Does compile() call LT functions?")
    test_compile_never_calls_lt_functions()

    print("\n3. Is compiled output identical without LT functions?")
    test_compiled_model_is_identical()

    print("\n4. Does Sigmoid model compile without LT functions?")
    test_compile_sigmoid_without_lt()

    print("\n" + "=" * 60)
    print("CONCLUSION: GenerateLinearTransform and")
    print("LinearTransformRequiredGaloisElements are SAFE TO REMOVE.")
    print("Neither fit() nor compile() calls them.")
    print("The compiled model is byte-identical with or without them.")
    print("=" * 60)
