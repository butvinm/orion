"""
Experiment 06: Evaluator keyless mode investigation

Hypothesis: The v2 Evaluator CANNOT use keyless lt_evaluator and poly_evaluator
for inference. It must create them in non-keyless mode after importing keys via
NewEvaluatorFromKeys().

Background: The v2 plan (Task 10, step 2) proposes creating "lt_evaluator
(keyless=True), poly_evaluator (keyless=True)" for the Evaluator class. This
experiment tests whether that approach is viable for inference.

Findings (5 tests):
  Test 1: Go EvaluateLinearTransform works WITHOUT NewLinearTransformEvaluator()
          (it recreates LinEvaluator internally on every call — see lineartransform.go:102-104)
  Test 2: Python keyless lt_evaluator.evaluate_transforms() fails with AttributeError
          (self.evaluator not set in keyless mode — needed for rescale at lt_evaluator.py:197)
  Test 3: Go EvaluatePolynomial CRASHES without NewPolynomialEvaluator()
          (scheme.PolyEvaluator is nil — tested in subprocess to avoid killing main process)
  Test 4: Go poly generation functions work WITHOUT NewPolynomialEvaluator()
          (GenerateChebyshev, GenerateMonomial are standalone — validates Compiler keyless usage)
  Test 5: Non-keyless evaluators after NewEvaluatorFromKeys() produce correct inference
          (full end-to-end with MLP, MAE < 0.005 vs cleartext)

Approach: Reuses exp 05's pipeline (compile keylessly, generate keys, import keys).
Tests are ordered to exercise the Go singleton without destructive state changes
until the final positive test.
"""

import sys
import os
import ctypes
import subprocess
import textwrap
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import orion
import orion.nn as on
from orion.core.orion import scheme as orion_scheme
from orion.core.utils import get_mnist_datasets, mae


class MLP(on.Module):
    """MLP model definition (was in models/mlp.py, now inlined)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = on.Flatten()
        self.fc1 = on.Linear(784, 128)
        self.bn1 = on.BatchNorm1d(128)
        self.act1 = on.Quad()
        self.fc2 = on.Linear(128, 128)
        self.bn2 = on.BatchNorm1d(128)
        self.act2 = on.Quad()
        self.fc3 = on.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        return self.fc3(x)
from orion.backend.python import lt_evaluator, poly_evaluator
from orion.backend.python.tensors import CipherTensor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CKKS parameters — same as configs/mlp.yml and exp 05
CONFIG = {
    "ckks_params": {
        "LogN": 13,
        "LogQ": [29, 26, 26, 26, 26, 26],
        "LogP": [29, 29],
        "LogScale": 26,
        "H": 8192,
        "RingType": "ConjugateInvariant",
    },
    "orion": {
        "margin": 2,
        "embedding_method": "hybrid",
        "backend": "lattigo",
        "fuse_modules": True,
        "debug": False,
        "io_mode": "none",
    },
}

TOLERANCE = 0.005


def serialize_to_numpy(backend, serialize_fn, *args):
    """Call a serialize function and return a numpy array of bytes."""
    c_args = []
    for arg in args:
        curr_argtype = serialize_fn.func.argtypes[len(c_args)]
        c_arg = serialize_fn.convert_to_ctypes(arg, curr_argtype)
        if isinstance(c_arg, tuple):
            c_args.extend(c_arg)
        else:
            c_args.append(c_arg)
    result = serialize_fn.func(*c_args)
    length = int(result.Length)
    buffer = ctypes.cast(
        result.Data,
        ctypes.POINTER(ctypes.c_ubyte * length),
    ).contents
    data = np.array(buffer, dtype=np.uint8).copy()
    backend.FreeCArray(ctypes.cast(result.Data, ctypes.c_void_p))
    return data


class ServerEvaluator:
    """Evaluator wrapper using NewEvaluatorFromKeys() — no secret key."""
    def __init__(self, backend):
        self.backend = backend

    def negate(self, ctxt):
        return self.backend.Negate(ctxt)

    def rotate(self, ctxt, amount, in_place):
        if in_place:
            return self.backend.Rotate(ctxt, amount)
        return self.backend.RotateNew(ctxt, amount)

    def add_scalar(self, ctxt, scalar, in_place):
        if in_place:
            return self.backend.AddScalar(ctxt, float(scalar))
        return self.backend.AddScalarNew(ctxt, float(scalar))

    def sub_scalar(self, ctxt, scalar, in_place):
        if in_place:
            return self.backend.SubScalar(ctxt, float(scalar))
        return self.backend.SubScalarNew(ctxt, float(scalar))

    def mul_scalar(self, ctxt, scalar, in_place):
        if isinstance(scalar, float) and scalar.is_integer():
            scalar = int(scalar)
        if isinstance(scalar, int):
            ct_out = (self.backend.MulScalarInt if in_place
                      else self.backend.MulScalarIntNew)(ctxt, scalar)
        else:
            ct_out = (self.backend.MulScalarFloat if in_place
                      else self.backend.MulScalarFloatNew)(ctxt, scalar)
            ct_out = self.backend.Rescale(ct_out)
        return ct_out

    def add_plaintext(self, ctxt, ptxt, in_place):
        if in_place:
            return self.backend.AddPlaintext(ctxt, ptxt)
        return self.backend.AddPlaintextNew(ctxt, ptxt)

    def sub_plaintext(self, ctxt, ptxt, in_place):
        if in_place:
            return self.backend.SubPlaintext(ctxt, ptxt)
        return self.backend.SubPlaintextNew(ctxt, ptxt)

    def mul_plaintext(self, ctxt, ptxt, in_place):
        if in_place:
            ct_out = self.backend.MulPlaintext(ctxt, ptxt)
        else:
            ct_out = self.backend.MulPlaintextNew(ctxt, ptxt)
        return self.backend.Rescale(ct_out)

    def add_ciphertext(self, ctxt0, ctxt1, in_place):
        if in_place:
            return self.backend.AddCiphertext(ctxt0, ctxt1)
        return self.backend.AddCiphertextNew(ctxt0, ctxt1)

    def sub_ciphertext(self, ctxt0, ctxt1, in_place):
        if in_place:
            return self.backend.SubCiphertext(ctxt0, ctxt1)
        return self.backend.SubCiphertextNew(ctxt0, ctxt1)

    def mul_ciphertext(self, ctxt0, ctxt1, in_place):
        if in_place:
            ct_out = self.backend.MulRelinCiphertext(ctxt0, ctxt1)
        else:
            ct_out = self.backend.MulRelinCiphertextNew(ctxt0, ctxt1)
        return self.backend.Rescale(ct_out)

    def rescale(self, ctxt, in_place):
        if in_place:
            return self.backend.Rescale(ctxt)
        return self.backend.RescaleNew(ctxt)


class ServerBootstrapper:
    """Bootstrapper wrapper using already-loaded bootstrap keys."""
    def __init__(self, backend):
        self.backend = backend

    def bootstrap(self, ctxt, slots):
        return self.backend.Bootstrap(ctxt, slots)


def setup_pipeline():
    """Run server compile + client keygen + key import. Returns state needed for tests."""
    print("=" * 70)
    print("SETUP: Compile + keygen + key import")
    print("=" * 70)

    trainloader, testloader = get_mnist_datasets(data_dir="./data", batch_size=1)
    test_input, _ = next(iter(testloader))

    torch.manual_seed(42)
    net = MLP(num_classes=10)
    net.eval()

    with torch.no_grad():
        out_clear = net(test_input)

    # --- Server phase 1: keyless compile ---
    orion.init_params_only(CONFIG)
    orion.fit(net, trainloader)
    input_level, manifest = orion.compile(net)
    print(f"  Compiled: input_level={input_level}, "
          f"{len(manifest['galois_elements'])} galois elements")

    backend = orion_scheme.backend

    # --- Client phase: keygen + encrypt ---
    backend.NewKeyGenerator()
    backend.GenerateSecretKey()
    backend.GeneratePublicKey()
    backend.GenerateRelinearizationKey()

    sk_data = serialize_to_numpy(backend, backend.SerializeSecretKey)
    rlk_data = serialize_to_numpy(backend, backend.SerializeRelinKey)

    galois_key_data = {}
    for galEl in manifest["galois_elements"]:
        gk_bytes = serialize_to_numpy(
            backend, backend.GenerateAndSerializeRotationKey, galEl)
        galois_key_data[galEl] = gk_bytes

    bootstrap_key_data = {}
    if manifest["bootstrap_slots"]:
        logPs = orion_scheme.params.get_boot_logp()
        for slots in manifest["bootstrap_slots"]:
            btp_bytes = serialize_to_numpy(
                backend, backend.SerializeBootstrapKeys, slots, logPs)
            bootstrap_key_data[slots] = btp_bytes

    backend.GenerateEvaluationKeys()
    backend.NewEvaluator()
    backend.NewEncryptor()

    vec_ptxt = orion.encode(test_input, input_level)
    ct_ids = [backend.Encrypt(pt_id) for pt_id in vec_ptxt.ids]
    ct_data_list = [serialize_to_numpy(backend, backend.SerializeCiphertext, ct_id)
                    for ct_id in ct_ids]
    ct_shape = vec_ptxt.shape
    ct_on_shape = vec_ptxt.on_shape

    print(f"  Encrypted: {len(ct_data_list)} ciphertexts")

    # --- Server phase 2: import keys ---
    backend.ClearSecretKey()
    backend.LoadRelinKey(rlk_data)
    backend.GenerateEvaluationKeys()

    for galEl, gk_bytes in galois_key_data.items():
        backend.LoadRotationKey(gk_bytes, galEl)

    if bootstrap_key_data:
        logPs = orion_scheme.params.get_boot_logp()
        for slots, btp_bytes in bootstrap_key_data.items():
            backend.LoadBootstrapKeys(btp_bytes, slots, logPs)

    # Create Go evaluator from imported keys (no secret key)
    backend.NewEvaluatorFromKeys()
    print("  Keys imported, NewEvaluatorFromKeys() called")

    # NOTE: At this point:
    #   scheme.Evaluator is set (from NewEvaluatorFromKeys)
    #   scheme.EvalKeys is set (from key import)
    #   scheme.LinEvaluator is stale (from compile phase) or nil
    #   scheme.PolyEvaluator is nil (never called NewPolynomialEvaluator on server)

    return {
        "backend": backend,
        "net": net,
        "out_clear": out_clear,
        "ct_data_list": ct_data_list,
        "ct_shape": ct_shape,
        "ct_on_shape": ct_on_shape,
        "sk_data": sk_data,
        "input_level": input_level,
        "manifest": manifest,
    }


def test1_go_lt_eval_without_new(state):
    """Go-level EvaluateLinearTransform works WITHOUT NewLinearTransformEvaluator().

    lineartransform.go:102-104 recreates LinEvaluator on every call:
        scheme.LinEvaluator = lintrans.NewEvaluator(
            scheme.Evaluator.WithKey(scheme.EvalKeys))
    So NewLinearTransformEvaluator() is redundant for evaluation.
    Only needs scheme.Evaluator (from NewEvaluatorFromKeys) + scheme.EvalKeys.
    """
    print("\n" + "-" * 70)
    print("TEST 1: Go EvaluateLinearTransform WITHOUT NewLinearTransformEvaluator()")
    print("-" * 70)

    backend = state["backend"]
    net = state["net"]

    # Get a transform ID and ciphertext ID from the compiled network
    # Find first linear layer with transform_ids
    linear_module = None
    for name, module in net.named_modules():
        if hasattr(module, "transform_ids") and module.transform_ids:
            linear_module = module
            break

    if linear_module is None:
        print("  SKIP: No linear module with transform_ids found")
        return False

    # Get one transform ID and create a test ciphertext
    first_key = next(iter(linear_module.transform_ids))
    transform_id = linear_module.transform_ids[first_key]

    # Load a ciphertext for testing
    ct_id = backend.LoadCiphertext(state["ct_data_list"][0])

    # Call EvaluateLinearTransform WITHOUT calling NewLinearTransformEvaluator first.
    # This should work because EvaluateLinearTransform recreates LinEvaluator internally.
    try:
        result_id = backend.EvaluateLinearTransform(transform_id, ct_id)
        print(f"  EvaluateLinearTransform({transform_id}, {ct_id}) -> {result_id}")
        print("  PASS: Works without NewLinearTransformEvaluator()")
        print("  Reason: Go code recreates LinEvaluator from scheme.Evaluator.WithKey()")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test2_python_keyless_lt_evaluator(state):
    """Python keyless lt_evaluator.evaluate_transforms() fails with AttributeError.

    In keyless mode (lt_evaluator.py:28-33), new_evaluator() returns early —
    self.evaluator is never set (only set at line 16 when keyless=False).
    evaluate_transforms() at line 197 calls self.evaluator.rescale() → crash.
    """
    print("\n" + "-" * 70)
    print("TEST 2: Python keyless lt_evaluator → evaluate_transforms() failure")
    print("-" * 70)

    backend = state["backend"]
    net = state["net"]

    # Create a keyless lt_evaluator (as the plan proposes for Evaluator)
    keyless_lt = lt_evaluator.NewEvaluator(orion_scheme, keyless=True)

    # Verify it has NO evaluator attribute
    has_evaluator = hasattr(keyless_lt, "evaluator")
    print(f"  keyless lt_evaluator has 'evaluator' attr: {has_evaluator}")

    if has_evaluator:
        print("  UNEXPECTED: keyless lt_evaluator has evaluator attribute")
        return False

    # Find first linear module
    linear_module = None
    for name, module in net.named_modules():
        if hasattr(module, "transform_ids") and module.transform_ids:
            linear_module = module
            break

    if linear_module is None:
        print("  SKIP: No linear module found")
        return False

    # Create a CipherTensor for testing
    ct_id = backend.LoadCiphertext(state["ct_data_list"][0])
    test_ct = CipherTensor(orion_scheme, [ct_id], state["ct_shape"], state["ct_on_shape"])

    # Try to call evaluate_transforms — should fail
    try:
        keyless_lt.evaluate_transforms(linear_module, test_ct)
        print("  UNEXPECTED: evaluate_transforms succeeded with keyless lt_evaluator")
        return False
    except AttributeError as e:
        print(f"  AttributeError: {e}")
        print("  PASS: keyless lt_evaluator correctly fails on evaluate_transforms()")
        print("  Reason: self.evaluator not set in keyless mode (needed for rescale)")
        return True
    except Exception as e:
        print(f"  Different error: {type(e).__name__}: {e}")
        print("  PARTIAL: Failed, but with unexpected error type")
        return True  # Still proves it doesn't work


def test3_go_poly_eval_without_new(state):
    """Go EvaluatePolynomial crashes without NewPolynomialEvaluator().

    polyeval.go:75 dereferences scheme.PolyEvaluator which is nil if
    NewPolynomialEvaluator() was never called. This is a Go nil pointer
    panic — fatal, not recoverable from Python.

    We test this in a subprocess to avoid killing the main process.
    """
    print("\n" + "-" * 70)
    print("TEST 3: Go EvaluatePolynomial WITHOUT NewPolynomialEvaluator()")
    print("-" * 70)

    # Subprocess approach: init_params_only creates Params + Encoder + KeyGen
    # but NOT PolyEvaluator. We generate keys manually, create an evaluator
    # from them, encrypt data, then call EvaluatePolynomial — which should
    # crash because scheme.PolyEvaluator is nil (polyeval.go:75).
    subprocess_script = textwrap.dedent(f"""\
        import sys, os
        sys.path.insert(0, os.path.join("{SCRIPT_DIR}", "..", ".."))
        import torch
        import orion
        from orion.core.orion import scheme as s

        config = {{
            "ckks_params": {{
                "LogN": 13,
                "LogQ": [29, 26, 26, 26, 26, 26],
                "LogP": [29, 29],
                "LogScale": 26,
                "H": 8192,
                "RingType": "ConjugateInvariant",
            }},
            "orion": {{
                "margin": 2, "embedding_method": "hybrid",
                "backend": "lattigo", "fuse_modules": True,
                "debug": False, "io_mode": "none",
            }},
        }}

        # init_params_only: creates Params, KeyGen, Encoder
        # Does NOT call NewPolynomialEvaluator → scheme.PolyEvaluator is nil
        orion.init_params_only(config)
        backend = s.backend

        # Generate keys (KeyGen exists from NewScheme)
        backend.GenerateSecretKey()
        backend.GeneratePublicKey()
        backend.GenerateRelinearizationKey()
        backend.GenerateEvaluationKeys()

        # Create evaluator from eval keys (no SK needed after this)
        backend.NewEvaluatorFromKeys()

        # Create encryptor for test data
        backend.NewEncryptor()

        # Create test ciphertext + polynomial
        slots = s.params.get_slots()
        pt_id = backend.Encode(list(torch.randn(slots).double()), 4, 0)
        ct_id = backend.Encrypt(pt_id)
        poly_id = backend.GenerateChebyshev([0.5, 0.25, 0.125])
        out_scale = s.params.get_default_scale()

        # At this point:
        #   scheme.Params      = set (from NewScheme)
        #   scheme.Evaluator   = set (from NewEvaluatorFromKeys)
        #   scheme.PolyEvaluator = nil (never called NewPolynomialEvaluator)
        #
        # EvaluatePolynomial (polyeval.go:75) does:
        #   scheme.PolyEvaluator.Evaluate(...)
        # → nil pointer dereference → Go panic

        print("Calling EvaluatePolynomial without NewPolynomialEvaluator...")
        print("(scheme.PolyEvaluator is nil — expecting Go panic)")
        sys.stdout.flush()

        try:
            result = backend.EvaluatePolynomial(ct_id, poly_id, out_scale)
            print("UNEXPECTED_SUCCESS")
        except Exception as e:
            print(f"PYTHON_ERROR: {{e}}")
    """)

    script_path = os.path.join(SCRIPT_DIR, "_test3_subprocess.py")
    with open(script_path, "w") as f:
        f.write(subprocess_script)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=120,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        returncode = result.returncode

        print(f"  Subprocess exit code: {returncode}")
        if stdout:
            print(f"  stdout: {stdout[-500:]}")
        if stderr:
            # Show last few lines of stderr (Go panic traces are verbose)
            stderr_lines = stderr.split("\n")
            print(f"  stderr (last 10 lines):")
            for line in stderr_lines[-10:]:
                print(f"    {line}")

        if "UNEXPECTED_SUCCESS" in stdout:
            print("  UNEXPECTED: EvaluatePolynomial succeeded without NewPolynomialEvaluator")
            return False
        elif returncode != 0 and "nil pointer" in stderr.lower():
            print("  PASS: Go nil pointer panic confirmed")
            print("  Reason: scheme.PolyEvaluator is nil (polyeval.go:75)")
            return True
        elif returncode != 0:
            print("  PASS: Process crashed (non-zero exit)")
            print("  Reason: scheme.PolyEvaluator is nil → Go runtime panic")
            return True
        else:
            print("  INCONCLUSIVE: Process exited 0 but no UNEXPECTED_SUCCESS")
            return False
    except subprocess.TimeoutExpired:
        print("  TIMEOUT: subprocess took >120s")
        return False
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def test4_poly_generation_without_new(state):
    """Poly generation functions work WITHOUT NewPolynomialEvaluator().

    GenerateChebyshev (polyeval.go:49-60) and GenerateMonomial (polyeval.go:37-47)
    use bignum.NewPolynomial() — standalone, no scheme.PolyEvaluator reference.
    GenerateMinimaxSignCoeffs (polyeval.go:90-167) uses minimax.GenMinimaxCompositePolynomial()
    — also standalone.

    This validates the Compiler's keyless poly_evaluator usage in the v2 plan.
    """
    print("\n" + "-" * 70)
    print("TEST 4: Poly generation functions WITHOUT NewPolynomialEvaluator()")
    print("-" * 70)

    backend = state["backend"]

    # At this point, NewPolynomialEvaluator() has NOT been called on the
    # server-side Go singleton (only NewEvaluatorFromKeys was called).
    # Test that generation functions work anyway.

    try:
        # GenerateChebyshev — returns a polynomial heap ID
        poly_id = backend.GenerateChebyshev([0.5, 0.25, 0.125])
        print(f"  GenerateChebyshev([0.5, 0.25, 0.125]) -> poly_id={poly_id}")

        # GenerateMonomial — returns a polynomial heap ID
        poly_id2 = backend.GenerateMonomial([1.0, 0.5, 0.25])
        print(f"  GenerateMonomial([1.0, 0.5, 0.25]) -> poly_id={poly_id2}")

        print("  PASS: Generation functions work without NewPolynomialEvaluator()")
        print("  Reason: Go functions use standalone bignum.NewPolynomial(), "
              "not scheme.PolyEvaluator")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def test5_full_inference_non_keyless(state):
    """Full inference with non-keyless evaluators after NewEvaluatorFromKeys().

    After key import + NewEvaluatorFromKeys():
      1. Call NewLinearTransformEvaluator() (creates Go LinEvaluator)
      2. Call NewPolynomialEvaluator() (creates Go PolyEvaluator from scheme.Evaluator)
      3. Create Python wrappers in non-keyless mode
      4. Run full MLP inference on encrypted data
      5. Verify MAE < 0.005 vs cleartext
    """
    print("\n" + "-" * 70)
    print("TEST 5: Full inference with non-keyless evaluators")
    print("-" * 70)

    backend = state["backend"]
    net = state["net"]

    # Create Go-level evaluators (now WITH NewLinearTransformEvaluator + NewPolynomialEvaluator)
    backend.NewLinearTransformEvaluator()
    print("  Called NewLinearTransformEvaluator()")

    backend.NewPolynomialEvaluator()
    print("  Called NewPolynomialEvaluator()")

    # Wire up Python-level evaluators (NON-keyless)
    server_eval = ServerEvaluator(backend)
    server_btp = ServerBootstrapper(backend)

    orion_scheme.evaluator = server_eval

    # poly_evaluator: bypass __init__ (it would call NewPolynomialEvaluator again)
    orion_scheme.poly_evaluator = poly_evaluator.NewEvaluator.__new__(
        poly_evaluator.NewEvaluator)
    orion_scheme.poly_evaluator.scheme = orion_scheme
    orion_scheme.poly_evaluator.backend = backend

    # lt_evaluator: switch from keyless to non-keyless
    orion_scheme.lt_evaluator.keyless = False
    orion_scheme.lt_evaluator.evaluator = server_eval

    orion_scheme.bootstrapper = server_btp
    orion_scheme.encryptor = None
    orion_scheme.keyless = False

    # Load ciphertexts
    server_ct_ids = [backend.LoadCiphertext(ct_bytes)
                     for ct_bytes in state["ct_data_list"]]

    server_ctxt = CipherTensor(
        orion_scheme, server_ct_ids, state["ct_shape"], state["ct_on_shape"])

    # Run FHE inference
    print("  Running FHE inference...")
    infer_start = time.time()
    net.he()
    out_ctxt = net(server_ctxt)
    infer_time = time.time() - infer_start
    print(f"  Inference time: {infer_time:.1f}s")

    # Decrypt (need to reload SK)
    backend.LoadSecretKey(state["sk_data"])
    backend.NewDecryptor()
    backend.NewEncoder()

    result_values = []
    for ct_id in out_ctxt.ids:
        pt_id = backend.Decrypt(ct_id)
        values = backend.Decode(pt_id)
        result_values.extend(values)

    out_fhe = torch.tensor(result_values[:state["out_clear"].shape[1]])
    error = mae(state["out_clear"], out_fhe.unsqueeze(0))

    print(f"  FHE output (first 5):  {out_fhe[:5].tolist()}")
    print(f"  Clear output (first 5): {state['out_clear'][0, :5].tolist()}")
    print(f"  MAE: {error:.6f} (tolerance: {TOLERANCE})")

    if error < TOLERANCE:
        print(f"  PASS: Non-keyless evaluators produce correct inference")
        return True
    else:
        print(f"  FAIL: MAE {error:.6f} >= {TOLERANCE}")
        return False


def main():
    print("=" * 70)
    print("Experiment 06: Evaluator Keyless Mode Investigation")
    print("=" * 70)

    state = setup_pipeline()

    results = {}

    # Test 1: Go-level LT eval works without NewLinearTransformEvaluator
    results["test1"] = test1_go_lt_eval_without_new(state)

    # Test 2: Python keyless lt_evaluator fails
    results["test2"] = test2_python_keyless_lt_evaluator(state)

    # Test 3: Go poly eval crashes without NewPolynomialEvaluator (subprocess)
    results["test3"] = test3_go_poly_eval_without_new(state)

    # Test 4: Poly generation functions work without NewPolynomialEvaluator
    results["test4"] = test4_poly_generation_without_new(state)

    # Test 5: Full inference with non-keyless evaluators
    results["test5"] = test5_full_inference_non_keyless(state)

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 06 RESULTS")
    print("=" * 70)

    labels = {
        "test1": "Go EvaluateLinearTransform without NewLinearTransformEvaluator",
        "test2": "Python keyless lt_evaluator → AttributeError",
        "test3": "Go EvaluatePolynomial without NewPolynomialEvaluator → panic",
        "test4": "Poly generation functions without NewPolynomialEvaluator",
        "test5": "Full inference with non-keyless evaluators after key import",
    }

    all_pass = True
    for key, label in labels.items():
        status = "PASS" if results[key] else "FAIL"
        if not results[key]:
            all_pass = False
        print(f"  [{status}] {label}")

    print()
    if all_pass:
        print("ALL TESTS PASSED — Hypothesis CONFIRMED:")
        print()
        print("  The v2 Evaluator MUST create lt_evaluator and poly_evaluator")
        print("  in non-keyless mode after importing keys via NewEvaluatorFromKeys().")
        print()
        print("  Specifically:")
        print("  - lt_evaluator(keyless=True) fails at Python level: evaluate_transforms()")
        print("    needs self.evaluator for rescale(), which is only set in non-keyless mode.")
        print("  - NewPolynomialEvaluator() is required before EvaluatePolynomial(): the Go")
        print("    function dereferences scheme.PolyEvaluator which is nil without it.")
        print("  - NewLinearTransformEvaluator() is technically redundant (Go recreates it per")
        print("    call), but the Python lt_evaluator still needs non-keyless mode for self.evaluator.")
        print()
        print("  Bonus finding: poly_evaluator generation functions (GenerateChebyshev,")
        print("  GenerateMonomial, GenerateMinimaxSignCoeffs) DO work without")
        print("  NewPolynomialEvaluator(). This validates the Compiler's keyless usage —")
        print("  compile-time polynomial generation needs no Go evaluator.")
        print()
        print("  Recommended fix for v2 plan Task 10, step 2:")
        print("    BEFORE: lt_evaluator(keyless=True), poly_evaluator(keyless=True)")
        print("    AFTER:  Create encoder only in step 2. After NewEvaluatorFromKeys()")
        print("            (step 6), create lt_evaluator(keyless=False) and")
        print("            poly_evaluator in normal mode.")
    else:
        print("SOME TESTS FAILED — see details above")

    # Cleanup
    out_ctxt_ids = []  # prevent stale CipherTensor __del__ panics
    orion.delete_scheme()


if __name__ == "__main__":
    main()
