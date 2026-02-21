# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Orion

Orion is a Fully Homomorphic Encryption (FHE) framework for deep learning inference. It takes PyTorch neural networks, analyzes them, and executes inference on encrypted data using the CKKS scheme. The core pipeline is: **fit** (collect value range statistics) → **compile** (assign FHE levels, place bootstraps, pack data) → **encrypt & infer** (run on ciphertexts).

## Build & Development

**System prerequisites:** Go 1.22+, C compiler (CGO), libgmp-dev, libssl-dev, Python 3.9–3.12.

```bash
# Install in editable mode (builds Lattigo shared library via tools/build_lattigo.py)
pip install -e .

# Build distributable wheels
pip install poetry
poetry build

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_v2_api.py::TestEvaluator::test_full_roundtrip

# Run tests for a specific model
pytest tests/models/test_mlp.py
```

The build process compiles Go code in `orionclient/bridge/` into a platform-specific shared library (`.so`/`.dylib`/`.dll`) that Python loads via ctypes.

## Architecture

### v2 API — Three-class pipeline

The API uses three purpose-built classes: `Compiler`, `Client`, and `Evaluator`. No global state, no YAML configs, no HDF5 files. All artifacts serialize to bytes.

**End-to-end usage:**

```python
import orion
from orion.models import MLP

# 1. Compile (requires Go backend, no keys)
compiler = orion.Compiler(net, orion.CKKSParams(logn=14, logq=[...], logp=[...], logscale=40))
compiler.fit(dataloader)
compiled = compiler.compile()  # -> CompiledModel
open("model.bin", "wb").write(compiled.to_bytes())

# 2. Client (has secret key)
compiled = orion.CompiledModel.from_bytes(open("model.bin", "rb").read())
client = orion.Client(compiled.params)
keys = client.generate_keys(compiled.manifest)
pt = client.encode(input_tensor, level=compiled.input_level)
ct = client.encrypt(pt)

# 3. Server (has model class definition, no trained weights needed)
compiled = orion.CompiledModel.from_bytes(open("model.bin", "rb").read())
net_skeleton = MLP()  # fresh instance — weights baked into CompiledModel blobs
evaluator = orion.Evaluator(net_skeleton, compiled, keys)
ct_result = evaluator.run(ct)
result = client.decode(client.decrypt(ct_result))
```

**Key files:**

- `orion/params.py` — `CKKSParams` (frozen dataclass for CKKS parameters), `CompilerConfig` (compilation settings)
- `orion/compiler.py` — `Compiler` class: traces, fits, and compiles networks. No keys needed.
- `orion/client.py` — `Client` class: key generation, encode/decode, encrypt/decrypt. `PlainText` and `CipherText` wrappers.
- `orion/evaluator.py` — `Evaluator` class: loads compiled model + keys, runs FHE inference. No secret key.
- `orion/compiled_model.py` — `CompiledModel`, `KeyManifest`, `EvalKeys` with binary serialization (`to_bytes()`/`from_bytes()`).

### Context passing (no global state)

Modules receive context via explicit parameters, not class variables:

- **Compile time:** `module.compile(context)` and `module.fit(context)` take a namespace with `backend`, `params`, `encoder`, `lt_evaluator`, `poly_evaluator`, `margin`, `config`.
- **Inference time:** `CipherTensor` carries a `context` attribute. Each module reads `x.context.evaluator`, `x.context.encoder`, etc. Output tensors propagate context automatically.
- `Module.scheme` and `Module.margin` class variables are deleted. No `set_scheme()` or `set_margin()`.

### Evaluator type split

Compile-time and inference-time evaluators are separate types:

- `PolynomialGenerator` (compile-time) / `PolynomialEvaluator` (inference-time, inherits Generator)
- `TransformEncoder` (compile-time) / `TransformEvaluator` (inference-time)

No `keyless` boolean — the type system enforces which operations are available.

### Custom NN modules (`orion/nn/`)

All layers extend `orion.nn.Module` (which extends `torch.nn.Module`) and operate in two modes toggled by `.he()`:

- **Cleartext mode**: standard PyTorch forward pass
- **FHE mode**: operates on `CipherTensor`/`PlainTensor` objects

Key modules: `Linear`, `Conv2d`, `AvgPool2d` (linear transforms with diagonal packing), `Quad`, `Sigmoid`, `SiLU`, `GELU`, `ReLU` (polynomial activations via Chebyshev), `BatchNorm1d/2d`, `Add`, `Mult`, `Bootstrap`, `Flatten`.

Each module carries a `level` (multiplicative depth) and `depth` (consumed depth), assigned during compile.

### Core algorithms (`orion/core/`)

- **`network_dag.py`** — Converts PyTorch FX trace to NetworkX DAG; detects residual connections (fork/join pairs).
- **`level_dag.py`** — Assigns multiplicative levels per layer; handles residual paths via aggregated level DAGs.
- **`auto_bootstrap.py`** — Minimizes bootstrap insertions while keeping levels valid across all paths including residuals.
- **`packing.py`** — Converts Conv2d/Linear to matrix-vector products via Toeplitz matrices, extracts diagonals for FHE-efficient sparse representation. Supports "hybrid" and "square" embedding methods.
- **`tracer.py`** — Custom PyTorch FX tracer (`OrionTracer`) with deeper recursion; `StatsTracker` wraps modules to capture value ranges.
- **`fuser.py`** — Fuses consecutive linear+activation ops to reduce multiplicative depth.

### Backend layer

- **orionclient (Go):** `orionclient/` — Instance-based Go library with `Client` (keygen, encrypt, decrypt) and `Evaluator` (FHE ops). No global state; multiple instances coexist. Bridge layer at `orionclient/bridge/` exports C functions via `cgo.Handle`.
- **Python FFI:** `orion/backend/orionclient/ffi.py` — ctypes bindings for the bridge shared library. All Go objects are opaque handles (`uintptr_t`). Error propagation via `errOut` pattern.
- **Compile-time helpers:** `orion/core/compiler_backend.py` — `CompilerBackend` (adapter wrapping FFI for the compiler), `NewEncoder`, `PolynomialGenerator`, `TransformEncoder`, `NewParameters`.

### Models (`orion/models/`)

Pre-built architectures using Orion's custom layers: LoLA, LeNet, MLP, VGG, AlexNet, ResNet, YOLOv1.

### Serialization formats

All artifacts use binary containers with magic headers, JSON metadata, length-prefixed blobs, and CRC32 checksums:

- `CompiledModel`: magic `ORMDL\x00\x01\x00` — stores params, manifest, module metadata, LinearTransform blobs
- `EvalKeys`: magic `ORKEY\x00\x01\x00` — stores RLK, Galois keys, bootstrap keys
- `CipherText`: custom format with per-ciphertext length-prefixed Lattigo binary

### Go backend — instance-based

The `orionclient` library is fully instance-based. Multiple `Client` and `Evaluator` instances with different parameters can coexist in the same process. No global state.

## Conventions

- Top-level API: `orion.Compiler`, `orion.Client`, `orion.Evaluator`, `orion.CompiledModel`, `orion.CKKSParams`, etc. (re-exported from `orion/__init__.py`).
- Go FFI uses `cgo.Handle` for opaque pointer passing. Bridge functions return `(result, errOut)` pairs. Python wrapper checks `errOut` and raises `RuntimeError`.
- Parameters are defined as frozen dataclasses (`CKKSParams`, `CompilerConfig`) — no YAML configs.
- Binary serialization (`to_bytes()`/`from_bytes()`) for all cross-process artifacts — no HDF5.
