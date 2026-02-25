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
- `orion/client.py` — `Client` class: key generation, encode/decode, encrypt/decrypt. Thin FFI wrapper over `orionclient`.
- `orion/ciphertext.py` — Unified `Ciphertext` class wrapping a Go `*orionclient.Ciphertext` via `cgo.Handle`. Replaces old `CipherText` and `CipherTensor`. Also contains `PlainText` wrapper.
- `orion/evaluator.py` — `Evaluator` class: loads compiled model + keys, runs FHE inference. No secret key.
- `orion/compiled_model.py` — `CompiledModel`, `KeyManifest`, `EvalKeys` with binary serialization (`to_bytes()`/`from_bytes()`).

### Context passing (no global state)

Modules receive context via explicit parameters, not class variables:

- **Compile time:** `module.compile(context)` and `module.fit(context)` take a namespace with `backend`, `params`, `encoder`, `lt_evaluator`, `poly_evaluator`, `margin`, `config`.
- **Inference time:** `Ciphertext` carries a `context` attribute with the evaluator FFI handle and param info. Each module reads `x.context.eval_handle`, `x.context.ckks_params`, etc. Output ciphertexts propagate context automatically.
- `Module.scheme` and `Module.margin` class variables are deleted. No `set_scheme()` or `set_margin()`.

### Evaluator type split

Compile-time and inference-time evaluators use separate implementations:

- `PolynomialGenerator` (compile-time, in `compiler_backend.py`) / `_EvalContext` methods (inference-time, in `evaluator.py`)
- `TransformEncoder` (compile-time, in `compiler_backend.py`) / `_EvalContext.evaluate_transforms` (inference-time, in `evaluator.py`)

At inference time, `_EvalContext` self-aliases as `encoder`, `poly_evaluator`, `lt_evaluator`, `params`, `evaluator`, and `bootstrapper` so nn modules can call e.g. `context.poly_evaluator.evaluate_polynomial(...)` uniformly.

### Custom NN modules (`orion/nn/`)

All layers extend `orion.nn.Module` (which extends `torch.nn.Module`) and operate in two modes toggled by `.he()`:

- **Cleartext mode**: standard PyTorch forward pass
- **FHE mode**: operates on `Ciphertext`/`PlainText` objects (unified types from `orion/ciphertext.py`)

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
- **Python FFI:** `orion/backend/orionclient/ffi.py` — ctypes bindings for the bridge shared library. All Go objects wrapped in `GoHandle` (RAII wrapper). Error propagation via `errOut` pattern.
- **Compile-time helpers:** `orion/core/compiler_backend.py` — `CompilerBackend` (adapter wrapping FFI for the compiler), `NewEncoder`, `PolynomialGenerator`, `TransformEncoder`, `NewParameters`.

### GoHandle — Go object lifecycle management

`GoHandle` (`orion/backend/orionclient/ffi.py`) is an RAII wrapper for `cgo.Handle` values (`uintptr_t`). Every FFI function returning a Go object returns `GoHandle`; no raw ints escape `ffi.py`.

**Five rules:**

1. **GoHandle wraps every Go object.** `GoHandle.__init__` accepts an optional `tag: str` (e.g. `"Client"`, `"Ciphertext"`, `"LinearTransform"`). All FFI wrapper functions pass a descriptive tag. `__repr__` includes the tag: `GoHandle(42 Client)` or `GoHandle(closed Client)`. `GoHandle.raw` returns the underlying int (raises `RuntimeError` if closed). `GoHandle.close()` calls `DeleteHandle` — idempotent, safe to call multiple times.
2. **Bridge functions borrow, never consume.** `ClientClose(h)` zeros the secret key but does NOT delete the cgo handle. `EvaluatorClose(h)` same. Only `DeleteHandle` (called by `GoHandle.close()`) frees the handle slot.
3. **Two-step close pattern.** `Client.close()` and `Evaluator.close()` call the Go Close method (resource cleanup) then `handle.close()` (handle table cleanup). Both idempotent.
4. **Evaluator owns reconstruction handles.** `Evaluator._tracked_handles: list[GoHandle]` collects all handles created during module reconstruction (LinearTransform, Polynomial, bias PlainText, bootstrap prescale). `Evaluator.close()` closes them all. Modules do NOT clean up Go handles.
5. **Intermediates freed immediately.** Multi-step FFI sequences (encrypt multiple CTs then combine, linear transform accumulation) close intermediate handles as soon as they're consumed. Error paths use try/except to clean up partial handles.
6. **Canonical `__del__` for handle-owning classes.** Every class wrapping a Go handle uses `def __del__(self): try: self.close() except Exception: pass`. All handle-owning classes (`GoHandle`, `Ciphertext`, `PlainText`, `Client`, `_MultiPlainText`, `Evaluator`, `Compiler`, `CompilerBackend`, `PlainTensor`) follow this pattern. `Ciphertext` and `PlainText` expose `close()` for explicit deterministic handle release.

### Models (`orion/models/`)

Pre-built architectures using Orion's custom layers: LoLA, LeNet, MLP, VGG, AlexNet, ResNet, YOLOv1.

### Serialization formats

All artifacts use binary containers with magic headers, JSON metadata, length-prefixed blobs, and CRC32 checksums:

- `CompiledModel`: magic `ORMDL\x00\x01\x00` — stores params, manifest, module metadata, LinearTransform blobs
- `EvalKeys`: magic `ORKEY\x00\x01\x00` — stores RLK, Galois keys, bootstrap keys
- `Ciphertext`: simple shape header (`[ndim, d0, d1, ...]` as little-endian int32s) followed by Go Lattigo marshal bytes. No magic header or CRC32.

### Go backend — instance-based

The `orionclient` library is fully instance-based. Multiple `Client` and `Evaluator` instances with different parameters can coexist in the same process. No global state.

## Conventions

- Top-level API: `orion.Compiler`, `orion.Client`, `orion.Evaluator`, `orion.CompiledModel`, `orion.CKKSParams`, etc. (re-exported from `orion/__init__.py`).
- Go FFI uses `cgo.Handle` for opaque pointer passing, wrapped in `GoHandle` on the Python side. Bridge functions return `(result, errOut)` pairs. Python wrapper checks `errOut` and raises `RuntimeError`. No raw `uintptr_t` values outside `ffi.py`.
- Parameters are defined as frozen dataclasses (`CKKSParams`, `CompilerConfig`) — no YAML configs.
- Binary serialization (`to_bytes()`/`from_bytes()`) for all cross-process artifacts — no HDF5.
