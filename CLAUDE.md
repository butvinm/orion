# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Orion

An opinionated fork of [baahl-nyu/orion](https://github.com/baahl-nyu/orion), a research-grade FHE framework for deep learning inference. The fork refactors Orion for practical usage: instance-based API (no global state), explicit context passing, and full access to underlying Lattigo primitives.

Orion takes PyTorch neural networks, analyzes them, and produces artifacts that enable encrypted inference using the CKKS scheme. The core pipeline is: **fit** (collect value range statistics) → **compile** (assign FHE levels, place bootstraps, pack data) → **encrypt & infer** (run on ciphertexts).

See `ARCH.md` for the full target architecture, compiled model format specification, evaluator API design, and repo structure plan.

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
pytest tests/test_v2_api.py::TestCompiler::test_compiler_produces_compiled_model

# Run tests for a specific model
pytest tests/models/test_mlp.py
```

The build process compiles Go code in `orionclient/bridge/` into a platform-specific shared library (`.so`/`.dylib`/`.dll`) that Python loads via ctypes.

## Design Principles

### Don't Constrain Lattigo Usage

Orion provides **model compilation**, **plaintext encoding**, and **model evaluation**. Encryption and decryption are the user's domain — Orion must not hide or restrict access to the underlying Lattigo primitives. Users may need per-ciphertext control for threshold encryption, custom key management, hybrid schemes, or any protocol Lattigo supports. Convenience methods (e.g. `encrypt_tensor`) are fine as shortcuts, but the per-ciphertext `encode`/`encrypt`/`decrypt`/`decode` path must always remain accessible.

### Compiled model stores raw numerical data, not Lattigo artifacts

The compiled model is a portable mathematical description: raw float64 diagonal matrices, bias vectors, polynomial coefficients, and a computation graph with edges. No Lattigo `MarshalBinary` blobs. The evaluator CKKS-encodes the raw data into Lattigo format at load time. This keeps the format Lattigo-version-independent and portable to any CKKS implementation.

### No backward compatibility with legacy code

This is a full refactor. Every line of code should serve the target architecture only. No compatibility shims, no legacy fallbacks.

## Target Architecture

Four components, detailed in `ARCH.md`:

1. **orion-compiler** (Python + Lattigo via Go bridge) — traces PyTorch models, fits statistics, assigns levels, places bootstraps, packs diagonals. Outputs `.orion` files containing a JSON header (computation graph with edges, CKKS params, key manifest) + binary blobs (raw float64 packed diagonals).
2. **orion-evaluator** (pure Go) — `LoadModel()` parses `.orion` and CKKS-encodes diagonals (shared, immutable). `NewEvaluator(params, keys)` creates per-client evaluators. `eval.Forward(model, ct)` walks the graph.
3. **lattigo bindings** (Go + Python/JS wrappers) — not Orion-specific. Keygen, encrypt, decrypt, encode, decode.
4. **orion-client** (pure Python/JS) — tensor-to-CKKS-slot mapping (flatten, pad, split) on top of lattigo bindings.

## Current Architecture

The codebase is being refactored toward the target. The sections below describe **what currently exists in the code**.

### Two-class pipeline (Compiler + Client)

The API uses two purpose-built classes: `Compiler` and `Client`. The Python `Evaluator` has been deleted — Phase 2 provides a pure Go evaluator. No global state, no YAML configs, no HDF5 files. All artifacts serialize to bytes.

**End-to-end usage:**

```python
import orion
from orion.models import MLP

# 1. Compile (fit() requires Go backend; compile() is Go-free)
compiler = orion.Compiler(net, orion.CKKSParams(logn=14, logq=[...], logp=[...], logscale=40))
compiler.fit(dataloader)
compiled = compiler.compile()  # -> CompiledModel (v2 format with computation graph)
open("model.orion", "wb").write(compiled.to_bytes())

# 2. Client (has secret key)
compiled = orion.CompiledModel.from_bytes(open("model.orion", "rb").read())
client = orion.Client(compiled.params)
keys = client.generate_keys(compiled.manifest)
ct = client.encrypt_tensor(input_tensor, level=compiled.input_level)

# 3. Server — Go evaluator (Phase 2, not yet implemented)
# The .orion v2 file is the input contract for the Go evaluator
```

**Key files:**

- `orion/params.py` — `CKKSParams` (frozen dataclass), `CompilerConfig`, `CostProfile`
- `orion/compiler.py` — `Compiler` class: traces, fits, compiles. `compile()` has zero Go/Lattigo dependency.
- `orion/client.py` — `Client` class: key generation, encode/decode, encrypt/decrypt.
- `orion/ciphertext.py` — `Ciphertext` and `PlainText` wrappers over Go objects via `cgo.Handle`.
- `orion/compiled_model.py` — `CompiledModel` (v2 format), `Graph`, `GraphNode`, `GraphEdge`, `KeyManifest`, `EvalKeys` with binary serialization.
- `orion/core/galois.py` — Pure Python BSGS Galois element computation (replaces Lattigo calls during compile).

### Context passing (no global state)

- **Compile time:** `module.fit(context)` takes a namespace with `backend`, `params`, `encoder`, `lt_evaluator`, `poly_evaluator`, `margin`, `config`. The `compile()` step does NOT call `module.compile()` — it reads module attributes directly (set by `fit()` + `generate_diagonals()`). Galois elements computed in pure Python.
- **Inference time:** `Ciphertext` carries a `context` attribute with the evaluator FFI handle and param info. Output ciphertexts propagate context automatically.

### Custom NN modules (`orion/nn/`)

All layers extend `orion.nn.Module` (which extends `torch.nn.Module`) and operate in two modes toggled by `.he()`:

- **Cleartext mode**: standard PyTorch forward pass
- **FHE mode**: operates on `Ciphertext`/`PlainText` objects

Key modules: `Linear`, `Conv2d`, `AvgPool2d`, `Quad`, `Sigmoid`, `SiLU`, `GELU`, `ReLU`, `BatchNorm1d/2d`, `Add`, `Mult`, `Bootstrap`, `Flatten`.

### Core algorithms (`orion/core/`)

- **`network_dag.py`** — Converts PyTorch FX trace to NetworkX DAG; detects residual connections.
- **`level_dag.py`** — Assigns multiplicative levels per layer; handles residual paths.
- **`auto_bootstrap.py`** — Minimizes bootstrap insertions while keeping levels valid. Inserts explicit bootstrap nodes into the DAG (not forward hooks).
- **`packing.py`** — Converts Conv2d/Linear to matrix-vector products via Toeplitz matrices, extracts diagonals.
- **`tracer.py`** — Custom PyTorch FX tracer (`OrionTracer`); `StatsTracker` captures value ranges.
- **`fuser.py`** — Fuses consecutive linear+activation ops to reduce multiplicative depth.
- **`galois.py`** — Pure Python BSGS Galois element computation. Reimplements Lattigo's `lintrans.GaloisElements()` algorithm, validated against Lattigo.

### Backend layer

- **orionclient (Go):** `orionclient/` — Instance-based Go library with `Client` and `Evaluator`. Bridge layer at `orionclient/bridge/` exports C functions via `cgo.Handle`.
- **Python FFI:** `orion/backend/orionclient/ffi.py` — ctypes bindings. All Go objects wrapped in `GoHandle` (RAII). Error propagation via `errOut` pattern.
- **Compile-time helpers:** `orion/core/compiler_backend.py` — `CompilerBackend`, `PolynomialGenerator`, `TransformEncoder`.

### GoHandle — Go object lifecycle management

`GoHandle` (`orion/backend/orionclient/ffi.py`) is an RAII wrapper for `cgo.Handle` values (`uintptr_t`). Rules:

1. **GoHandle wraps every Go object.** Tagged with descriptive strings (`"Client"`, `"Ciphertext"`, etc.).
2. **Bridge functions borrow, never consume.** Only `DeleteHandle` (called by `GoHandle.close()`) frees the handle slot.
3. **Two-step close pattern.** Resource cleanup (Go Close method) then handle table cleanup (`handle.close()`). Both idempotent.
4. **Intermediates freed immediately.** Error paths use try/except to clean up partial handles.
5. **Canonical `__del__`.** Every handle-owning class uses `def __del__(self): try: self.close() except Exception: pass`.

### Current serialization formats

All artifacts use binary containers with magic headers, JSON metadata, and length-prefixed blobs:

- `CompiledModel`: magic `ORION\x00\x02\x00` — v2 format with computation graph (nodes + edges), raw float64 diagonal blobs, raw float64 bias blobs. Polynomial coefficients inline in node config. No Lattigo artifacts.
- `EvalKeys`: magic `ORKEY\x00\x01\x00` — RLK, Galois keys, bootstrap keys
- `Ciphertext`: shape header + Go Lattigo marshal bytes

### Models (`orion/models/`)

Pre-built architectures using Orion's custom layers: LoLA, LeNet, MLP, VGG, AlexNet, ResNet, YOLOv1.

## Conventions

- Top-level API re-exported from `orion/__init__.py`: `Compiler`, `Client`, `CompiledModel`, `CKKSParams`, etc. (Python `Evaluator` deleted — Go evaluator in Phase 2).
