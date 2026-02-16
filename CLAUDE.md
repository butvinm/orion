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
pytest tests/test_imports.py::test_linear_transforms

# Run tests for a specific model
pytest tests/models/test_mlp.py
```

The build process compiles Go code in `orion/backend/lattigo/` into a platform-specific shared library (`.so`/`.dylib`/`.dll`) that Python loads via ctypes.

## Architecture

### Three-phase pipeline (`orion/core/orion.py` — `Scheme` class)

1. **`fit(net, dataloader)`** — Runs cleartext forward passes, uses `StatsTracker` to record per-layer min/max ranges. These ranges parameterize Chebyshev polynomial approximations for activations.
2. **`compile(net)`** — Traces the network into a DAG (`NetworkDAG`), computes multiplicative depth levels (`LevelDAG`), solves bootstrap placement (`BootstrapSolver`/`BootstrapPlacer`), and generates packed diagonal matrices for linear transforms. Returns `input_level` in normal mode, or `(input_level, manifest)` in keyless mode.
3. **Inference** — `encode` → `encrypt` → `net.he()` → forward pass on ciphertexts → `decrypt` → `decode`.

#### Client-server split (experimental, v2 branch)

The pipeline can be split into keyless compilation and key-imported inference:

1. **`init_params_only(config)`** — Creates backend + encoder only, no keys. Sets `scheme.keyless = True`.
2. **`compile(net)` in keyless mode** — Returns `(input_level, manifest)` where manifest lists all required Galois elements, bootstrap slot counts, and RLK flag. Rotation key generation is replaced by manifest collection via `lt_evaluator.required_galois_elements`.
3. **Key import path** — Server loads eval keys via `LoadRelinKey`, `LoadRotationKey`, `LoadBootstrapKeys` FFI functions, then constructs evaluators via `NewEvaluatorFromKeys()`.

The Go `Scheme` struct has an `EvalKeys *rlwe.MemEvaluationKeySet` field for holding imported evaluation keys. `Rotate()`/`RotateNew()` no longer lazy-generate keys when `scheme.SecretKey` is nil. See `experiments/REPORT.md` for detailed findings.

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

- **Lattigo (Go):** `orion/backend/lattigo/` — Go implementation of CKKS operations (12 files), exposed to Python via ctypes FFI in `bindings.py`. This is the only fully implemented backend.
- **Python:** `orion/backend/python/` — Python-side wrappers that call into Lattigo: `parameters.py`, `key_generator.py`, `encoder.py`, `encryptor.py`, `evaluator.py`, `poly_evaluator.py`, `lt_evaluator.py`, `bootstrapper.py`, `tensors.py` (`PlainTensor`/`CipherTensor`).
- **HEAAN, OpenFHE:** Placeholder directories, not yet implemented.

### Models (`orion/models/`)

Pre-built architectures using Orion's custom layers: LoLA, LeNet, MLP, VGG, AlexNet, ResNet, YOLOv1.

### Configuration (`configs/*.yml`)

YAML files specifying CKKS parameters (`LogN`, `LogQ`, `LogP`, `LogScale`, `H`, `RingType`) and Orion settings (`margin`, `embedding_method`, `backend`, `fuse_modules`, `diags_path`, `keys_path`, `io_mode`).

## Conventions

- Top-level API is flat: `orion.init_scheme()`, `orion.init_params_only()`, `orion.fit()`, `orion.compile()`, `orion.encode()`, `orion.encrypt()`, etc. (re-exported from `orion/core/`).
- Backend objects follow a `NewXxx(scheme)` factory pattern (e.g., `NewEncoder`, `NewEvaluator`). `NewEvaluatorFromKeys()` is a server-side variant that builds evaluators from pre-imported keys.
- Go FFI functions in `bindings.py` use `LattigoFunction` wrapper for type marshalling and `LattigoLibrary` for shared library lifecycle.
- Go FFI serialization exports follow the pattern: `SerializeXxx() -> (*C.char, C.ulong)` returning `ArrayResultByte`. Loading uses `LoadXxx(dataPtr *C.char, lenData C.ulong)`.
- HDF5 files (`.h5`) are used to persist diagonals and keys for reuse across runs.
