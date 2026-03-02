# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Orion

An opinionated fork of [baahl-nyu/orion](https://github.com/baahl-nyu/orion), a research-grade FHE framework for deep learning inference. The fork refactors Orion for practical usage: instance-based API (no global state), explicit context passing, and full access to underlying Lattigo primitives.

Orion takes PyTorch neural networks, analyzes them, and produces artifacts that enable encrypted inference using the CKKS scheme. The core pipeline is: **fit** (collect value range statistics) → **compile** (assign FHE levels, place bootstraps, pack data) → **encrypt & infer** (run on ciphertexts).

See `ARCH.md` for the full target architecture, compiled model format specification, evaluator API design, and repo structure plan.

## Repository Structure

Three independent Python packages plus a Go evaluator plus a JS/WASM package:

```
python/lattigo/           # pip install lattigo — Lattigo CKKS bindings
  lattigo/                # Python: ckks.py, rlwe.py, ffi.py, gohandle.py
  bridge/                 # Go CGO: lattigo.go, types.go, main.go → .so

python/orion-compiler/    # pip install orion-compiler — Model compiler
  orion_compiler/         # Python: compiler.py, params.py, compiled_model.py
    nn/                   # Custom torch.nn.Module layers (Linear, Conv2d, etc.)
    core/                 # Algorithms: tracer, packing, level_dag, auto_bootstrap, galois
    models/               # Pre-built architectures (MLP, LeNet, VGG, etc.)

python/orion-evaluator/   # pip install orion-evaluator — Python bindings to Go evaluator
  orion_evaluator/        # Python: model.py, evaluator.py, ffi.py
  bridge/                 # Go CGO: evaluator.go, main.go → .so

evaluator/                # Pure Go FHE inference engine (subpackage of root module)
client/                   # Go client logic (keygen, encrypt, decrypt)
go.mod                    # Root module: github.com/baahl-nyu/orion
python/tests/             # All Python tests

js/lattigo/               # @orion/lattigo npm package — Lattigo WASM bindings
  bridge/                 # Go WASM: builds to wasm/lattigo.wasm (GOOS=js GOARCH=wasm)
  src/                    # TypeScript wrappers: ckks.ts, rlwe.ts, encoder.ts, loader.ts
  wasm/                   # lattigo.wasm + wasm_exec.js runtime
  tests/                  # vitest integration tests
  dist/                   # Built output: index.js + index.d.ts

js/examples/              # JS usage examples
  node/                   # Node.js: roundtrip.ts, eval-keys.ts

examples/wasm-demo/       # Browser demo: Go HTTP server + HTML/JS client
  server/                 # Go server: /params, /session, /session/{id}/infer endpoints
  client/                 # HTML/JS browser client
  model.orion             # Pre-compiled demo model
```

**Dependency graph:** `lattigo` ← `orion-compiler` (+ torch, networkx). `orion-evaluator` is independent. `js/lattigo` depends only on Lattigo (no Orion-specific code).

## Build & Development

**System prerequisites:** Go 1.22+, C compiler (CGO), libgmp-dev, libssl-dev, Python 3.9–3.12, Node.js 18+.

```bash
# Build the Python CGO shared library (required before installing Python packages)
python tools/build_lattigo.py

# Install Python packages in editable mode
cd python/lattigo && pip install -e .
cd python/orion-compiler && pip install -e .
cd python/orion-evaluator && pip install -e .

# Run all Python tests
pytest python/tests/

# Run a single Python test
pytest python/tests/test_v2_api.py::TestCompiler::test_compiler_produces_compiled_model

# Go evaluator tests
go test ./evaluator/...
go vet ./...

# Build JS/WASM binary (requires Go with js/wasm support)
python tools/build_lattigo_wasm.py

# JS/WASM package — install deps, build TypeScript, run tests
cd js/lattigo && npm install && npm run build && npm test

# JS/WASM — build only WASM (Go bridge to lattigo.wasm)
cd js/lattigo && npm run build:wasm

# JS/WASM — build only TypeScript wrappers
cd js/lattigo && npm run build:ts

# JS/WASM — type-check without emitting
cd js/lattigo && npm run typecheck

# JS/WASM — lint TypeScript
cd js/lattigo && npm run lint
```

## Design Principles

### Don't Constrain Lattigo Usage

Orion provides **model compilation**, **plaintext encoding**, and **model evaluation**. Encryption and decryption are the user's domain — Orion must not hide or restrict access to the underlying Lattigo primitives. Users may need per-ciphertext control for threshold encryption, custom key management, hybrid schemes, or any protocol Lattigo supports.

### Compiled model stores raw numerical data, not Lattigo artifacts

The compiled model is a portable mathematical description: raw float64 diagonal matrices, bias vectors, polynomial coefficients, and a computation graph with edges. No Lattigo `MarshalBinary` blobs. The evaluator CKKS-encodes the raw data into Lattigo format at load time.

### No backward compatibility with legacy code

This is a full refactor. Every line of code should serve the target architecture only. No compatibility shims, no legacy fallbacks.

## End-to-end Usage

```python
from orion_compiler import Compiler, CKKSParams, CompiledModel
from orion_compiler.models import MLP
from lattigo.ckks import Parameters, Encoder
from lattigo.rlwe import KeyGenerator, Encryptor, Decryptor, MemEvaluationKeySet
from orion_evaluator import Model, Evaluator

# 1. Compile
net = MLP()
compiler = Compiler(net, CKKSParams(logn=14, logq=[...], logp=[...], logscale=40))
compiler.fit(dataloader)
compiled = compiler.compile()
model_bytes = compiled.to_bytes()

# 2. Client — keygen + encrypt using Lattigo primitives directly
params = Parameters.from_logn(logn=14, logq=[...], logp=[...], logscale=40)
kg = KeyGenerator.new(params)
sk = kg.gen_secret_key()
pk = kg.gen_public_key(sk)
encoder = Encoder.new(params)
encryptor = Encryptor.new(params, pk)
pt = encoder.encode(input_values, level=compiled.input_level, scale=params.default_scale())
ct = encryptor.encrypt_new(pt)
ct_bytes = ct.marshal_binary()

# 3. Server — Go evaluator via orion-evaluator
model = Model.load(model_bytes)
keys_bytes = evk.marshal_binary()  # MemEvaluationKeySet
evaluator = Evaluator(model, keys_bytes)
result_bytes = evaluator.forward(model, ct_bytes)

# 4. Client — decrypt
from lattigo.rlwe import Ciphertext as RLWECiphertext
result_ct = RLWECiphertext.unmarshal_binary(result_bytes)
decryptor = Decryptor.new(params, sk)
result_pt = decryptor.decrypt_new(result_ct)
output = encoder.decode(result_pt, params.max_slots())
```

## Package Details

### lattigo (Python Lattigo bindings)

- `lattigo.ckks` — `Parameters`, `Encoder`
- `lattigo.rlwe` — `SecretKey`, `PublicKey`, `RelinearizationKey`, `GaloisKey`, `Ciphertext`, `Plaintext`, `KeyGenerator`, `Encryptor`, `Decryptor`, `MemEvaluationKeySet`
- `lattigo.gohandle` — `GoHandle` RAII wrapper for cgo.Handle values
- `lattigo.ffi` — Low-level ctypes bindings to bridge .so

### orion-compiler

- `orion_compiler.compiler` — `Compiler` class: traces, fits, compiles. `compile()` has zero Go/Lattigo dependency.
- `orion_compiler.params` — `CKKSParams` (frozen dataclass), `CompilerConfig`, `CostProfile`
- `orion_compiler.compiled_model` — `CompiledModel` (v2 format), `Graph`, `GraphNode`, `GraphEdge`, `KeyManifest`
- `orion_compiler.nn` — Custom `torch.nn.Module` layers for FHE (cleartext-only forward)
- `orion_compiler.core` — Compilation algorithms (tracer, packing, level assignment, auto-bootstrap, galois, compiler_backend)

### orion-evaluator

- `orion_evaluator.Model` — `load()`, `client_params()`, `close()`
- `orion_evaluator.Evaluator` — `__init__(model, keys_bytes)`, `forward(model, ct_bytes) → bytes`, `close()`

### Go evaluator (`evaluator/`)

Pure Go FHE inference engine. Reads `.orion` v2 files, CKKS-encodes diagonals at load time, walks the computation graph.

- `evaluator/format.go` — Binary format parser
- `evaluator/graph.go` — Computation graph with topological ordering
- `evaluator/model.go` — `Model` (immutable, shareable): `LoadModel`, `ClientParams()`
- `evaluator/evaluator.go` — `Evaluator` (per-client): `NewEvaluator`, `Forward`

### GoHandle — Go object lifecycle management

`GoHandle` (`lattigo/gohandle.py`) is an RAII wrapper for `cgo.Handle` values (`uintptr_t`). Rules:

1. **GoHandle wraps every Go object.** Tagged with descriptive strings (`"CKKSParams"`, `"RLWECiphertext"`, etc.).
2. **Bridge functions borrow, never consume.** Only `DeleteHandle` (called by `GoHandle.close()`) frees the handle slot.
3. **Canonical `__del__`.** Every handle-owning class uses `def __del__(self): try: self.close() except Exception: pass`.

### Serialization — Lattigo native, no custom formats

- Keys: `MemEvaluationKeySet.MarshalBinary()` / `UnmarshalBinary()`
- Ciphertexts: `rlwe.Ciphertext.MarshalBinary()` / `UnmarshalBinary()`
- Models: `.orion` v2 format (Go parser in `evaluator/format.go`)

## Conventions

- Three separate packages: `lattigo`, `orion-compiler`, `orion-evaluator`
- No `Client` class — users use Lattigo primitives directly
- Go evaluator is a subpackage of the root module (`github.com/baahl-nyu/orion/evaluator`)
- Tests in `python/tests/`, run with `pytest python/tests/`
