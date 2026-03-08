# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Orion

An opinionated fork of [baahl-nyu/orion](https://github.com/baahl-nyu/orion), a research-grade FHE framework for deep learning inference. The fork refactors Orion for practical usage: instance-based API (no global state), explicit context passing, and full access to underlying Lattigo primitives.

Orion takes PyTorch neural networks, analyzes them, and produces artifacts that enable encrypted inference using the CKKS scheme. The core pipeline is: **fit** (collect value range statistics) → **compile** (assign FHE levels, place bootstraps, pack data) → **encrypt & infer** (run on ciphertexts).

See `ARCH.md` for the full target architecture, compiled model format specification, evaluator API design, repo structure, and design rationale.

## Repository Structure

Three Python packages (`python/lattigo/`, `python/orion-compiler/`, `python/orion-evaluator/`), a Go evaluator (`evaluator/`), a JS/WASM package (`js/lattigo/`), and a browser demo (`examples/wasm-demo/`). Model examples under `examples/models/` (`{mlp,lenet,lola,alexnet,vgg,resnet}.py`) with unified `run.py` and `train.py`. See `ARCH.md § Repo Structure` for the full directory tree.

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

Orion provides **compilation**, **encoding**, and **evaluation** — never constrain the user's access to Lattigo primitives. No `Client` class, no `orion-client` package. Compiled model stores raw float64 data, not Lattigo artifacts. No backward compatibility with legacy code. See `ARCH.md § Components → No orion-client package` for full rationale.

## End-to-end Usage

```python
import orion_compiler.nn as on
from orion_compiler import Compiler, CKKSParams, CompiledModel
from lattigo.ckks import Parameters, Encoder
from lattigo.rlwe import KeyGenerator, Encryptor, Decryptor, MemEvaluationKeySet
from orion_evaluator import Model, Evaluator

# 1. Define model using orion_compiler.nn layers
class MLP(on.Module):
    def __init__(self):
        super().__init__()
        self.flatten = on.Flatten()
        self.fc1 = on.Linear(784, 128)
        self.act1 = on.Quad()
        self.fc2 = on.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        return self.fc2(x)

# 2. Compile
net = MLP()
compiler = Compiler(net, CKKSParams(logn=14, logq=[...], logp=[...], logscale=40))
compiler.fit(dataloader)
compiled = compiler.compile()
model_bytes = compiled.to_bytes()

# 3. Client — keygen + encrypt using Lattigo primitives directly
params = Parameters.from_logn(logn=14, logq=[...], logp=[...], logscale=40)
kg = KeyGenerator.new(params)
sk = kg.gen_secret_key()
pk = kg.gen_public_key(sk)
encoder = Encoder.new(params)
encryptor = Encryptor.new(params, pk)
pt = encoder.encode(input_values, level=compiled.input_level, scale=params.default_scale())
ct = encryptor.encrypt_new(pt)
ct_bytes = ct.marshal_binary()

# 4. Server — Go evaluator via orion-evaluator
model = Model.load(model_bytes)
params_dict, _, _ = model.client_params()
keys_bytes = evk.marshal_binary()  # MemEvaluationKeySet
evaluator = Evaluator(params_dict, keys_bytes)
result_bytes = evaluator.forward(model, ct_bytes)

# 5. Client — decrypt
from lattigo.rlwe import Ciphertext as RLWECiphertext
result_ct = RLWECiphertext.unmarshal_binary(result_bytes)
decryptor = Decryptor.new(params, sk)
result_pt = decryptor.decrypt_new(result_ct)
output = encoder.decode(result_pt, params.max_slots())
```

## Package Details

### Python packages

- `lattigo.ckks` — `Parameters`, `Encoder`
- `lattigo.rlwe` — `SecretKey`, `PublicKey`, `RelinearizationKey`, `GaloisKey`, `Ciphertext`, `Plaintext`, `KeyGenerator`, `Encryptor`, `Decryptor`, `MemEvaluationKeySet`
- `lattigo.gohandle` — `GoHandle` RAII wrapper for cgo.Handle values
- `orion_compiler` — `Compiler`, `CKKSParams`, `CompiledModel`, `Graph`, `GraphNode`, `GraphEdge`, `KeyManifest`, `CompilerConfig`, `CostProfile`
- `orion_compiler.nn` — FHE-compatible layers (cleartext-only forward)
- `orion_compiler.core` — Compilation algorithms (tracer, packing, level assignment, auto-bootstrap, galois)
- `orion_evaluator.Model` — `load()`, `client_params()`, `close()`
- `orion_evaluator.Evaluator` — `__init__(params, keys_bytes, btp_keys_bytes=None)`, `forward(model, ct_bytes) → bytes`, `close()`

### Go evaluator (`evaluator/`)

- `evaluator/format.go` — Binary format parser
- `evaluator/graph.go` — Computation graph with topological ordering
- `evaluator/model.go` — `Model` (immutable, shareable): `LoadModel`, `ClientParams()`
- `evaluator/evaluator.go` — `Evaluator` (per-client): `NewEvaluatorFromKeySet`, `Forward`

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
