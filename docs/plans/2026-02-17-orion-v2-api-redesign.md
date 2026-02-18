# Orion v2 API Redesign

## Overview

Replace the monolithic `Scheme` singleton with three purpose-built classes: `Compiler`, `Client`, and `Evaluator` (all use Go backend). Eliminate global state, YAML configs, and HDF5 file management. All artifacts serialize to bytes — the library user controls storage and transport.

**Problem:** The current `Scheme` class is a parameter holder, key store, encoder factory, evaluator factory, and compilation context in one mutable singleton. The client-server split (validated in experiments 1-5) requires different subsets of that state on different machines, making the monolith untenable.

**Key design decisions (from brainstorming):**

- Three roles: `Compiler` (Go backend, no keys), `Client` (Go backend, has secret key), `Evaluator` (Go backend, no secret key)
- `CompiledModel` stores pre-encoded LinearTransform blobs (Lattigo binary) — no double encoding, instant load on Evaluator
- `Manifest` stores Galois elements (returned by Lattigo during transform encoding) — no rotation↔galEl conversion needed
- `Evaluator(net, compiled, keys)` — single constructor, no two-step load/init split (multi-client deferred until Go supports concurrent handles)
- **Context-on-tensor, not global state:** nn modules get their backend context from the input tensor (`x.context.evaluator`), not from a class variable. `module.compile(context)` takes an explicit parameter. Both `Module.scheme` and `Module.margin` class variables are deleted entirely — margin flows through `context.margin` alongside everything else.
- **Type split for compile vs inference evaluators (validated in Experiment 6):** The old `lt_evaluator(keyless=True/False)` boolean is replaced with separate types: `TransformEncoder` (Compiler: encodes diagonals, collects Galois elements, no Go evaluator needed) and `TransformEvaluator` (Evaluator: evaluates transforms on ciphertexts, needs Go evaluator + EvalKeys). Similarly `poly_evaluator` splits into `PolynomialGenerator` (Compiler: generates polynomial objects from coefficients, standalone Go functions) and `PolynomialEvaluator` (inherits Generator, adds `evaluate_polynomial()`, needs Go PolyEvaluator). No `keyless` booleans — the type system enforces which operations are available.
- **Server needs the model class definition** (e.g. `models.MLP()`) to construct a fresh skeleton, but NOT trained weights — weights are baked into LinearTransform blobs in CompiledModel
- No YAML configs, no HDF5, no global variables at module level
- Old flat API (`orion.init_scheme`, `orion.fit`, etc.) removed entirely — clean break, v2.0.0

**Known constraint — Go backend singleton:** The Lattigo shared library uses process-global state. Only one set of CKKS parameters and one set of evaluation keys can be active at a time. This means you cannot run two `Evaluator` instances (or a `Client` + `Evaluator`) with different parameters in the same process. The Python-side design is clean (no global state), but the Go layer enforces single-tenant semantics. Fixing this requires a Go-side handle/ID system — deferred to future work.

**Experiment 6 findings (evaluator keyless mode):** Validated that the Evaluator MUST create full (non-keyless) evaluator wrappers after `NewEvaluatorFromKeys()`. Specifically: (1) `EvaluatePolynomial` crashes with Go nil panic if `NewPolynomialEvaluator()` was never called (`polyeval.go:75`); (2) Python keyless `lt_evaluator.evaluate_transforms()` crashes with `AttributeError` because `self.evaluator` is not set (needed for `rescale()`); (3) Go `EvaluateLinearTransform` recreates `LinEvaluator` internally per call, making `NewLinearTransformEvaluator()` technically redundant at Go level; (4) polynomial generation functions (`GenerateChebyshev`, `GenerateMonomial`, `GenerateMinimaxSignCoeffs`) work without `NewPolynomialEvaluator()`, validating the Compiler's compile-time-only usage. These findings drove the type split design: `PolynomialGenerator`/`PolynomialEvaluator` and `TransformEncoder`/`TransformEvaluator` replace the `keyless` boolean.

**Target end-to-end usage:**

```python
# Compile (requires Go backend, no keys)
compiler = orion.Compiler(net, params)
compiler.fit(dataloader)
compiled = compiler.compile()  # -> CompiledModel
open("model.bin", "wb").write(compiled.to_bytes())

# Client (has secret key)
compiled = orion.CompiledModel.from_bytes(open("model.bin", "rb").read())
client = orion.Client(compiled.params)
keys = client.generate_keys(compiled.manifest)
pt = client.encode(input_tensor, level=compiled.input_level)
ct = client.encrypt(pt)

# Server (has model class definition, no trained weights needed)
compiled = orion.CompiledModel.from_bytes(open("model.bin", "rb").read())
net_skeleton = models.MLP()  # fresh instance — weights come from CompiledModel
evaluator = orion.Evaluator(net_skeleton, compiled, keys)
ct_result = evaluator.run(ct)
result = client.decode(client.decrypt(ct_result))
```

## Context (from discovery)

**Files to create:**

- `orion/params.py` — `CKKSParams`, `CompilerConfig`
- `orion/compiler.py` — `Compiler`
- `orion/client.py` — `Client`, `PlainText`, `CipherText`
- `orion/evaluator.py` — `Evaluator`
- `orion/compiled_model.py` — `CompiledModel`, `KeyManifest`, `EvalKeys`

**Files to modify:**

- `orion/__init__.py` — replace flat function exports with class exports
- `orion/core/__init__.py` — remove global scheme re-exports, make internal-only
- `orion/backend/python/parameters.py` — accept `CKKSParams` instead of YAML-parsed dict
- `orion/backend/python/poly_evaluator.py` — split into `PolynomialGenerator` (compile-time: `generate_chebyshev`, `generate_monomial`, `generate_minimax_sign_coeffs` — standalone Go functions, no Go PolyEvaluator needed) and `PolynomialEvaluator(PolynomialGenerator)` (inherits generation, adds `evaluate_polynomial` — calls `NewPolynomialEvaluator()` in constructor). Validated in Experiment 6: generation functions confirmed working without Go PolyEvaluator; `EvaluatePolynomial` confirmed crashing (nil panic at `polyeval.go:75`) without it.
- `orion/backend/python/lt_evaluator.py` — split into `TransformEncoder` (compile-time: `generate_transforms`, `get_galois_elements` — no Go evaluator needed) and `TransformEvaluator` (inference-time: `evaluate_transforms` — needs Go evaluator for rescale). Remove `keyless` boolean, HDF5 save/load methods, `io_mode`/`diags_path`/`keys_path` params. Note: Go `EvaluateLinearTransform()` recreates `LinEvaluator` internally on every call (`lineartransform.go:102-104`), so `NewLinearTransformEvaluator()` is technically redundant at the Go level, but the Python `TransformEvaluator` still needs a Python evaluator reference for `rescale()`.
- `orion/backend/python/tensors.py` — `CipherTensor`/`PlainTensor` gain a `context` field; add `CipherText.to_bytes()`/`from_bytes()` serialization
- `orion/backend/lattigo/lineartransform.go` — add `SerializeLinearTransform`, `LoadLinearTransform` exports
- `orion/backend/lattigo/bindings.py` — add FFI bindings for new Go exports
- `orion/core/auto_bootstrap.py` — update `BootstrapPlacer` to accept `context` parameter: `_create_bootstrapper(module, context)` passes context to `bootstrapper.compile(context)` instead of setting `bootstrapper.scheme = self.net.scheme` (which reads the deleted `Module.scheme` class variable)
- `orion/nn/*.py` — all modules refactored: `self.scheme.X` → `x.context.X` in HE forward, `compile()` → `compile(context)` with explicit parameter. `Module.scheme` class variable and `set_scheme()` deleted.

**Files to gut/delete:**

- `orion/core/orion.py` — extract logic into new classes, then delete `Scheme` class and module-level `scheme` singleton

**Files unchanged:**

- `orion/core/{tracer,network_dag,level_dag,packing,fuser}.py` — internal algorithms, used by Compiler
- `orion/models/*` — model architectures unchanged (they only define layer structure, not forward logic)

**Key coupling patterns (verified from source) and how they change:**

1. **`Module.scheme` and `Module.margin` class variables** — currently set via `Module.set_scheme(scheme)` and `Module.set_margin(margin)`, all modules share one instance. **v2: both deleted entirely.** Modules receive context via `module.compile(context)` at compile time and via `x.context` (from input tensor) at inference time. Margin flows through `context.margin` — only read during `fit()`, never during `compile()` or `forward()`.
2. **`CipherTensor`/`PlainTensor` store backend references at construction** — currently from `scheme`. **v2: from a `context` object passed at construction.** Each tensor carries its context and propagates it to output tensors.
3. **`module.compile()` reads `scheme.*` for backend calls** — **v2: `module.compile(context)` takes an explicit parameter.**
4. Bootstrap insertion uses **forward hooks** via `module.register_forward_hook()` — NOT module tree surgery. `BootstrapPlacer` creates `Bootstrap` instances and attaches them as hooks. Original module tree stays intact. **v2: `BootstrapPlacer` takes an explicit `context` parameter** instead of reading `self.net.scheme` (the deleted class variable). `_create_bootstrapper(module, context)` calls `bootstrapper.compile(context)` and sets `bootstrapper.margin` from `context.config.margin`.
5. The `@timer` decorator on every module's `forward()` **short-circuits when `he_mode=False`** — never touches context during cleartext forward passes. Safe for Compiler's `fit()`. **v2: `@timer` reads context from input tensor instead of `self.scheme`; short-circuit unchanged.**
6. **BSGS rotation computation is internal to Lattigo.** Python passes diagonal indices to Go's `GenerateLinearTransform()`, which internally computes BSGS decomposition. `GetLinearTransformRotationKeys()` returns Galois elements. Python never computes rotations from diagonals — Lattigo does it as a side effect of encoding. **Unchanged in v2.**
7. CipherText cross-process serialization: **confirmed working** in Experiment 5 via Lattigo's `MarshalBinary`/`UnmarshalBinary`. Receiver MUST have identical CKKS params initialized. **Unchanged in v2.**

**What gets pre-encoded vs raw in CompiledModel:**

| Data                        | Storage                          | Reason                                                                                    |
| --------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------- |
| LinearTransform (diagonals) | Pre-encoded Lattigo binary blobs | Expensive to encode (O(N log N) per diagonal), same memory footprint on server regardless |
| Bias vectors                | Raw float64                      | One encode per layer, cheap                                                               |
| Chebyshev coefficients      | JSON (list[float])               | Small, cheap to regenerate polynomial object                                              |
| Bootstrap prescale          | Raw float64                      | One encode per bootstrap, cheap                                                           |
| Module metadata             | JSON                             | Small structured data                                                                     |

**Context passing strategy:**

No facade. No class variable. Context flows through the data:

- **At compile time:** `Compiler` passes a context object to `module.compile(context)`. The context contains `backend`, `params`, `encoder`, `lt_evaluator` (a `TransformEncoder`), `poly_evaluator` (a `PolynomialGenerator`), `config`.
- **At inference time:** `CipherTensor` carries a `context` attribute with `backend`, `evaluator`, `encoder`, `lt_evaluator` (a `TransformEvaluator`), `poly_evaluator` (a `PolynomialEvaluator`), `bootstrapper`. Each module reads `x.context.evaluator` instead of `self.scheme.evaluator`. When a module produces an output `CipherTensor`, it propagates `x.context` to the output. The `Evaluator` constructs the initial `CipherTensor` with its context; all subsequent tensors inherit it automatically. **Critical:** `lt_evaluator.evaluate_transforms()` and `poly_evaluator.evaluate_polynomial()` must also propagate context from the input tensor to output tensors (currently they construct `CipherTensor(self.scheme, ...)` — must become `CipherTensor(in_ctensor.context, ...)`).
- **Bootstrap hooks:** The hook receives the output tensor, which already carries the context. `bootstrapper(output)` reads `output.context.bootstrapper` internally.

This eliminates all global state on the Python side. Two `Evaluator` instances with different contexts in the same process would work correctly at the Python level (blocked only by Go backend singleton).

## Development Approach

- **Testing approach**: Task-boundary testing, not per-sub-step
- Tests must pass at end of each task, not after every sub-step
- **Old test suite is expected to break during Task 3** (core refactor) — no Scheme compatibility shim
- New tests validate the new code paths as they're built; old tests are rewritten in Task 5
- **CRITICAL: update this plan file when scope changes during implementation**
- Run tests after each phase to catch regressions

## Testing Strategy

- **Tasks 1–2**: Pure additive code. Old test suite passes unchanged. New unit tests for new code.
- **Task 3**: Old test suite breaks (Scheme no longer works). New unit tests validate refactored modules directly.
- **Task 4**: Integration tests for new API classes. Compiler→Client→Evaluator roundtrip is the primary validation.
- **Task 5**: Old tests rewritten to new API. Full suite passes. Old code deleted.
- **Task 6**: Acceptance criteria verification. All tests green.

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix
- Update plan if implementation deviates from original scope

## Implementation Steps

### Task 1: Data structures and backend adapter

**Files:** `orion/params.py` (new), `orion/compiled_model.py` (new), `orion/backend/python/parameters.py` (modify)

Pure additive — no existing code changes, old test suite untouched.

**1a — CKKSParams and CompilerConfig:**

- [x] Create `CKKSParams` frozen dataclass with fields: `logn`, `logq`, `logp`, `logscale`, `h` (default 192), `ring_type` (default "conjugate_invariant"), `boot_logp` (optional)
- [x] Add computed properties: `max_level` (= `len(logq) - 1`), `max_slots` (= `2^logn` for conjugate_invariant, `2^(logn-1)` for standard), `ring_degree` (= `2^logn`)
- [x] Create `CompilerConfig` frozen dataclass with fields: `margin` (default 2), `embedding_method` (default "hybrid"), `fuse_modules` (default True)
- [x] Add `__post_init__` validation (logn > 0, logq/logp non-empty, ring_type in allowed values, embedding_method in allowed values)

**1b — KeyManifest, CompiledModel, EvalKeys:**

- [x] Create `KeyManifest` frozen dataclass with fields: `galois_elements` (frozenset[int]), `bootstrap_slots` (tuple[int, ...]), `boot_logp` (tuple[int, ...] | None), `needs_rlk` (bool). Validate: if `bootstrap_slots` is non-empty, `boot_logp` must not be None.
- [x] Implement `CompiledModel` internal structure: `params`, `manifest`, `input_level`, `module_metadata` (dict), `topology` (list[str]), `blobs` (list[bytes])
- [x] Implement `CompiledModel.to_bytes()` / `from_bytes()` binary format (see Technical Details): magic + metadata JSON + length-prefixed blobs + CRC32
- [x] Implement `EvalKeys` with `to_bytes()` / `from_bytes()`:
  - Internal storage: `rlk_data: bytes | None`, `galois_keys: dict[int, bytes]`, `bootstrap_keys: dict[int, bytes]`
  - Distinct magic (`ORKEY\x00\x01\x00`), same container pattern
  - Properties: `galois_elements` (set of ints), `has_rlk` (bool)

**1c — Backend adapter:**

- [x] Add `NewParameters.from_ckks_params(ckks_params: CKKSParams, config: CompilerConfig)` classmethod that builds the `params_json` dict from the new dataclasses and delegates to existing `__post_init__`
- [x] Ensure all existing `get_*()` methods work correctly with the new construction path

**Tests:** Unit tests for all dataclasses (construction, validation, computed properties, serialization roundtrips, CRC32 corruption detection). Run old test suite — must still pass.

---

### Task 2: Go FFI for LinearTransform serialization

**Files:** `orion/backend/lattigo/lineartransform.go` (modify), `orion/backend/lattigo/bindings.py` (modify)

Pure additive — new Go exports, old test suite untouched.

- [x] Add `SerializeLinearTransform(transformID C.int) (*C.char, C.ulong)` to `lineartransform.go`:
  - Retrieve the LinearTransform by ID from the Go heap
  - Custom serialization (~70 LOC): encode MetaData via `MarshalBinary()`, encode int fields (LogBabyStepGiantStepRatio, N1, LevelQ, LevelP), encode Vec map length, then for each `(diagIndex, poly)` pair encode index + `poly.MarshalBinary()`. Note: `lintrans.LinearTransformation` does NOT implement `encoding.BinaryMarshaler` — custom marshaling is required.
  - Return the byte array via `SliceToCArray`
- [x] Add `LoadLinearTransform(dataPtr *C.char, lenData C.ulong) C.int` to `lineartransform.go`:
  - Custom deserialization (reverse of above)
  - Push to Go heap, return ID
- [x] Add Python FFI bindings in `bindings.py`:
  - `self.SerializeLinearTransform = LattigoFunction(self.lib.SerializeLinearTransform, argtypes=[ctypes.c_int], restype=ArrayResultByte)`
  - `self.LoadLinearTransform = LattigoFunction(self.lib.LoadLinearTransform, argtypes=[ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulong], restype=ctypes.c_int)`
- [x] Rebuild shared library: `python tools/build_lattigo.py`

**Tests:** Serialize/deserialize roundtrip — create a LinearTransform via `GenerateLinearTransform`, serialize, deserialize, verify loaded ID is valid. Run old test suite — must still pass.

---

### Task 3: Core refactor — context passing and evaluator type split

**Files:** `orion/nn/module.py`, `orion/nn/*.py` (all modules), `orion/backend/python/tensors.py`, `orion/backend/python/poly_evaluator.py`, `orion/backend/python/lt_evaluator.py`, `orion/core/auto_bootstrap.py`

This is the foundational refactor. **Old test suite will break here** — `Scheme` is not maintained as a compatibility shim. The old tests are rewritten in Task 5.

**3a — Evaluator type split:**

- [x] Split `poly_evaluator.py` into two classes:
  - `PolynomialGenerator(backend)` — compile-time only. Constructor stores `self.backend = backend`. Methods: `generate_chebyshev(coeffs)`, `generate_monomial(coeffs)`, `generate_minimax_sign_coeffs(...)`. These call standalone Go functions that do NOT need Go `PolyEvaluator`. Confirmed in Experiment 6 test 4.
  - `PolynomialEvaluator(backend)` inherits from `PolynomialGenerator`. Constructor additionally calls `self.backend.NewPolynomialEvaluator()`. Adds method: `evaluate_polynomial(ciphertensor, poly, out_scale)`. Output `CipherTensor` must propagate context from input tensor.
- [x] Split `lt_evaluator.py` into two classes:
  - `TransformEncoder(backend, params)` — compile-time only. Methods: `generate_transforms(linear_layer)`, `get_galois_elements(transform_id)`. Accumulates `self.required_galois_elements: set[int]`. No Go evaluator needed.
  - `TransformEvaluator(backend, evaluator)` — inference-time only. Stores Python evaluator wrapper, calls `backend.NewLinearTransformEvaluator()`. Method: `evaluate_transforms(linear_layer, in_ctensor)`. Output `CipherTensor` propagates `in_ctensor.context`.
  - Remove `keyless` boolean, HDF5 methods, `io_mode`/`diags_path`/`keys_path` fields.

**3b — Context-on-tensor:**

- [x] Add `context` field to `CipherTensor` and `PlainTensor`:
  - `CipherTensor.__init__(self, context, ctxt_ids, shape, on_shape=None)` — stores `self.context = context` and derives `self.backend`, `self.evaluator`, etc. from it
  - `PlainTensor.__init__(self, context, ptxt_ids, shape, on_shape=None)` — stores `self.context = context` and derives `self.backend`, `self.encoder` from it

**3c — Module refactor:**

- [x] Change `module.compile()` → `module.compile(context)` across all nn modules:
  - Every module that reads `self.scheme.encoder`, `self.scheme.lt_evaluator`, `self.scheme.poly_evaluator` in `compile()` now reads `context.encoder`, `context.lt_evaluator`, `context.poly_evaluator`
- [x] Change all HE-mode `forward()` methods to read context from input tensor:
  - Replace `self.scheme.evaluator` → `x.context.evaluator` (or first CipherTensor arg's context)
  - Replace `self.scheme.encoder` → `x.context.encoder`, etc.
  - When constructing output `CipherTensor`/`PlainTensor`, propagate `x.context`
  - Update `@timer` decorator to read debug status from `args[0].context.params` instead of `self.scheme.params`
- [x] Delete `Module.scheme` class variable, `Module.set_scheme()` static method, `Module.margin` class variable, and `Module.set_margin()` static method

**3d — BootstrapPlacer and fit():**

- [x] Update `BootstrapPlacer` to accept `context` parameter:
  - `BootstrapPlacer.__init__(self, net, network_dag, context)` — stores `self.context = context`
  - `_create_bootstrapper(module)`: pass context to `bootstrapper.fit(context)` and `bootstrapper.compile(context)`. Margin comes from `context.margin`.
- [x] Update all `module.fit()` methods to accept context: `module.fit(context)`:
  - `Bootstrap.fit(context)`: replace `self.margin` → `context.margin`
  - `Chebyshev.fit(context)`: replace `self.margin` → `context.margin`
  - `ReLU.fit(context)`: replace `self.margin` → `context.margin`
  - `_Sign.fit(context)`: needs `context.poly_evaluator` and `context.margin`

**Tests:** New unit tests for: PolynomialGenerator/PolynomialEvaluator split, TransformEncoder/TransformEvaluator split, CipherTensor/PlainTensor context propagation, `module.compile(context)` signature. **Old test suite expected to fail** — Scheme references removed, will be rewritten in Task 5.

---

### Task 4: New API classes — Compiler, Client, Evaluator

**Files:** `orion/compiler.py` (new), `orion/client.py` (new), `orion/evaluator.py` (new), `orion/compiled_model.py` (extend EvalKeys)

Build all three classes together. The primary test is the end-to-end roundtrip: Compiler→Client→Evaluator.

**4a — Compiler:**

- [x] Create `Compiler.__init__(self, net: Module, params: CKKSParams, config: CompilerConfig | None = None)`:
  - Create `NewParameters` via the adapter from Task 1
  - Initialize Go backend with params (no keys — `init_params_only` equivalent)
  - Build compilation context: `backend`, `params`, `encoder`, `poly_evaluator` (`PolynomialGenerator`), `lt_evaluator` (`TransformEncoder`), `config` (with `config.margin` for fit)
- [x] Implement `fit()` — extract from `Scheme.fit()`:
  - `OrionTracer.trace_model(net)`, `StatsTracker.propagate()`, `tracer.sync_module_attributes()`
  - Call `module.fit(context=self.context)` on activations and bootstraps
- [x] Implement `compile()` — extract from `Scheme.compile()`, all steps in order:
  - Build `NetworkDAG`, call `build_dag()`
  - `init_orion_params()` on all modules (clones weights/biases into `on_weight`, `on_bias`)
  - `update_params()` on pooling modules
  - Run `Fuser` if `config.fuse_modules`
  - Force last `LinearTransform` to `"square"` embedding
  - `module.generate_diagonals(last=...)` for each LinearTransform
  - `network_dag.find_residuals()`
  - `BootstrapSolver` → assigns levels, returns `(input_level, num_bootstraps, bootstrapper_slots)`
  - `BootstrapPlacer(net, network_dag, self.context)` → registers forward hooks
  - `module.compile(self.context)` on all modules in topological order. For LinearTransform:
    - `context.lt_evaluator.generate_transforms()` (encode diagonals)
    - `GetLinearTransformRotationKeys(transformID)` (Galois elements)
    - `SerializeLinearTransform(transformID)` (blob for CompiledModel)
    - Store raw bias float64 as blob
  - Collect Galois elements: from transforms, power-of-2, hybrid output
  - Build `KeyManifest` and `CompiledModel`
- [x] Implement `_extract_module_metadata()` and `_extract_topology()` helpers

**4b — Client:**

- [x] Create `PlainText` class wrapping backend plaintext IDs (local-only, no cross-process serialization)
- [x] Create `CipherText` class wrapping backend ciphertext IDs with `to_bytes()` / `from_bytes()`
- [x] Create `Client.__init__(self, params: CKKSParams)`:
  - Initialize Go backend with full key generation
  - Create `NewKeyGenerator`, `NewEncoder`, `NewEncryptor`
- [x] Implement `generate_keys(manifest: KeyManifest) -> EvalKeys`:
  - Generate and serialize rlk, rotation keys (per Galois element), bootstrap keys (per slot count)
- [x] Implement `encode`/`decode`/`encrypt`/`decrypt`

**4c — Evaluator:**

- [x] Implement `Evaluator.__init__(self, net: Module, compiled: CompiledModel, keys: EvalKeys)`:
  1. Init Go backend with params, no keys
  2. Create encoder
  3. Load rlk, rotation keys, bootstrap keys into Go backend
  4. `backend.NewEvaluatorFromKeys()` — sets `scheme.Evaluator` on Go side
  5. Create inference-time wrappers (**MUST happen after step 4**):
     - Python evaluator wrapper (wraps Go evaluator, no SK needed)
     - `TransformEvaluator(backend, python_evaluator)`
     - `PolynomialEvaluator(backend)` — needs `scheme.Evaluator` set
     - **Rationale (Experiment 6):** `PolynomialGenerator` insufficient — `evaluate_polynomial()` nil panics at `polyeval.go:75` without `NewPolynomialEvaluator()`. `TransformEncoder` insufficient — `evaluate_transforms()` needs `self.evaluator.rescale()`.
  6. Create bootstrapper evaluators
  7. Build inference context namespace
  8. Walk `compiled.module_metadata`, apply to `net`:
     - LinearTransform: `LoadLinearTransform(blob)` → set `module.transform_ids`, encode bias
     - Activations: set coeffs/prescale/etc, `module.compile(context)` to regenerate polynomial objects
     - Bootstrap hooks: create Bootstrap instances, register forward hooks on targets
- [x] Implement `Evaluator.run(ct: CipherText) -> CipherText`:
  - Convert to `CipherTensor(self.context, ...)`, `net.he()`, forward pass, convert back

**Tests:** Integration tests: Compiler→Client→Evaluator roundtrip with simple MLP, CompiledModel `to_bytes()`→`from_bytes()`→Evaluator works, verify modules have correct levels/depths after Evaluator construction, verify CipherText serialization roundtrip.

---

### Task 5: Switchover — rewrite tests, update API, delete old code

**Files:** `orion/__init__.py`, `orion/core/__init__.py`, `orion/core/orion.py` (delete), `tests/` (rewrite), `examples/` (rewrite)

Single atomic cut: old API out, new API in, all tests rewritten.

- [x] Rewrite `orion/__init__.py`:
  ```python
  from orion.params import CKKSParams, CompilerConfig
  from orion.compiler import Compiler
  from orion.client import Client, PlainText, CipherText
  from orion.evaluator import Evaluator
  from orion.compiled_model import CompiledModel, KeyManifest, EvalKeys
  __version__ = "2.0.0"
  ```
- [x] Remove flat API re-exports from `orion/core/__init__.py`
- [x] Delete `orion/core/orion.py` entirely
- [x] Remove YAML config parsing
- [x] Rewrite all existing tests to new API pattern (Compiler→Client→Evaluator)
- [x] Add integration test: full roundtrip with MLP/MNIST
- [x] Add test: CompiledModel `to_bytes()` → `from_bytes()` → `Evaluator()` works
- [x] Rewrite examples (`run_mlp.py`, `run_lola.py`, `run_resnet.py`) to new API

**Tests:** Full test suite must pass. All old `orion.init_scheme`/`orion.fit`/etc. references eliminated.

---

### Task 6: Verification and cleanup

- [ ] Verify Compiler produces correct CompiledModel for both smooth-activation and ReLU networks
- [ ] Verify CompiledModel serialization roundtrip preserves all data
- [ ] Verify EvalKeys serialization roundtrip works
- [ ] Verify CipherText serialization roundtrip works (client → bytes → server)
- [ ] Verify no global state: no `Module.scheme`, no module-level `scheme` variable, no YAML file paths, no HDF5 file paths
- [ ] Verify context-on-tensor: two `Evaluator` instances in one process don't interfere at Python level
- [ ] Remove unused imports across modified files
- [ ] Update CLAUDE.md with new API documentation and architecture description
- [ ] Verify `pip install -e .` still works with new structure
- [ ] Run full test suite
- [ ] Run linter — all issues must be fixed

## Technical Details

### Manifest stores Galois elements

The `KeyManifest` stores **Galois elements** (uint64 integers) directly, as returned by Lattigo during transform encoding. The flow:

1. Compiler calls `context.lt_evaluator.generate_transforms(module)` (a `TransformEncoder`) → Go encodes diagonals, creates LinearTransform. `TransformEncoder` calls `get_galois_elements(transformID)` internally and accumulates them into `self.required_galois_elements`.
2. Under the hood, `get_galois_elements()` calls `GetLinearTransformRotationKeys(transformID)` → Go returns Galois elements via `transform.GaloisElements(params)` (BSGS decomposition is internal to Lattigo)
3. Compiler calls `backend.GetGaloisElement(rotation)` for power-of-2 and hybrid output rotations
4. All Galois elements are collected into `KeyManifest.galois_elements`

Client uses Galois elements directly in `GenerateAndSerializeRotationKey(galEl)`. Evaluator uses them directly in `LoadRotationKey(data, len, galEl)`. No conversion needed.

### Pre-encoded LinearTransform blobs

The Compiler encodes diagonals into Lattigo LinearTransform objects during `compile()`, then serializes them via `SerializeLinearTransform()`. CompiledModel stores these opaque Lattigo binary blobs.

The Evaluator deserializes them via `LoadLinearTransform()` — this is pure deserialization (fast I/O), not re-encoding. The Go-side transform is immediately ready for use in `evaluate_transforms()`.

Small data (bias vectors, Chebyshev coefficients, bootstrap prescale) is stored raw and encoded at `Evaluator` construction time — this is cheap (one encode per layer).

### Bootstrap insertion mechanism (verified from source)

Bootstraps are NOT inserted via module tree surgery or FX graph rewriting. The mechanism is:

1. `BootstrapSolver.mark_bootstrap_locations()` sets `network_dag.nodes[node]["bootstrap"] = True` on marked nodes
2. `BootstrapPlacer.place_bootstraps()` iterates marked nodes and:
   - Creates a `Bootstrap(input_min, input_max, input_level)` instance
   - Sets `module.bootstrapper = bootstrap_instance` as an attribute
   - Calls `module.register_forward_hook(lambda mod, inp, out: bootstrapper(out))`
3. During inference, PyTorch's hook mechanism intercepts the module's output and passes it through the bootstrapper

**Implication for Evaluator constructor:** replicate this exactly — create Bootstrap instances from CompiledModel metadata, register forward hooks on the corresponding modules. No graph manipulation needed.

### CompiledModel binary format

```
[8 bytes]  MAGIC ("ORMDL\x00\x01\x00")
[4 bytes]  METADATA_LENGTH (uint32 LE)
[N bytes]  METADATA_JSON (utf-8 encoded)
[4 bytes]  BLOB_COUNT (uint32 LE)
for each blob:
  [8 bytes]  BLOB_LENGTH (uint64 LE)
  [N bytes]  BLOB_DATA
[4 bytes]  CRC32 of everything above
```

**Metadata JSON schema:**

```json
{
  "version": 1,
  "params": {
    "logn": 14,
    "logq": [55, 40, 40],
    "logp": [61, 61],
    "logscale": 40,
    "h": 192,
    "ring_type": "conjugate_invariant",
    "boot_logp": null
  },
  "config": {
    "margin": 2,
    "embedding_method": "hybrid",
    "fuse_modules": true
  },
  "manifest": {
    "galois_elements": [5, 25, 125, 625, 3125],
    "bootstrap_slots": [4096],
    "boot_logp": [61, 61],
    "needs_rlk": true
  },
  "input_level": 10,
  "modules": {
    "fc1": {
      "type": "Linear",
      "level": 10,
      "depth": 1,
      "fused": false,
      "bsgs_ratio": 1.0,
      "output_rotations": 3,
      "transform_blobs": {
        "0,0": 0,
        "0,1": 1
      },
      "bias_blob": 2
    },
    "act1": {
      "type": "Sigmoid",
      "level": 9,
      "depth": 2,
      "fused": false,
      "coeffs": [0.5, 0.25, 0.125],
      "prescale": 1.5,
      "constant": -0.3,
      "output_scale": null,
      "input_min": -3.0,
      "input_max": 3.0,
      "fhe_input_shape": [1, 64],
      "fhe_output_shape": [1, 64]
    },
    "__bootstrap_0": {
      "type": "Bootstrap",
      "hook_target": "act1",
      "input_level": 9,
      "input_min": -1.5,
      "input_max": 1.5,
      "prescale": 0.5,
      "postscale": 2.0,
      "constant": -0.3,
      "fhe_input_shape": [1, 64],
      "slots": 4096
    },
    "fc2": {
      "type": "Linear",
      "level": 5,
      "depth": 1,
      "fused": false,
      "bsgs_ratio": 1.0,
      "output_rotations": 0,
      "transform_blobs": {
        "0,0": 3
      },
      "bias_blob": 4
    }
  },
  "topology": ["fc1", "act1", "__bootstrap_0", "fc2"],
  "blob_count": 5
}
```

### EvalKeys binary format

Same container pattern but distinct magic: `ORKEY\x00\x01\x00` + JSON index + length-prefixed blobs + CRC32.

```json
{
  "version": 1,
  "rlk_blob_index": 0,
  "galois_keys": {
    "5": 1,
    "25": 2,
    "125": 3,
    "625": 4
  },
  "bootstrap_keys": {
    "4096": 5
  },
  "blob_count": 6
}
```

Galois keys are indexed by **Galois element** (uint64). Both Client and Evaluator use Galois elements natively — no conversion needed.

### CipherText cross-process transfer (verified in Experiment 5)

Go serialization at `tensors.go:133-156`:

```go
// SerializeCiphertext: ct.MarshalBinary() -> bytes
// LoadCiphertext: ct.UnmarshalBinary(bytes) -> heap ID
```

Requirements:

- Receiver MUST have identical CKKS params initialized (same LogN, LogQ, ring type)
- Ciphertext binary is self-contained (includes degree, level, scale, polynomial data)
- Tested end-to-end in Experiment 5: MAE 0.000599 vs cleartext

## Post-Completion

**Manual verification:**

- Test with a real MNIST model end-to-end (compile → serialize → deserialize → inference)
- Benchmark `Evaluator` construction time for production-size models
- Measure CompiledModel serialized size for different model architectures

**Future work (not in this plan):**

- Go handle system for concurrent multi-client Evaluators (the only remaining singleton constraint — Python side is clean after this plan)
- Pure Go inference server (execute compiled DAG without Python)
- Pure Python compiler (port `GenerateMinimaxSignCoeffs` and Galois element formulas to Python, separate `orion-compiler` pip package)
- Key manifest completeness verification
- Streaming/chunked serialization for very large CompiledModels
