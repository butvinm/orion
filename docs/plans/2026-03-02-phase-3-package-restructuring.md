# Phase 3: Package Restructuring

## Overview

Split the Python monolith (`orion/`) into three packages, clean up dead code, restructure Go modules, and eliminate custom serialization formats in favor of Lattigo's native serialization.

Target structure:

- `python/lattigo/` — Python bindings for Lattigo primitives (`pip install lattigo`)
- `python/orion-compiler/` — Compiler, nn modules, core algorithms, CKKSParams, CompiledModel (`pip install orion-compiler`)
- `python/orion-evaluator/` — Python bindings to Go evaluator (`pip install orion-evaluator`)

No `orion-client` package. Users interact with Lattigo directly for keygen, encrypt, decrypt, encode, decode. This keeps Orion independent from the encryption scheme — users can use threshold encryption, custom key management, or any Lattigo-compatible protocol.

Plus:

- Root `go.mod` with single module for evaluator + shared types
- Delete `orionclient/` entirely
- `Client`, `Ciphertext`/`PlainText` Python classes, `EvalKeys`/`ORKEY` format — not moved to new packages (die when `orion/` is deleted)

## Context

**Completed prerequisites:**

- Phase 1 (compiled model v2 with computation graph, pure Python BSGS, `compile()` is Go-free)
- Phase 2 (pure Go evaluator in `evaluator/` — LoadModel, NewEvaluator, Forward)

**Key decisions:**

- **RESOLVED:** `GenerateLinearTransform` / `LinearTransformRequiredGaloisElements` confirmed safe to remove (see Task 3)
- **No `Client` class** — users use Lattigo bindings directly (design principle: don't constrain Lattigo usage)
- **No custom serialization formats** — keys use `MemEvaluationKeySet.MarshalBinary()`, ciphertexts use Lattigo's native `MarshalBinary()`. ORKEY format and Python ciphertext shape header are not carried forward
- **Evaluator accepts raw Lattigo bytes** — no Orion-specific key or ciphertext types in the evaluator API
- Regular testing approach (code first, then tests)

**Ordering principle:** `Client`, `Ciphertext`/`PlainText`, `EvalKeys`, and `Client*` bridge exports are still used by existing tests. They are NOT deleted early — they remain functional until the new `lattigo` package replaces them. They die naturally when the old `orion/` directory is deleted at the end.

## Development Approach

- **Testing approach**: Regular (implement, then verify with existing + new tests)
- Complete each task fully before moving to the next
- **CRITICAL: every task MUST include new/updated tests**
- **CRITICAL: all tests must pass before starting next task**
- **CRITICAL: update this plan file when scope changes during implementation**

## Testing Strategy

- **Python tests**: `pytest tests/`
- **Go tests**: `cd evaluator && go test ./...`

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix

## Implementation Steps

### Task 1: Strip FHE forward branches from nn modules (ARCH.md 3.3)

Dead code removal. The Python evaluator was deleted in Phase 1, so all `if self.he_mode` branches are unreachable.

- [x] Remove `he_mode` flag, `he()` method, `_set_mode_for_all()`, and `train()`/`eval()` overrides from `orion/nn/module.py`
- [x] Remove `timer` decorator from `orion/nn/module.py` — its only purpose is FHE timing (`if not self.he_mode: return`; else measure). Without `he_mode` it's a no-op. Remove `@timer` from all decorated methods in `linear.py`, `activation.py`, `normalization.py`, `operations.py`
- [x] Strip `if self.he_mode` / `if not self.he_mode` branches from `orion/nn/linear.py` — keep only cleartext path
- [x] Remove dead `LinearTransform.compile()` and `LinearTransform.evaluate_transforms()` methods from `orion/nn/linear.py` (never called — see Task 3). Also remove dead attributes: `self.transform_ids = {}`, `self.transform_handles = {}`
- [x] Strip FHE branches from `orion/nn/activation.py`
- [x] Strip FHE branches from `orion/nn/normalization.py`
- [x] Strip FHE branches from `orion/nn/pooling.py`
- [x] Strip FHE branches from `orion/nn/operations.py`
- [x] Strip FHE branch from `orion/nn/reshape.py`
- [x] Remove dead imports (`Ciphertext`, `PlainText`, etc.) from nn modules
- [x] Fix `compiler.py` line 216: `if hasattr(module, "he_mode")` guard for `module.scheme = self` — replace with `if isinstance(module, Module)` (import from `orion.nn.module`). This is needed because `packing.py` and `level_dag.py` read `module.scheme.params`
- [x] Run `pytest tests/` — all tests must pass

### Task 2: Clean up Ciphertext/PlainText dead code (ARCH.md 3.3)

Remove only the truly dead code from `ciphertext.py`. Keep `Client`, `Ciphertext`, `PlainText` classes functional — they are still used by tests until Task 7 replaces them.

- [x] Remove `context` attribute from `Ciphertext.__init__()` and all references
- [x] Remove dead arithmetic methods from `Ciphertext`: `add()`, `sub()`, `mul()`, `roll()`, `bootstrap()`, `_eval_h()`, `_wrap()`, all dunder operators (`__add__`, `__sub__`, `__mul__`, `__rmul__`, `__neg__`, `__iadd__`, `__isub__`, `__imul__`)
- [x] Remove `on_shape` attribute (only read by dead arithmetic methods and skipped tests)
- [x] Remove dead `PlainText.mul()` and `PlainText.__mul__`
- [x] Remove `_EvalContext` remnants if any exist
- [x] Verify `Ciphertext` still works for client-side usage: shape tracking, Go handle wrapping, serialization, `Client.encrypt()`/`Client.decrypt()`
- [x] Verify no other files reference removed methods (grep for `ct.add(`, `ct.mul(`, `ct.context`, etc.)
- [x] Run `pytest tests/` — all tests must pass (including Client tests)

### Task 3: ~~Investigate GenerateLinearTransform redundancy~~ ✅ COMPLETED

**Verdict: SAFE TO REMOVE.** Investigation completed via `experiments/test_lt_removal.py`.

- [x] Trace all callers — `LinearTransform.compile()` (`linear.py:43-45`) is the only caller, and it is **never invoked** in the standard pipeline (`Compiler.compile()` line 271: "no module.compile() calls")
- [x] `fit()` never calls LT functions (verified by bomb-patching: no calls triggered)
- [x] `compile()` never calls LT functions (verified by bomb-patching: no calls triggered)
- [x] Compiled output is **byte-identical** with or without LT functions (4,916,817 bytes both ways)
- [x] Pure Python BSGS matches Lattigo exactly (`test_galois.py::TestGaloisVsLattigo` — 3/3 pass)
- [x] LT handles leak memory in normal flow (`delete_transforms()` never called outside `test_galois.py`)

### Task 4: Prune dead bridge functions and compiler code (ARCH.md 3.1)

Remove only truly dead exports: eval-only, LT functions, EvalKeyBundle. Keep `Client*` bridge exports — they are still used by `orion/client.py` and tests.

**Bridge Go code:**

- [x] Delete `orionclient/bridge/evaluator.go` entirely (all Eval\* exports — dead since Python evaluator deleted)
- [x] Delete from `orionclient/bridge/types.go`: `NewEvalKeyBundle`, all `EvalKeyBundle*` setters, `GenerateLinearTransformFromParams`, `LinearTransformRequiredGaloisElements`, `LinearTransformMarshal`, `LinearTransformUnmarshal`

**Python FFI:**

- [x] Remove all eval-only function prototypes and wrappers from `orion/backend/orionclient/ffi.py`
- [x] Remove LT function wrappers
- [x] Remove `EvalKeyBundle*` wrappers
- [x] Keep all `Client*` wrappers (still used by `orion/client.py` until replaced in Task 6)

**Dead compiler code:**

- [x] Delete `TransformEncoder` class from `orion/core/compiler_backend.py`
- [x] Remove `Compiler._lt_evaluator` attribute and `lt_evaluator` property from `orion/compiler.py`
- [x] Remove `ctx.lt_evaluator` from `Compiler._build_context()`
- [x] Remove `test_galois.py::TestGaloisVsLattigo` class (equivalence proven; pure-Python BSGS unit tests remain)
- [x] Delete `experiments/07_galois_elements_python/` and `experiments/test_lt_removal.py`

**Verify:**

- [x] Rebuild shared library: `python tools/build_lattigo.py`
- [x] Run `pytest tests/` — all tests must pass (Client tests still work)
- [x] Run `pytest tests/test_galois.py` — pure-Python BSGS unit tests still pass

### Task 5: Restructure Go modules (ARCH.md 3.4)

Create root `go.mod`, move shared types and client logic out of `orionclient/`. Move bridge to `python/lattigo/bridge/`. Update all imports. Delete `orionclient/` last. `Client*` bridge exports must keep working (they import from the new locations).

**Inventory (do first):**

- [x] Identify all types evaluator imports from `orionclient/`: `Params`, `EvalKeyBundle`, `Client`, `Manifest`, `Ciphertext`, `Plaintext`, `NewCiphertext`, etc.
- [x] Identify all types bridge imports from `orionclient/` (~20+ references across `client.go`, `types.go`, `evaluator.go`, `main.go`)
- [x] Identify all types evaluator tests import from `orionclient/`

**Create root module + move code:**

- [x] Create root `go.mod` with `module github.com/baahl-nyu/orion`
- [x] Move shared types (`Params`, `Manifest`, `EvalKeyBundle`) to root-level package
- [x] Move `Ciphertext`/`Plaintext` Go types (multi-ct wrapper, marshal/unmarshal) to root-level or `ciphertext/` subpackage
- [x] Move client logic (`Client`, keygen, encrypt, decrypt, encode, decode) to `client/` subpackage
- [x] Merge `evaluator/go.mod` into root `go.mod` — delete `evaluator/go.mod` and its `replace` directive
- [x] Update evaluator imports: `orionclient.X` → `orion.X` / `orion/client.X` throughout `evaluator/*.go`

**Move bridge + update imports:**

- [x] Move bridge Go code from `orionclient/bridge/` to `python/lattigo/bridge/`
- [x] Update bridge `go.mod` to import from root module (`github.com/baahl-nyu/orion`)
- [x] Update all `orionclient.X` references in bridge files to new import paths (~20+ changes across `client.go`, `types.go`, `main.go`)
- [x] `Client*` bridge exports must still compile and work after import update

**Delete + verify:**

- [x] Delete `orionclient/` directory (all code moved out)
- [x] Run `go test ./evaluator/...` — all Go tests must pass
- [x] Run `go vet ./...` — no issues
- [x] Rebuild shared library, run `pytest tests/` — all Python tests must pass (Client tests still work via updated bridge)

### Task 6: Create python/lattigo/ package with new Lattigo-primitive API (ARCH.md 3.1)

Create standalone `lattigo` Python package. Exposes Lattigo primitives directly (KeyGenerator, Encoder, Encryptor, Decryptor) alongside the existing `Client*` bridge functions during transition.

- [x] Create directory structure: `python/lattigo/lattigo/`
- [x] Bridge already in `python/lattigo/bridge/` from Task 5
- [x] Add new bridge Go exports for Lattigo primitives: `NewKeyGenerator`, `GenSecretKey`, `GenRelinearizationKey`, `GenGaloisKey`, `NewEncoder`, `Encode`, `Decode`, `NewEncryptor`, `Encrypt`, `NewDecryptor`, `Decrypt`, `NewMemEvaluationKeySet`, `MarshalBinary`, `UnmarshalBinary` (for keys, ciphertexts, plaintexts)
- [x] Keep old `Client*` bridge exports temporarily (still used by old tests)
- [x] Move `GoHandle` class to `python/lattigo/lattigo/gohandle.py`
- [x] Write Python FFI bindings for new exports: `python/lattigo/lattigo/ckks.py`, `python/lattigo/lattigo/rlwe.py`
- [x] Keep old FFI wrappers accessible (in `python/lattigo/lattigo/ffi.py` or similar) for backward compat during transition
- [x] Create `python/lattigo/lattigo/__init__.py` — export `ckks`, `rlwe`
- [x] Create `python/lattigo/pyproject.toml` with build system
- [x] Verify: `cd python/lattigo && pip install -e .` succeeds and builds `.so`
- [x] Verify: `from lattigo import ckks, rlwe` works
- [x] Write tests for keygen, encode/decode roundtrip, encrypt/decrypt roundtrip using new Lattigo-primitive API
- [x] Write test for `MemEvaluationKeySet` marshal/unmarshal roundtrip

### Task 7: Migrate tests to Lattigo primitives

Replace all `Client` usage in tests with direct Lattigo primitive calls. After this task, `Client` class and old `Client*` bridge exports are no longer needed.

- [ ] Update `tests/test_v2_api.py`: replace `Client(params)` + `client.encode/encrypt/decrypt/decode` with `lattigo.ckks`/`lattigo.rlwe` calls
- [ ] Update `tests/test_gohandle.py`: replace all `Client` usage (skip or remove tests for deleted `Evaluator` class that are already skipped)
- [ ] Update any other test files using `Client`
- [ ] Remove `EvalKeys` usage from tests — use `MemEvaluationKeySet` marshal/unmarshal instead
- [ ] Run `pytest tests/` — all tests must pass using only Lattigo primitives
- [ ] Verify: `grep -r "Client(" tests/` returns nothing (except skipped test classes)

### Task 8: Extract python/orion-compiler/ package (ARCH.md 3.2)

Move compiler code to standalone package. `orion/client.py`, `orion/ciphertext.py` are NOT moved — they die when `orion/` is deleted.

- [ ] Create directory structure: `python/orion-compiler/orion_compiler/`
- [ ] Move `orion/compiler.py` → `orion_compiler/compiler.py`
- [ ] Move `orion/nn/` → `orion_compiler/nn/`
- [ ] Move `orion/core/` → `orion_compiler/core/`
- [ ] Move `orion/params.py` → `orion_compiler/params.py`
- [ ] Move `orion/compiled_model.py` → `orion_compiler/compiled_model.py` (remove `EvalKeys` class and `ORKEY` format during move — no longer needed after Task 7)
- [ ] Move `orion/models/` → `orion_compiler/models/` (if used)
- [ ] Update all imports: `from orion.xxx` → `from orion_compiler.xxx`, FFI imports → `from lattigo import ...`. Critical: `orion/core/compiler_backend.py` imports `from orion.backend.orionclient import ffi` — must change to `from lattigo import ...`
- [ ] Create `python/orion-compiler/orion_compiler/__init__.py` — export `Compiler`, `CKKSParams`, `CompiledModel`, `KeyManifest`
- [ ] Create `python/orion-compiler/pyproject.toml` — depends on `lattigo`, `torch`, `networkx`
- [ ] Verify: `cd python/orion-compiler && pip install -e .` succeeds
- [ ] Verify: `from orion_compiler import Compiler, CKKSParams, CompiledModel` works
- [ ] Run compiler tests with updated imports — all must pass

### Task 9: Create python/orion-evaluator/ package (ARCH.md 3.5)

New CGO bridge wrapping Go evaluator. Accepts raw Lattigo bytes.

**Prerequisite — refactor evaluator Go API:**

- [ ] Add `NewEvaluatorFromKeySet(params ckks.Parameters, keys *rlwe.MemEvaluationKeySet) (*Evaluator, error)` to `evaluator/evaluator.go` — constructs evaluator directly from Lattigo types instead of `EvalKeyBundle`. The existing `NewEvaluator(p Params, keys EvalKeyBundle)` can stay for backward compat or be refactored
- [ ] Run `go test ./evaluator/...` — verify existing tests still pass

**Bridge + Python package:**

- [ ] Create directory structure: `python/orion-evaluator/orion_evaluator/`, `python/orion-evaluator/bridge/`
- [ ] Write bridge Go code: CGO exports for `LoadModel`, `ModelClientParams`, `ModelClose`, `NewEvaluator` (accepts `MemEvaluationKeySet.MarshalBinary()` bytes — bridge unmarshals into `*rlwe.MemEvaluationKeySet`, calls `NewEvaluatorFromKeySet`), `EvaluatorForward` (accepts/returns `rlwe.Ciphertext.MarshalBinary()` bytes), `EvaluatorClose`
- [ ] Create bridge `go.mod` importing `github.com/baahl-nyu/orion/evaluator` and Lattigo
- [ ] Write Python FFI bindings: `python/orion-evaluator/orion_evaluator/ffi.py`
- [ ] Write `orion_evaluator/model.py` — `Model` class with `load()`, `client_params()`, `close()`
- [ ] Write `orion_evaluator/evaluator.py` — `Evaluator` class with `__init__(model, keys_bytes)`, `forward(ct_bytes) → bytes`, `close()`
- [ ] Create `__init__.py` and `pyproject.toml`
- [ ] Verify: `cd python/orion-evaluator && pip install -e .` succeeds and builds `.so`
- [ ] Write unit tests for Model load/close lifecycle
- [ ] Write unit tests for Evaluator creation/close lifecycle
- [ ] Write E2E test: compile (orion-compiler) → keygen+encrypt (lattigo) → forward (orion-evaluator) → decrypt (lattigo) → correct output

### Task 10: Final cleanup and acceptance verification

- [ ] Delete old `orion/` directory (includes `client.py`, `ciphertext.py` — not moved)
- [ ] Delete old `orion/backend/` directory
- [ ] Remove old `Client*` bridge exports from `python/lattigo/bridge/` (no longer needed — Task 7 migrated all tests)
- [ ] Remove old `Client*` FFI wrappers from `python/lattigo/lattigo/ffi.py`
- [ ] Update root `pyproject.toml` or remove
- [ ] Move tests to `python/tests/` or per-package test dirs
- [ ] Update all test imports to new package names
- [ ] Run full test suite — all must pass
- [ ] Verify no circular imports between packages
- [ ] Verify ARCH.md Phase 3 acceptance checklist (all items)
- [ ] Run `go vet ./...` — no issues
- [ ] Update CLAUDE.md with new project structure and build commands

## Technical Details

**Go module structure after restructure:**

```
go.mod                              # module github.com/baahl-nyu/orion
├── params.go, client/ ...          # Shared types + client logic
├── evaluator/                      # LoadModel, NewEvaluator, Forward
├── python/lattigo/bridge/          # Own go.mod, CGO → .so for Lattigo primitives
└── python/orion-evaluator/bridge/  # Own go.mod, CGO → .so for evaluator
```

**Python package dependency graph:**

```
lattigo (CGO bridge → Lattigo primitives)
  ↑
orion-compiler (+ torch, networkx)
  (no dependency between compiler and evaluator)
orion-evaluator (CGO bridge → Go evaluator)
```

**Serialization — Lattigo native, no custom formats:**

- Keys: `MemEvaluationKeySet.MarshalBinary()` / `UnmarshalBinary()`
- Bootstrap keys: `bootstrapping.EvaluationKeys.MarshalBinary()` / `UnmarshalBinary()` (separate blob)
- Ciphertexts: `rlwe.Ciphertext.MarshalBinary()` / `UnmarshalBinary()`
- Models: `.orion` v2 format (unchanged, Go parser in `evaluator/format.go`)

**Deleted (not moved to new packages — dies with `orion/` directory):**

- `Client` class (`orion/client.py`)
- `Ciphertext`/`PlainText` Python wrapper classes (`orion/ciphertext.py`)
- `EvalKeys` class, `ORKEY` container format
- Python ciphertext shape header (redundant with ORTXT)
- `orionclient/` Go package
- `TransformEncoder`, `LinearTransform.compile()`, `LinearTransform.evaluate_transforms()`
- All `Eval*` bridge exports (dead since Phase 1)
- `Client*` bridge exports (replaced by Lattigo primitive exports in Task 6)

## Post-Completion

**Manual verification:**

- Install each package independently in fresh venvs to verify isolation
- Test threshold encryption scenario using Lattigo directly → evaluator
- Verify bridge `.so` sizes decreased

**Documentation updates:**

- Update README.md with new installation instructions and examples
- Update ARCH.md Phase 3 status
- Update CLAUDE.md build/test commands
