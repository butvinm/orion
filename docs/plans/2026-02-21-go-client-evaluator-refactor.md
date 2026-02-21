# Go Client & Evaluator Library Refactor

## Overview

Extract an instance-based Go library (`orionclient`) from the existing global-state Go backend. Two core types — `Client` (keygen, encrypt, decrypt) and `Evaluator` (FHE operations) — each holding their own state. No global singletons.

Python and WASM become thin wrappers. The 864 lines of Python wrappers (`backend/python/`) and the duplicated WASM crypto module (`demo/wasm-fhe-demo/wasm/crypto.go`) are replaced by bridge layers calling `orionclient`.

## Target Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Go Package: orionclient                                   │
│                                                            │
│  Client               Evaluator           Types            │
│  ─ New()              ─ New()             ─ Params         │
│  ─ FromSecretKey()    ─ Add/Sub/Mul()     ─ Plaintext      │
│  ─ Close()            ─ Rotate()          ─ Ciphertext     │
│  ─ Encode/Decode()    ─ Rescale()           ─ Marshal()    │
│  ─ Encrypt/Decrypt()  ─ Bootstrap()         ─ Unmarshal()  │
│  ─ GenRLK()           ─ EvalPoly()        ─ EvalKeyBundle  │
│  ─ GenGaloisKey()     ─ LinearTransform() ─ Manifest       │
│  ─ GenBootstrapKeys() ─ LoadLT()          ─ LinearTransform│
│                       ─ Close()                            │
│                                                            │
│  Instance-based. No global state. Multiple coexist.        │
└──────────┬────────────────────────┬────────────────────────┘
           │                        │
    ┌──────┴──────┐          ┌──────┴───────┐
    │ WASM bridge │          │  FFI bridge  │
    │ bindings_js │          │  cgo.Handle  │
    │ plaintextIDs│          │  C exports   │
    └──────┬──────┘          └──────┬───────┘
           │                        │
    ┌──────┴──────┐          ┌──────┴────────┐
    │ JS wrapper  │          │ Python wrapper│
    └─────────────┘          └───────────────┘
```

## Serialization Ownership

```
Go orionclient owns:         Python compiler owns:
─ Ciphertext wire format     ─ CompiledModel (params + module metadata
─ EvalKeys                      + topology + LT/bias blob container)
─ LinearTransform blobs      ─ Module graph (forward() connections)
─ Secret key

CompiledModel (Python) ──decomposed by Python──→ Go Evaluator:
  ├── params ──────────────→ NewEvaluator(params, keys)
  ├── manifest ────────────→ Client.GenerateKeys(manifest)
  ├── LT blobs ────────────→ Evaluator.LoadLinearTransform(blob)
  ├── bias values ─────────→ Evaluator.Encode(values, level)
  └── module_metadata ─────→ stays in Python (module reconstruction)
      topology ────────────→ stays in Python (forward() graph)
```

## Design Decisions

- **Instance-based, no global state.** Multiple `Client` and `Evaluator` instances coexist in one process.
- **Evaluator constructor handles key loading internally.** No exposed `LoadRelinKey → GenerateEvaluationKeys → LoadRotationKey` ordering. Pass `EvalKeyBundle`, get an evaluator.
- **Crypto serialization owned by Go.** `Ciphertext.Marshal()`, `EvalKeys`, `LinearTransform` blobs. Model serialization stays in Python (compiler artifact, PyTorch-specific metadata).
- **LTs loaded incrementally.** `Evaluator.LoadLinearTransform(blob) → *LinearTransform`. Python extracts blobs from `CompiledModel` during module reconstruction and loads them one at a time. Go Evaluator knows nothing about `CompiledModel`.
- **Individual ops on Evaluator.** `Add`, `Mul`, `Rotate`, `Rescale`, `Bootstrap`, `EvalPoly`, `LinearTransform`. Matches current nn module call pattern.
- **`cgo.Handle` for FFI.** Go objects cross to Python as opaque `uintptr_t`. `cgo.NewHandle` / `.Value()` / `.Delete()`.
- **Unified `Ciphertext` type.** One type for both transport and computation. Backed by `cgo.Handle` pointing to Go `*Ciphertext`. Serialize (`Marshal`) only at I/O boundaries (client→server, server→client). Current `CipherText` and `CipherTensor` merge into one — same data (ciphertext IDs + shape), no reason for two types. Arithmetic ops (`add`, `sub`, `mul`, `rotate`, `bootstrap`) and metadata queries (`level`, `scale`, `slots`) live on this single type. Python wrapper holds the handle; `__del__` calls `DeleteHandle`.
- **Explicit `Close()`.** Python wrapper provides context manager.
- **Error propagation.** Bridge functions return `(result, errMsg)` pairs. `errMsg` is a C string (NULL on success). Python wrapper converts non-NULL to exceptions. Go panics caught with `recover()`.
- **Go module location.** `orionclient/` at repo root with its own `go.mod`. Bridge at `orionclient/bridge/` (same module, CGo build tag). WASM demo imports via `replace` directive.
- **Phased migration.** Old backend stays until phase 6. Both coexist during phases 1–5.

## Phase 1: Go Client Library — Types and Client

Create `orionclient/` Go package with `Client`, `Params`, `Plaintext`, `Ciphertext`, `EvalKeyBundle`, `Manifest`.

```
orionclient/
├── params.go, client.go, ciphertext.go, plaintext.go, keys.go
├── client_test.go, ciphertext_test.go
├── go.mod, go.sum
```

### Tasks

- [x] `params.go`: `Params` struct, `NewCKKSParameters() (ckks.Parameters, error)`, `MaxSlots()`, `DefaultScale()`
- [x] `plaintext.go`: `Plaintext` wrapping `*rlwe.Plaintext` + shape
- [x] `ciphertext.go`: `Ciphertext` wrapping `[]*rlwe.Ciphertext` + shape. `Marshal()`/`UnmarshalCiphertext()` with magic `ORTXT\x00\x01\x00`, CRC32. Metadata queries: `Level()`, `Scale()`, `Slots()`, `Degree()`, `Shape()`
- [x] `keys.go`: `EvalKeyBundle` (RLK, Galois map, Bootstrap map, BootLogP), `Manifest`
- [x] `client.go`: `Client` struct with `New`, `FromSecretKey`, `Close`, `SecretKey`, `Encode`, `Encrypt`, `Decrypt`, `Decode`, `GenerateRLK`, `GenerateGaloisKey`, `GenerateBootstrapKeys`, `MaxSlots`, `DefaultScale`
- [x] `client_test.go`: round-trip, key generation, multiple instances coexisting
- [x] `ciphertext_test.go`: wire format round-trip, test vectors

## Phase 2: Go Evaluator

Add `Evaluator` to `orionclient/`. Loads keys in constructor, loads LTs incrementally, exposes individual FHE ops. All ops take and return `*Ciphertext` (the unified type).

```
orionclient/
├── evaluator.go, lineartransform.go, polynomial.go, bootstrapper.go
├── evaluator_test.go
```

### Tasks

- [x] `evaluator.go`: `NewEvaluator(p Params, keys EvalKeyBundle) (*Evaluator, error)`. Methods: `Close`, `Encode`, `Add`, `Sub`, `Mul`, `AddPlaintext`, `SubPlaintext`, `MulPlaintext`, `AddScalar`, `MulScalar`, `Negate`, `Rotate`, `Rescale`, `Bootstrap`. All take/return `*Ciphertext`.
- [x] `lineartransform.go`: `LoadLinearTransform(blob []byte) (*LinearTransform, error)` for loading pre-compiled LTs from `CompiledModel` blobs. `GenerateLinearTransform(...)` for compile-time generation. `EvalLinearTransform(ct, lt)`. `(*LinearTransform).Marshal()`/`UnmarshalLinearTransform()`. `RequiredGaloisElements()`
- [x] `polynomial.go`: `Polynomial` type, `GenerateMonomial`, `GenerateChebyshev`, `EvalPoly`
- [x] `bootstrapper.go`: per-slot-count bootstrap evaluators, integrated into `Evaluator.Bootstrap()`
- [x] `evaluator_test.go`: key loading, LT load from blob, arithmetic round-trips, polynomial eval, linear transform

## Phase 3: FFI Bridge

C-export layer wrapping `orionclient` via `cgo.Handle`. Replaces current 47 flat C exports.

```
orionclient/bridge/
├── client.go, evaluator.go, types.go, main.go
```

### Error handling pattern

```go
//export NewClient
func NewClient(paramsJSON *C.char, errOut **C.char) C.uintptr_t {
    c, err := orionclient.New(parseParams(C.GoString(paramsJSON)))
    if err != nil {
        *errOut = C.CString(err.Error())
        return 0
    }
    return C.uintptr_t(cgo.NewHandle(c))
}

//export DeleteHandle
func DeleteHandle(h C.uintptr_t) { cgo.Handle(h).Delete() }
```

### Tasks

- [x] `bridge/client.go`: `NewClient`, `ClientEncode`, `ClientEncrypt`, `ClientDecrypt`, `ClientDecode → float64[]`, `ClientGenerateRLK → bytes`, etc. `ClientEncrypt` returns a `cgo.Handle` to `*Ciphertext` (not bytes). All with `errOut **C.char`.
- [x] `bridge/evaluator.go`: `NewEvaluator`, `EvalAdd`, `EvalRotate`, `EvalLoadLinearTransform`, etc. All ciphertext args and returns are `cgo.Handle`s to the same `*orionclient.Ciphertext` type.
- [x] `bridge/types.go`: `CiphertextMarshal(ctH) → bytes`, `CiphertextUnmarshal(bytes) → uintptr_t`, `CiphertextLevel(ctH) → int`, `CiphertextScale(ctH) → uint64`, etc.
- [x] `bridge/main.go`: CGO entry point, `DeleteHandle`, `FreeCArray`.
- [x] Build shared library, verify on Linux/macOS.

## Phase 4: Python Wrapper Migration

Replace `backend/python/` and rewrite `client.py`/`evaluator.py` as thin FFI wrappers. Merge `CipherText` and `CipherTensor` into a single `Ciphertext` class.

### Tasks

- [ ] `orion/backend/orionclient_ffi.py`: ctypes bindings for the new bridge. Checks `errOut` after every call, raises `RuntimeError` if non-NULL. Replaces `backend/lattigo/bindings.py`.
- [ ] Rewrite `orion/client.py`: single FFI call per method, `PlainText` wraps handle, context manager. `Ciphertext` is the unified type (see below).
- [ ] Create unified `orion/ciphertext.py`: single `Ciphertext` class wrapping a `cgo.Handle`. Replaces both `CipherText` (from `client.py`) and `CipherTensor` (from `backend/python/tensors.py`).
  - Transport: `to_bytes()` → calls `CiphertextMarshal`, `from_bytes()` → calls `CiphertextUnmarshal`
  - Metadata: `level()`, `scale()`, `set_scale()`, `slots()`, `degree()`, `shape`
  - Arithmetic (delegated to evaluator handle): `add`, `sub`, `mul`, `__neg__`, `roll`, `bootstrap`. These require an evaluator reference, set when entering evaluation scope.
  - Lifecycle: `__del__` calls `DeleteHandle`
- [ ] Rewrite `orion/evaluator.py`: `NewEvaluator` FFI with key bundle. Module reconstruction extracts LT blobs from `CompiledModel`, calls `EvalLoadLinearTransform` per blob. `evaluator.run(ct)` passes the handle directly — no conversion between types. Context manager.
- [ ] Update `orion/nn/` modules: context collapses from 5 objects (`evaluator`, `lt_evaluator`, `poly_evaluator`, `bootstrapper`, `encoder`) to one evaluator handle. Modules receive and return `Ciphertext` (the unified type). Affected: `activation.py`, `linear.py`, `normalization.py`, `operations.py`.
- [ ] Delete `orion/backend/python/tensors.py` (CipherTensor/PlainTensor no longer needed).
- [ ] Full test suite passes (`pytest tests/`).

## Phase 5: WASM Migration

Port WASM demo to import `orionclient` instead of standalone `crypto.go`.

```
demo/wasm-fhe-demo/wasm/
├── go.mod          (replace directive → ../../orionclient)
├── bindings_js.go  (wraps orionclient.Client)
├── main.go
```

### Tasks

- [ ] Update `wasm/go.mod`: add `require` + `replace` for `orionclient`.
- [ ] Rewrite `wasm/bindings_js.go`: wrap `orionclient.Client` methods. Keep `plaintextStore` here (JS can't hold Go pointers).
- [ ] Delete `wasm/crypto.go` (replaced by `orionclient` import).
- [ ] Verify WASM builds, Go tests, browser test.

## Phase 6: Cleanup

### Tasks

- [ ] Delete `orion/backend/python/` (9 files, 864 lines)
- [ ] Delete `orion/backend/lattigo/*.go` (12 files)
- [ ] Delete `orion/backend/lattigo/bindings.py`
- [ ] Update imports, CLAUDE.md

## Phase 7: Validation, Verification and Acceptance

### Tasks

- [ ] Full Go test suite passes (`go test ./...` in `orionclient/`)
- [ ] Full Python test suite passes (`pytest tests/`)
- [ ] WASM demo builds and runs end-to-end (compile model → generate keys → encrypt → infer → decrypt)
- [ ] No global state in Go: instantiate two `Client`s with different params simultaneously, verify independence
- [ ] No memory leaks: run inference loop, verify Go heap doesn't grow unboundedly (use `runtime.ReadMemStats`)
- [ ] Secret key containment: `Client.Close()` zeroes secret key memory, verify with test
- [ ] Wire format compatibility: `Ciphertext.Marshal()` output from Go matches expected test vectors, `from_bytes()` in JS (WASM) can consume it
- [ ] Error propagation: Go errors surface as Python exceptions with message, not crashes
- [ ] No deleted code is still imported anywhere (`grep` for old module paths)
- [ ] CLAUDE.md architecture section updated to reflect new structure

## Out of Scope

- Compiler refactoring (stays in Python)
- ONNX export (future: replaces `CompiledModel` with standard graph format)
- Wire format version negotiation protocol
- Performance optimization of Go operations
