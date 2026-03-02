# Phase 4: JS/WASM Lattigo Bindings, Examples, and Browser Demo

## Overview

Build JS/WASM bindings for Lattigo's CKKS operations, enabling browser-side key generation, encryption, and decryption. This lets web clients keep secret keys local while sending encrypted data to a Go server running the Orion evaluator.

Three deliverables:

1. **`js/lattigo/`** — Go WASM bridge + TypeScript wrappers (npm package)
2. **`js/examples/`** — Node.js roundtrip examples validating the bindings
3. **`examples/wasm-demo/`** — Browser demo: Go HTTP server + HTML/JS client, end-to-end encrypted inference

## Context

- **Proven approach**: Two WASM experiments exist in `experiments/` — basic keygen roundtrip (1.7KB WASM) and full bootstrap test (7.6MB WASM). Both use `syscall/js` with `GOOS=js GOARCH=wasm`.
- **Python bridge as template**: `python/lattigo/bridge/lattigo.go` exposes 36 CGO functions. The WASM bridge mirrors these but uses `syscall/js` instead of CGO, and a Go-side handle map instead of `cgo.Handle`. The WASM bridge also adds bootstrap primitives (not in Python bridge).
- **Binary size**: Lattigo compiles to ~7.6MB uncompressed WASM (~3MB gzipped). Target: < 10MB uncompressed.
- **Key constraint**: WASM bridge exposes client-side operations only (keygen, encode, encrypt, decrypt, serialize). No evaluator in WASM — evaluation stays server-side.

### Files/components involved

| Component                  | Path                                        | Purpose                                           |
| -------------------------- | ------------------------------------------- | ------------------------------------------------- |
| Python bridge (template)   | `python/lattigo/bridge/lattigo.go`          | 36 CGO functions to mirror                        |
| Python wrappers (template) | `python/lattigo/lattigo/ckks.py`, `rlwe.py` | TypeScript API shape reference                    |
| WASM experiment 1          | `experiments/lattigo-wasm-test/`            | Sync pattern: `js.FuncOf` + `js.Global().Set()`   |
| WASM experiment 2          | `experiments/lattigo-wasm-bootstrap/`       | Async pattern: Promise + goroutine                |
| Go module                  | `go.mod`                                    | `github.com/baahl-nyu/lattigo/v6 v6.2.0`, Go 1.23 |
| Build script (template)    | `tools/build_lattigo.py`                    | CGO build — adapt for WASM target                 |

### Existing patterns

- **Handle management (Python)**: `cgo.Handle` wrapped by `GoHandle` RAII class. WASM equivalent: Go-side `map[uint32]any` + JS-side TypeScript classes with `.free()`.
- **Error propagation (Python)**: `errOut` double-pointer C strings. WASM equivalent: return `{handle, error}` objects or throw JS exceptions.
- **Async ops (experiments)**: Bootstrap key generation returns `Promise` via goroutine + `resolve`/`reject`. All other ops (keygen, encode, encrypt) are synchronous.

## Development Approach

- **Testing approach**: Regular (code first, then tests)
- Complete each task fully before moving to the next
- Make small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes in that task
- **CRITICAL: all tests must pass before starting next task**
- **CRITICAL: update this plan file when scope changes during implementation**
- Run tests after each change

## Testing Strategy

- **Go bridge tests**: All Go tests run via WASM+Node.js (`GOOS=js GOARCH=wasm go test` with `node` as test runner, set `GOWASMRUNTIME=node`). Pure Go logic (handle map, helpers) cannot be tested with regular `go test` because the bridge package imports `syscall/js`.
- **Node.js integration tests (Tasks 2–5)**: Lightweight raw JS scripts (like the existing experiments) that load WASM and exercise bridge functions directly. These validate the Go bridge before TypeScript wrappers exist. Superseded by Task 8's formal tests.
- **TypeScript integration tests (Task 8+)**: Formal test suite using vitest, exercising the TypeScript API. These are the permanent tests.
- **Browser demo**: Manual E2E test (compile model in Python → serve with Go → query from browser → verify decrypted result)

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix
- Update plan if implementation deviates from original scope

## Implementation Steps

### Task 1: Go WASM bridge — handle map and lifecycle

Set up the bridge scaffolding: module structure, handle management, and the two foundational functions (create params, delete handle).

- [x] Create `js/lattigo/bridge/go.mod` importing `github.com/baahl-nyu/lattigo/v6` only. No root module dependency — the bridge is a thin binding to Lattigo with zero Orion-specific logic.
- [x] Create `js/lattigo/bridge/handles.go` — handle map (`map[uint32]any`, counter, `Store(obj) → id`, `Load(id) → obj`, `Delete(id)`). No mutex needed — Go WASM is single-threaded (goroutines are cooperative). Keep it simple.
- [x] Create `js/lattigo/bridge/main.go` — `//go:build js && wasm` tag, `func main()` with `select{}` keepalive, register `deleteHandle` under `globalThis.lattigo` namespace, set `globalThis.lattigo.__ready = true` as LAST registration (readiness signal for JS loader)
- [x] Create `js/lattigo/bridge/helpers.go` — JS↔Go conversion helpers: `jsToFloat64Slice(js.Value) → []float64`, `float64SliceToJS([]float64) → js.Value`, `bytesToJS([]byte) → js.Value (Uint8Array)`, `jsToBytes(js.Value) → []byte`
- [x] Create `js/lattigo/bridge/build.sh` — `GOOS=js GOARCH=wasm go build -o ../wasm/lattigo.wasm .`
- [x] Copy `wasm_exec.js` from Go distribution (`$(go env GOROOT)/misc/wasm/wasm_exec.js`) into `js/lattigo/wasm/`
- [x] Build WASM, verify it compiles and binary size < 10MB
- [x] Write Go tests for handle map (`handles_test.go` — store, load, delete, double-delete). Run via `GOOS=js GOARCH=wasm go test` with `GOWASMRUNTIME=node`.
- [x] Run tests — must pass before next task

### Task 2: Go WASM bridge — CKKS parameters and key generation

Port parameter construction and key generation from `python/lattigo/bridge/lattigo.go`.

- [x] Add `params.go` — `newCKKSParams(paramsJSON) → handleID` (mirrors `NewCKKSParams` in Python bridge). Parse JSON with `LogN`, `LogQ`, `LogP`, `LogDefaultScale`, `H`, `RingType`.
- [x] Add param accessors: `ckksMaxSlots(hID)`, `ckksMaxLevel(hID)`, `ckksDefaultScale(hID)`, `ckksGaloisElement(hID, rotation)`, `ckksModuliChain(hID)`, `ckksAuxModuliChain(hID)`
- [x] Add `keygen.go` — `newKeyGenerator(paramsHID) → hID`, `keyGenGenSecretKey(kgHID) → hID`, `keyGenGenPublicKey(kgHID, skHID) → hID`, `keyGenGenRelinKey(kgHID, skHID) → hID`, `keyGenGenGaloisKey(kgHID, skHID, galoisElement) → hID`
- [x] Register all functions in `main.go` via `js.Global().Get("lattigo").Set(name, js.FuncOf(...))` — namespace under `globalThis.lattigo.*`
- [x] Create minimal Node.js test: load WASM → create params → keygen → verify handles are valid numbers
- [x] Run tests — must pass before next task

### Task 3: Go WASM bridge — encoder, encryptor, decryptor

Port encoding/encryption/decryption operations.

- [x] Add `encoder.go` — `newEncoder(paramsHID) → hID`, `encoderEncode(encHID, values[], level, scale) → ptHID` (sync — encoding is <1ms even for 2^14 slots), `encoderDecode(encHID, ptHID, numSlots) → float64[]` (sync)
- [x] Add `encryptor.go` — `newEncryptor(paramsHID, pkHID) → hID`, `encryptorEncryptNew(encHID, ptHID) → ctHID`
- [x] Add `decryptor.go` — `newDecryptor(paramsHID, skHID) → hID`, `decryptorDecryptNew(decHID, ctHID) → ptHID`
- [x] Register all in `main.go`
- [x] Node.js test: params → keygen → encode → encrypt → decrypt → decode → verify values match (roundtrip)
- [x] Run tests — must pass before next task

### Task 4: Go WASM bridge — serialization (marshal/unmarshal)

Port serialization for all key and ciphertext types. Critical for client↔server communication.

- [x] Add `serialize.go` — marshal/unmarshal for: `SecretKey`, `PublicKey`, `RelinearizationKey`, `GaloisKey`, `Ciphertext`, `Plaintext`, `MemEvaluationKeySet`
- [x] Each marshal returns `Uint8Array`, each unmarshal accepts `Uint8Array` and returns handle ID
- [x] Add `MemEvaluationKeySet`: `newMemEvalKeySet(rlkHID, galoisKeyHIDs[]) → hID`, `memEvalKeySetMarshal(hID) → bytes`, `memEvalKeySetUnmarshal(bytes) → hID`
- [x] Add ciphertext/plaintext accessors: `ciphertextLevel(ctHID) → int`, `plaintextLevel(ptHID) → int`
- [x] Node.js test: keygen → marshal all keys → unmarshal → verify roundtrip (marshal again, compare bytes)
- [x] Node.js test: encrypt → marshal ciphertext → unmarshal → decrypt → verify values
- [x] Run tests — must pass before next task

### Task 5: Go WASM bridge — bootstrap parameter primitives

Expose Lattigo's bootstrap API as thin bindings. No orchestration logic — the user (or example code) decides how to construct bootstrap parameters and generate keys.

**Note:** This is NEW code, not a port — the Python bridge has NO bootstrap functions. Reference implementation is `experiments/lattigo-wasm-bootstrap/main.go` which uses `bootstrapping.NewParametersFromLiteral()` + `btpParams.GenEvaluationKeys(sk)`.

- [x] Add `bootstrap.go` with thin bindings to `circuits/ckks/bootstrapping`:
  - `newBootstrapParametersFromLiteral(paramsHID, btpLitJSON) → btpParamsHID` — accepts `bootstrapping.ParametersLiteral` as JSON (user constructs it with `LogN`, `LogP`, `Xs`, `LogSlots`). Sync — just parameter construction.
  - `btpParamsGenEvaluationKeys(btpParamsHID, skHID) → Promise<{evkHID, btpEvkHID}>` — async (goroutine+resolve/reject). Returns both the `*rlwe.MemEvaluationKeySet` and `bootstrapping.EvaluationKeys` handles. Heavy operation (5–30s).
- [x] Register in `main.go`
- [x] Node.js test: construct bootstrap params literal JSON (with ConjugateInvariant LogN+1 adjustment done in JS), generate keys, verify handles are valid
- [x] Node.js test: marshal the resulting `MemEvaluationKeySet` → unmarshal → verify roundtrip
- [x] Run tests — must pass before next task

### Task 6: TypeScript wrappers — project setup and types

Set up the npm package structure and TypeScript foundation.

- [x] Create `js/lattigo/package.json` — name `@orion/lattigo`, entry point `dist/index.js`, types `dist/index.d.ts`, scripts for build/test
- [x] Create `js/lattigo/tsconfig.json` — target ES2020, strict mode, declaration output
- [x] Create `js/lattigo/src/types.ts` — `Handle` type (number), `WasmBridge` interface (typed function signatures for all `globalThis.lattigo.*` functions)
- [x] Create `js/lattigo/src/loader.ts` — `loadLattigo(wasmPath?: string): Promise<WasmBridge>` — loads WASM binary, instantiates Go runtime via `wasm_exec.js`'s `Go` class, calls `go.run(instance)`, polls for `globalThis.lattigo.__ready === true` (set by Go's `main()` as last registration), returns typed bridge object. Works in both Node.js (reads `.wasm` file from disk) and browser (fetches URL).
- [x] `npm install` dev deps: `typescript`, `esbuild`
- [x] Verify `npx tsc --noEmit` passes
- [x] Run tests — must pass before next task

### Task 7: TypeScript wrappers — core classes

Wrap the raw WASM functions in ergonomic TypeScript classes. All classes receive the `WasmBridge` instance via a module-level variable set by `loadLattigo()`. Classes are only usable after `loadLattigo()` resolves — calling methods before that throws.

- [x] Create `js/lattigo/src/ckks.ts` — `CKKSParameters` class: `static fromLogn(...)`, `static fromJSON(paramsJson)`, `maxSlots()`, `maxLevel()`, `defaultScale()`, `galoisElement(rotation)`, `moduliChain()`, `auxModuliChain()`, `free()`
- [x] Create `js/lattigo/src/rlwe.ts` — `KeyGenerator`, `SecretKey`, `PublicKey`, `RelinearizationKey`, `GaloisKey`, `Encryptor`, `Decryptor`, `Ciphertext`, `Plaintext`, `MemEvaluationKeySet` — each wrapping handle + calling bridge functions + `.free()` → `deleteHandle` + `FinalizationRegistry` safety net (catches forgotten `.free()` calls)
- [x] Create `js/lattigo/src/encoder.ts` — `Encoder` class: `static new(params)`, `encode(values, level, scale)`, `decode(plaintext, numSlots)`, `free()`
- [x] Create `js/lattigo/src/index.ts` — re-export `loadLattigo` + all public classes
- [x] Add `esbuild` build script in `package.json` — bundle to `dist/`
- [x] Verify build produces `dist/index.js` + `dist/index.d.ts`
- [x] Run tests — must pass before next task

### Task 8: TypeScript integration tests

Comprehensive Node.js tests exercising the TypeScript API.

- [x] Set up test infrastructure: Node.js test runner (or vitest), WASM loader for test environment
- [x] Test: `CKKSParameters.fromLogn()` → verify `maxSlots()`, `maxLevel()`, `defaultScale()` return correct values
- [x] Test: full roundtrip — `KeyGenerator` → `genSecretKey()` → `genPublicKey()` → `Encoder.encode()` → `Encryptor.encryptNew()` → `Decryptor.decryptNew()` → `Encoder.decode()` → compare values within CKKS tolerance
- [x] Test: serialization roundtrip — marshal all key types → unmarshal → marshal again → bytes match
- [x] Test: `MemEvaluationKeySet` — create from RLK + Galois keys → marshal → unmarshal
- [x] Test: memory cleanup — create objects, call `.free()`, verify no errors on double-free (should be no-op)
- [x] Test: cross-platform serialization — encrypt in WASM, marshal ciphertext bytes, decrypt via Python bridge (or vice versa) to verify byte-level compatibility. If Python test infra is too heavy, at minimum verify WASM-marshaled bytes can be unmarshaled by Go evaluator via a small Go test.
- [x] Test: error handling — invalid params JSON, wrong handle IDs, type mismatches
- [x] Run tests — must pass before next task

### Task 9: JS examples — Node.js roundtrip

Create self-contained example scripts in `js/examples/`.

- [x] Create `js/examples/node/package.json` — depends on `@orion/lattigo` (local path)
- [x] Create `js/examples/node/roundtrip.ts` — full keygen → encode → encrypt → decrypt → decode flow with console output showing values and timing
- [x] Create `js/examples/node/eval-keys.ts` — reference implementation showing full manual key generation from a `KeyManifest` JSON: parse manifest, generate RLK (if `needs_rlk`), generate each Galois key (from `galois_elements`), construct bootstrap `ParametersLiteral` (handling ConjugateInvariant `LogN+1`, `Xs` propagation, `LogSlots` from `bootstrap_slots`), call `btpParamsGenEvaluationKeys`, assemble final `MemEvaluationKeySet`. Print timing for each step and total key sizes. This is the example users copy from.
- [x] Run `npx tsx js/examples/node/roundtrip.ts` — verify it prints decoded values matching input within CKKS tolerance
- [x] Run `npx tsx js/examples/node/eval-keys.ts` — verify it prints key sizes and completes without error
- [x] Run tests — must pass before next task

### Task 10: Browser demo — Go HTTP server

Build the server side of the end-to-end demo.

**Prerequisite:** A compiled `.orion` model file is needed. Generate one using the Python compiler (e.g., compile MLP with random weights — accuracy doesn't matter, pipeline correctness does). Document the generation command in the demo README.

- [x] Generate a demo `.orion` model file using the Python compiler (MLP on MNIST, random weights OK). Save as `examples/wasm-demo/model.orion`. Document the command to regenerate it.
- [x] Create `examples/wasm-demo/server/main.go` — HTTP server that:
  - `GET /params` — loads `.orion` model file, returns `model.ClientParams()` JSON (CKKS params, key manifest, input level)
  - `POST /session` — accepts `MemEvaluationKeySet` bytes, creates evaluator, returns session ID
  - `POST /session/{id}/infer` — accepts ciphertext bytes, runs `evaluator.Forward()`, returns result ciphertext bytes
  - Serves static files from `examples/wasm-demo/client/`
- [x] Create `examples/wasm-demo/server/go.mod` — imports `github.com/baahl-nyu/orion/evaluator` with `replace github.com/baahl-nyu/orion => ../../../` (3 levels up: `server/` → `wasm-demo/` → `examples/` → root)
- [x] Write Go tests for server handlers (mock model, test request/response format)
- [x] Run tests — must pass before next task

### Task 11: Browser demo — HTML/JS client

Build the browser client demonstrating end-to-end encrypted inference.

- [x] Create `examples/wasm-demo/client/index.html` — minimal UI: input field (comma-separated floats), "Run Inference" button, output area, status/timing display
- [x] Create `examples/wasm-demo/client/client.ts` — client logic (full manual Lattigo usage, no convenience wrappers):
  1. Fetch `/params` → parse CKKS params JSON + key manifest JSON + input level
  2. Initialize WASM Lattigo → create `CKKSParameters` from JSON
  3. Generate SK, PK via `KeyGenerator`
  4. Generate RLK if manifest `needs_rlk`
  5. Generate each Galois key from manifest `galois_elements` — show progress (N/total)
  6. If manifest `bootstrap_slots` non-empty: construct `ParametersLiteral` JSON (handle ConjugateInvariant `LogN+1`, propagate `Xs`, compute `LogSlots`), call `btpParamsGenEvaluationKeys` — show progress
  7. Assemble `MemEvaluationKeySet`, marshal, POST to `/session` → get session ID
  8. Encode + encrypt user input → POST to `/session/{id}/infer`
  9. Receive result ciphertext → decrypt → decode → display with timing breakdown
- [x] Bundle with esbuild for browser
- [x] Manual test: compile MLP with Python → start Go server → open browser → run inference → verify result
- [x] Run tests — must pass before next task

### Task 12: Build tooling and documentation

Polish step — replaces Task 1's `build.sh` with a proper Python build script matching the existing `tools/build_lattigo.py` pattern.

- [ ] Create `tools/build_lattigo_wasm.py` (or extend `build_lattigo.py`) — builds `js/lattigo/bridge/` to WASM, copies `wasm_exec.js`, reports binary size
- [ ] Add npm scripts: `build:wasm` (Go WASM build), `build:ts` (esbuild), `build` (both), `test`
- [ ] Verify clean build from scratch: `tools/build_lattigo_wasm.py && cd js/lattigo && npm install && npm run build && npm test`

### Task 13: Verify acceptance criteria

- [ ] `js/lattigo/` builds to `.wasm` (< 10 MB uncompressed)
- [ ] TypeScript wrappers compile without errors
- [ ] Full key generation flow (SK → PK → RLK → Galois keys → bootstrap keys) works in Node.js
- [ ] JS example: keygen → encode → encrypt → decrypt → decode roundtrip works in Node.js
- [ ] Browser demo: compile MLP (Python) → serve (Go) → query (browser) → correct decrypted result
- [ ] WASM loads and initializes in < 3 seconds on modern browser
- [ ] No Go objects leaked after `.free()` calls
- [ ] Run full test suite
- [ ] Run linter — all issues must be fixed

### Task 14: [Final] Update documentation

- [ ] Update `CLAUDE.md` if new build commands or conventions emerged
- [ ] Update `ARCH.md` Phase 4 checklist items to `[x]`

## Technical Details

### Handle map design (Go side)

```go
// handles.go — no mutex needed, Go WASM is single-threaded
var (
    handleMap  = make(map[uint32]any)
    nextHandle uint32 = 1
)

func storeHandle(obj any) uint32 { ... }
func loadHandle(id uint32) (any, bool) { ... }
func deleteHandle(id uint32) { ... }  // no-op if id not found (idempotent)
```

All WASM-exported functions accept/return `uint32` handle IDs. TypeScript classes wrap these IDs and call `deleteHandle` on `.free()`.

### TypeScript wrapper lifecycle (`.free()` + FinalizationRegistry safety net)

```typescript
// Shared registry — one per module
const registry = new FinalizationRegistry((handleId: number) => {
  bridge.deleteHandle(handleId); // catch forgotten .free()
});

class SecretKey {
  private _handle: number;
  constructor(handle: number) {
    this._handle = handle;
    registry.register(this, handle, this); // safety net
  }
  free(): void {
    if (this._handle !== 0) {
      registry.unregister(this); // cancel safety net
      bridge.deleteHandle(this._handle);
      this._handle = 0;
    }
  }
}
```

Primary cleanup: explicit `.free()`. Safety net: `FinalizationRegistry` catches forgotten calls. `.free()` unregisters from the registry to avoid double-delete. This mirrors wasm-bindgen's proven pattern.

### JS function registration namespace

All Go functions registered under `globalThis.lattigo`:

```go
ns := js.Global().Get("Object").New()
ns.Set("newCKKSParams", js.FuncOf(newCKKSParams))
ns.Set("deleteHandle", js.FuncOf(deleteHandle))
// ... all other functions ...
ns.Set("__ready", true)  // MUST be last — signals JS loader that all functions are registered
js.Global().Set("lattigo", ns)
```

TypeScript `WasmBridge` interface types all these functions. The `loadLattigo()` loader polls `globalThis.lattigo?.__ready` to detect when Go has finished registering.

### Async vs sync boundary

| Operation                     | Pattern         | Reason                              |
| ----------------------------- | --------------- | ----------------------------------- |
| Create params                 | Sync            | Fast, < 1ms                         |
| Param accessors               | Sync            | Trivial lookups                     |
| Encode/decode                 | Sync            | Fast for typical slot counts        |
| Encrypt/decrypt               | Sync            | Fast single-ciphertext ops          |
| Marshal/unmarshal             | Sync            | Pure serialization                  |
| Key generation (SK, PK)       | Sync            | Fast, < 50ms                        |
| RLK generation                | Sync            | ~100ms–1s, acceptable for sync      |
| Galois key generation         | Sync            | ~100ms per key, called in user loop |
| Bootstrap params construction | Sync            | Just parameter parsing              |
| `btpParamsGenEvaluationKeys`  | Async (Promise) | 5–30s, MUST not block UI            |

### Server API (browser demo)

| Endpoint              | Method | Request                     | Response                                                          |
| --------------------- | ------ | --------------------------- | ----------------------------------------------------------------- |
| `/params`             | GET    | —                           | `{"ckks_params": {...}, "key_manifest": {...}, "input_level": N}` |
| `/session`            | POST   | `MemEvaluationKeySet` bytes | `{"session_id": "uuid"}`                                          |
| `/session/{id}/infer` | POST   | Ciphertext bytes            | Result ciphertext bytes                                           |
| `/`                   | GET    | —                           | Static files (index.html, client.js, lattigo.wasm)                |

### Serialization compatibility

Keys and ciphertexts are serialized with Lattigo's `MarshalBinary()` — byte-compatible across Python bridge, Go evaluator, and WASM bridge. A ciphertext encrypted in the browser can be decrypted by the Python client, and vice versa.

## Post-Completion

**Manual verification:**

- Browser demo tested on Chrome, Firefox, Safari
- WASM load time measured on real network (with gzip compression)
- Key generation timing profiled in browser (should match `experiments/lattigo-wasm-bootstrap/` benchmarks)
- Memory usage profiled in browser DevTools (check for handle leaks)

**Future work (not in scope):**

- npm registry publishing (`@orion/lattigo`)
- Pre-built WASM binary in CI/CD
- Web Worker offloading for keygen (parallel key generation)
- Streaming inference (multiple ciphertexts)
