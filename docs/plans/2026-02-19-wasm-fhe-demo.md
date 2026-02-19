# WASM FHE Demo: Browser-Based Encrypted Inference

## Overview

Build a complete demo application showing FHE inference from a web browser:

- **FastAPI server** loads a pre-compiled MLP model and performs encrypted inference
- **Browser client** uses Lattigo compiled to WASM for client-side key generation, encryption, and decryption
- **Server is untrusted**: the secret key never leaves the browser. Server only receives evaluation keys (public) and ciphertexts
- Keys are uploaded individually (streaming), allowing progress tracking and reducing peak memory

The MLP model classifies MNIST digits (784 → 128 → 128 → 10). Input: 28×28 grayscale pixel values. Output: 10-class scores.

## Context

- **Existing Go backend**: `orion/backend/lattigo/` — CGO-based (`//export`), incompatible with WASM
- **WASM requires separate Go module**: `GOOS=js GOARCH=wasm` uses `syscall/js`, not CGO
- **Key serialization already exists**: Lattigo's `MarshalBinary()`/`UnmarshalBinary()` for all key types
- **CipherText serialization**: length-prefixed Lattigo binary blobs (see `client.py:46-68`)
- **EvalKeys is a plain dataclass**: can be constructed directly from raw byte blobs without going through the container format
- **Go backend singleton**: fine for single-tenant demo (one client session at a time)
- **Lattigo fork**: `github.com/baahl-nyu/lattigo/v6` (not upstream tuneinsight)
- **MLP params**: `CKKSParams(logn=13, logq=(29,26,26,26,26,26), logp=(29,29), logscale=26, h=8192, ring_type="conjugate_invariant")`

## Development Approach

- **Testing approach**: Regular (code first, then tests)
- Complete each task fully before moving to the next
- Make small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes in that task
- **CRITICAL: all tests must pass before starting next task**
- **CRITICAL: update this plan file when scope changes during implementation**
- Run tests after each change

## Testing Strategy

- **Go WASM module**: Go unit tests for wrapper logic (tested without WASM target, pure Go)
- **FastAPI server**: pytest with httpx.AsyncClient (mock Orion backend for unit tests)
- **Integration**: Python script that exercises the full HTTP flow with real Orion backend
- **Browser**: manual testing (no e2e framework for this demo)

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix

## Directory Structure

```
demo/wasm-fhe-demo/
├── wasm/
│   ├── crypto.go            # Pure Lattigo logic (no syscall/js — testable on any platform)
│   ├── bindings_js.go       # syscall/js wrappers (//go:build js && wasm)
│   ├── crypto_test.go       # Tests for crypto.go (runs with normal go test)
│   ├── go.mod
│   ├── go.sum
│   └── Makefile
├── server/
│   ├── app.py               # FastAPI application
│   ├── requirements.txt     # fastapi, uvicorn
│   └── test_app.py          # Server endpoint tests
├── client/
│   ├── index.html           # Single-page demo UI
│   ├── orion-client.js       # JS wrapper around WASM
│   └── wasm_exec.js         # Go WASM runtime (copied from Go toolchain)
├── tools/
│   ├── compile_model.py     # Script to compile MLP → model.bin
│   └── build_wasm.sh        # Script to compile Go → orion.wasm
├── model.bin                # Pre-compiled MLP model (generated)
└── README.md
```

## Implementation Steps

### Task 1: Project scaffolding and model compilation script

- [x] Create `demo/wasm-fhe-demo/` directory structure as shown above
- [x] Create `demo/wasm-fhe-demo/tools/compile_model.py`:
  - Uses `orion.Compiler` to compile `MLP()` with the params above
  - Calls `compiler.fit()` with random MNIST-shaped data (1×1×28×28)
  - Saves `compiled.to_bytes()` as `demo/wasm-fhe-demo/model.bin`
  - Also prints manifest info (galois_elements count, needs_rlk, bootstrap_slots) for verification
- [x] Create `demo/wasm-fhe-demo/server/requirements.txt` with fastapi, uvicorn, python-multipart
- [x] Run `compile_model.py` to generate `model.bin` and verify it loads back
- [x] Write test: load model.bin, verify params, manifest, input_level match expected values

### Task 2: Go WASM module — scheme and key generation

- [x] Create `demo/wasm-fhe-demo/wasm/go.mod` importing `github.com/baahl-nyu/lattigo/v6`
- [x] Create `demo/wasm-fhe-demo/wasm/crypto.go` (no `syscall/js` — pure Lattigo, testable on any platform):
  - Global `scheme` struct (Params, KeyGen, SecretKey, PublicKey, RelinKey, Encoder, Encryptor, Decryptor)
  - `InitScheme(logN int, logQ, logP []int, logScale, h int, ringType string) error`
  - `GetMaxSlots() int`
  - `SerializeRelinKey() ([]byte, error)`
  - `GenerateAndSerializeGaloisKey(galEl uint64) ([]byte, error)`
  - `SerializeBootstrapKeys(numSlots int, logP []int) ([]byte, error)`
- [x] Create `demo/wasm-fhe-demo/wasm/bindings_js.go` (`//go:build js && wasm`):
  - `syscall/js` wrappers that call `crypto.go` functions
  - Each wrapper converts `js.Value` args → Go types, calls crypto func, converts result → JS
  - All heavy functions return JS Promises via goroutine pattern for non-blocking browser execution
  - Registers functions on `js.Global()` in `main()`: `orionInit`, `orionGetMaxSlots`, `orionSerializeRelinKey`, `orionGenerateAndSerializeGaloisKey`, `orionSerializeBootstrapKeys`
  - Helper: `goBytesToJSUint8Array([]byte) js.Value`
- [x] Create `demo/wasm-fhe-demo/wasm/Makefile` with target: `GOOS=js GOARCH=wasm go build -ldflags="-s -w" -o ../client/orion.wasm .`
- [x] Create `demo/wasm-fhe-demo/tools/build_wasm.sh` that runs make and copies `wasm_exec.js` from `$(go env GOROOT)/misc/wasm/`
- [x] Write `crypto_test.go`: init scheme with MLP params, generate keys, serialize/deserialize round-trip (runs with normal `go test` — no WASM target needed)
- [x] Build WASM binary with `GOOS=js GOARCH=wasm` and verify it compiles without errors

### Task 3: Go WASM module — encode, encrypt, decrypt

- [x] Add to `crypto.go`:
  - `Encode(values []float64, level int, scale uint64) (int, error)` — encodes into plaintext, stores in internal heap, returns plaintext ID
  - `Encrypt(ptxtID int) ([]byte, error)` — encrypts plaintext → serializes raw Lattigo ciphertext → returns bytes (no wire format header — JS wrapper handles framing)
  - `Decrypt(ctBytes []byte) ([]float64, error)` — loads ciphertext from raw Lattigo bytes → decrypts → decodes → returns flat float64 slice
  - `GetDefaultScale() uint64`
  - Internal plaintext heap (slice-based, matching the minheap pattern in existing Go code)
- [x] Add to `bindings_js.go`:
  - `orionEncode`, `orionEncrypt`, `orionDecrypt`, `orionGetDefaultScale` wrappers
  - `orionEncrypt` converts Go `[]byte` → JS `Uint8Array`
  - `orionDecrypt` converts JS `Uint8Array` → Go `[]byte`, returns JS `Float64Array`
- [x] Write `crypto_test.go`: encode → encrypt → decrypt → decode round-trip with tolerance check (runs with `go test`)
- [x] Rebuild WASM, verify compile

### Task 4: JavaScript WASM wrapper

- [x] Copy `wasm_exec.js` from Go toolchain into `demo/wasm-fhe-demo/client/`
- [x] Create `demo/wasm-fhe-demo/client/orion-client.js`:
  - `OrionClient` class:
    - `async init(wasmUrl)` — loads WASM binary, instantiates Go runtime
    - `async setupScheme(params)` — calls `orionInit` with CKKSParams fields
    - `getMaxSlots()` — returns slot count
    - `async generateAndSerializeRlk()` — returns `Uint8Array`
    - `async generateAndSerializeGaloisKey(galEl)` — returns `Uint8Array`
    - `async generateAndSerializeBootstrapKeys(numSlots, logP)` — returns `Uint8Array`
    - `async encryptInput(float64Array, shape, level)` — encode + encrypt via WASM, then wrap raw ciphertext bytes with wire format header (num_cts, shape, ct_len) in JS to match Python `CipherText.to_bytes` format → `Uint8Array`
    - `async decryptOutput(uint8Array, numElements)` — strip wire format header in JS, pass raw ciphertext bytes to WASM decrypt → `Float64Array` (trimmed to numElements)
  - Progress callback support: `onProgress(stage, current, total)`
  - Error handling: catches Go panics (which become JS errors) and wraps them
- [x] No automated tests (JS tested via browser integration)

### Task 5: FastAPI server — manifest and key upload endpoints

- [ ] Create `demo/wasm-fhe-demo/server/app.py`:
  - On startup: load `model.bin` into `CompiledModel`
  - `GET /api/manifest` — returns JSON: `{params: {logn, logq, logp, logscale, h, ring_type}, manifest: {galois_elements, bootstrap_slots, boot_logp, needs_rlk}, input_level}`
  - `POST /api/keys/rlk` — accepts raw bytes body, stores in session dict
  - `POST /api/keys/galois/{gal_el}` — accepts raw bytes body, stores in session dict keyed by gal_el
  - `POST /api/keys/bootstrap/{slot_count}` — accepts raw bytes body, stores in session dict
  - `POST /api/keys/finalize` — constructs `EvalKeys(rlk_data=..., galois_keys=..., bootstrap_keys=...)`, creates `Evaluator(MLP(), compiled, keys)`, stores evaluator in session. Returns `{status: "ready"}`
  - `GET /api/keys/progress` — returns `{received_galois: N, total_galois: M, rlk: bool, bootstrap: [slot_counts]}` for client progress tracking
  - Session is a module-level dict (single-tenant demo, no auth needed)
- [ ] Mount static files: `app.mount("/", StaticFiles(directory="../client", html=True))`
- [ ] Write tests in `test_app.py`: test manifest endpoint returns correct structure, test key upload stores bytes, test finalize without all keys returns error
- [ ] Run tests

### Task 6: FastAPI server — inference endpoint

- [ ] Add to `app.py`:
  - `POST /api/infer` — accepts raw bytes body (serialized CipherText), calls `evaluator.run()`, returns serialized result CipherText as `application/octet-stream`
  - Error handling: return 400 if evaluator not initialized, 500 if inference fails
  - Response headers: include `Content-Type: application/octet-stream`
- [ ] Add `POST /api/reset` — clears session state (allows new client to connect). Destroys evaluator and Go backend
- [ ] Write tests: test infer without evaluator returns 400, test reset clears session
- [ ] Integration test in `test_app.py`: full flow using real Orion backend (skip if Go backend not available — mark with `pytest.mark.skipif`)
- [ ] Run tests

### Task 7: Web UI — initialization and key generation flow

- [ ] Create `demo/wasm-fhe-demo/client/index.html`:
  - Clean, minimal UI with step-by-step flow panels:
    1. **Step 1: Initialize** — "Connect" button fetches manifest, loads WASM, initializes scheme
    2. **Step 2: Generate & Upload Keys** — Shows manifest summary (N Galois keys, RLK yes/no, bootstrap slots). "Generate Keys" button starts key gen + upload with progress bar
    3. **Step 3: Encrypt & Infer** — 28×28 pixel grid canvas (drawable) or pre-loaded digit. "Encrypt & Send" button
    4. **Step 4: Result** — Shows encrypted result class scores after decryption
  - Security info panel: explains what data goes where, what server can/cannot see
  - Status log: shows each operation with timing
  - All network calls use `fetch()` API
  - Use CSS Grid/Flexbox, no external CSS frameworks
- [ ] Wire up Steps 1-2: manifest fetch → WASM init → key generation with progress
- [ ] Manual browser test: verify WASM loads, keys generate, progress updates

### Task 8: Web UI — encryption, inference, and decryption

- [ ] Wire up Step 3: capture canvas pixel data → normalize to [0,1] float64 → call `orionClient.encryptInput()` → POST to `/api/infer` → call `orionClient.decryptOutput()` → display result
- [ ] Add digit drawing canvas: 280×280 pixel canvas that downsamples to 28×28 for input
- [ ] Display results: bar chart of class probabilities (0-9), highlight predicted digit
- [ ] Add timing display: key gen time, encryption time, server inference time, decryption time
- [ ] Add "Reset" button that calls `/api/reset` and clears client state
- [ ] Manual browser test: full end-to-end flow

### Task 9: Verify acceptance criteria

- [ ] Verify all requirements: WASM client, FastAPI server, streaming key upload, untrusted server model
- [ ] Verify secret key never leaves browser (code review — no serialize/send of SK)
- [ ] Verify edge cases: what happens if WASM fails to load, if server is down, if keys are incomplete
- [ ] Run full test suite (pytest for server tests)
- [ ] Run Go tests for WASM module
- [ ] Verify WASM binary builds cleanly

### Task 10: [Final] Documentation

- [ ] Create `demo/wasm-fhe-demo/README.md`:
  - Prerequisites (Go 1.22+, Python 3.9+, Orion installed)
  - Quick start: build WASM, compile model, start server, open browser
  - Architecture diagram (ASCII): Browser ↔ FastAPI ↔ Orion/Lattigo
  - Security model explanation
  - Known limitations (WASM binary size, single-tenant, performance)
- [ ] Update main README.md to reference the demo

## Technical Details

### WASM Binary Size

Lattigo compiled to WASM will produce a large binary (~15-30MB uncompressed). Mitigation:

- Strip debug symbols: `-ldflags="-s -w"`
- Serve with gzip/brotli compression (FastAPI middleware): ~5-10MB compressed
- Show loading progress in UI

### CipherText Wire Format

Both WASM client and Python server use identical binary format:

```
[4 bytes]  NUM_CIPHERTEXTS (uint32 LE)
[4 bytes]  SHAPE_LEN (uint32 LE)
[N × 4 bytes]  SHAPE_DIMS (int32 LE each)
for each ciphertext:
    [8 bytes]  CT_LEN (uint64 LE)
    [N bytes]  CT_DATA (Lattigo MarshalBinary output)
```

### Key Upload Protocol

```
Client                          Server
  |                               |
  |--- GET /api/manifest -------->|  Returns CKKSParams + KeyManifest
  |                               |
  |--- POST /api/keys/rlk ------>|  Raw Lattigo RLK bytes
  |                               |
  |--- POST /api/keys/galois/N ->|  One Galois key at a time
  |    (repeat for each gal_el)   |  (server stores in dict)
  |                               |
  |--- POST /api/keys/finalize ->|  Server constructs EvalKeys + Evaluator
  |                               |
  |--- POST /api/infer ---------->|  Serialized CipherText
  |<-- encrypted result ----------|  Serialized CipherText
  |                               |
  | (client decrypts locally)     |
```

### Go WASM Promise Pattern

```go
func asyncOp(this js.Value, args []js.Value) interface{} {
    handler := js.FuncOf(func(_ js.Value, pArgs []js.Value) interface{} {
        resolve, reject := pArgs[0], pArgs[1]
        go func() {
            result, err := doWork()
            if err != nil {
                reject.Invoke(err.Error())
                return
            }
            resolve.Invoke(result)
        }()
        return nil
    })
    return js.Global().Get("Promise").New(handler)
}
```

### Security Model

| Data                               | Location                               | Confidentiality                      |
| ---------------------------------- | -------------------------------------- | ------------------------------------ |
| Secret Key                         | Browser WASM memory only               | Never transmitted                    |
| Public Key                         | Not needed by server                   | Generated but not sent               |
| Eval Keys (RLK, Galois, Bootstrap) | Sent to server                         | Public — cannot derive SK            |
| Input Plaintext                    | Browser only                           | Encrypted before transmission        |
| Input Ciphertext                   | Sent to server                         | Encrypted — server cannot decrypt    |
| Output Ciphertext                  | Returned from server                   | Encrypted — only client can decrypt  |
| Model Weights                      | Server only (baked into CompiledModel) | Not revealed to client               |
| Compiled Model Metadata            | Server only                            | Params + manifest sent (public info) |

## Post-Completion

**Manual verification:**

- Test in Chrome, Firefox, Safari (Go WASM support varies)
- Test with slow network (throttle in DevTools) — verify progress indicators work
- Measure actual WASM binary size and key generation time
- Verify memory usage doesn't exceed browser limits (Lattigo is memory-hungry)

**Performance expectations (rough):**

- WASM load: 2-5s (compressed transfer)
- Scheme init: 5-15s in WASM
- Key generation (MLP, ~20 Galois keys + RLK): 30-120s in WASM
- Key upload: depends on network, ~50-200MB total
- Encryption: 1-3s
- Server inference: 5-30s (native Go, much faster than WASM)
- Decryption: 1-3s
