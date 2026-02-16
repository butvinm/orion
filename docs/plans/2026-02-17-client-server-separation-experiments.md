# Client-Server Key Separation — Phase 1: Experiments

## Overview

Orion currently couples scheme initialization, key generation, and compilation into a single monolithic flow. This prevents client-server deployment where:

- **Server** owns the model, compiles it, and runs FHE inference
- **Client** generates all cryptographic keys (sk, pk, eval keys), sends only evaluation keys to the server

This plan covers **Phase 1: throw-away experiments**. Each experiment proves a single hypothesis. Results feed into Phase 2 (production implementation).

### Target architecture

```
Client                              Server
──────                              ──────
                                    compile(model) → key_manifest
                              ←── key_manifest (galois elements, bootstrap params, rlk)
generate_keys(sk, manifest)
  → pk, rlk, galois_keys,
    bootstrap_keys
                              ──→ eval_keys (pk, rlk, galois, bootstrap)
encrypt(data, pk) ──────────────→ ciphertext
                                    infer(ciphertext, eval_keys) → result_ct
                              ←── result_ciphertext
decrypt(result_ct, sk) → result
```

## Context

### Current coupling points (from codebase analysis)

1. **Go global singleton** (`scheme.go`): `Scheme` struct holds Params + KeyGen + SecretKey + PublicKey + all evaluators together
2. **`init_scheme()`** (`orion.py:55-77`): Creates params AND generates all keys AND initializes all evaluators in one call
3. **`compile()`** (`orion.py:~280-294`): Generates bootstrap keys via `btpParams.GenEvaluationKeys(scheme.SecretKey)` — needs sk
4. **Rotation key generation** (`lineartransform.go:124-128`): `scheme.KeyGen.GenGaloisKeyNew(galEl, scheme.SecretKey)` — needs sk
5. **Evaluator init** (`evaluator.go:14-23`): `NewEvaluator()` calls `AddPo2RotationKeys()` which generates all power-of-2 rotation keys from sk — **server cannot use this constructor**
6. **Lazy key generation during inference** (`evaluator.go:34-46`): `AddRotationKey()` is called from `Rotate()`/`RotateNew()` and generates missing keys on the fly from sk — **server would crash on any missing rotation key**
7. **`module.compile()` generates keys** (`linear.py:124-135`, `linear.py:234-245`): `Linear.compile()` and `Conv2d.compile()` call `lt_evaluator.generate_transforms()` which triggers `generate_rotation_keys()` — needs sk
8. **Hybrid method output rotations** (`linear.py:71-72`): `evaluate_transforms()` calls `out.roll(slots // (2**i))` which calls Go `Rotate()` → lazy key generation from sk
9. **Key persistence** (`key_generator.py`, `lt_evaluator.py`): Single HDF5 file mixes sk with eval keys

### Key files

- `orion/core/orion.py` — `Scheme` class, orchestrates everything
- `orion/backend/lattigo/scheme.go` — Go global state
- `orion/backend/lattigo/keygenerator.go` — Key generation
- `orion/backend/lattigo/evaluator.go` — Evaluator setup
- `orion/backend/lattigo/bootstrapper.go` — Bootstrap key gen + eval
- `orion/backend/lattigo/lineartransform.go` — Rotation key gen + linear transform eval
- `orion/backend/lattigo/bindings.py` — Python-Go FFI
- `orion/backend/python/key_generator.py` — Python key management
- `orion/backend/python/lt_evaluator.py` — Python linear transform + rotation key management

## Development Approach

- **Testing approach**: Each experiment is its own test — a standalone script that either succeeds or fails
- Experiments are **throw-away code** in `experiments/` directory
- Each experiment proves exactly one hypothesis
- Experiments build on each other's findings but are independently runnable
- **Library modifications are allowed** for experiments, but every change must be documented in the experiment's notes (file, function, what changed, why) so they can be reverted after Phase 1 and properly reimplemented in Phase 2

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix
- Update plan if experiments reveal unexpected constraints

## Implementation Steps

### Task 1: Experiment 1 — Go client-server eval key roundtrip

**Hypothesis**: Keys (pk, rlk, galois keys) serialized from one Lattigo instance can be deserialized and used for ciphertext evaluation (ct-ct multiply, relinearize, rotate) on another instance that **never had the secret key**.

**What this proves**: The fundamental building block — that Lattigo's key serialization produces self-contained key objects usable without the originating keygen/sk.

- [x] Create `experiments/01_key_roundtrip/` with `go.mod` (depends on `github.com/baahl-nyu/lattigo/v6`)
- [x] Write `main.go` that simulates client and server as separate function scopes:
  - "Client" scope: create params, keygen, generate sk/pk/rlk/galois keys for specific rotations, serialize all eval keys to bytes via `MarshalBinary`
  - "Server" scope: create params (same config), deserialize pk/rlk/galois keys via `UnmarshalBinary`, construct `rlwe.MemEvaluationKeySet` from deserialized keys, create `ckks.Evaluator` from eval key set — **without calling `AddPo2RotationKeys` or any keygen function**
  - Server: receive ciphertext (serialized from client), perform ciphertext-ciphertext multiply + relinearize + rotate (using only pre-loaded keys)
  - Client: decrypt result, verify correctness against cleartext reference
- [x] Verify: server scope never touches `rlwe.SecretKey` or `rlwe.KeyGenerator`
- [x] Verify: server rotation with a **missing** galois key fails explicitly (not silent corruption)
- [x] Document which Lattigo API calls are needed for key deserialization and evaluator construction
- [x] Run experiment — must produce correct results

### Task 2: Experiment 2 — Go bootstrap with imported keys

**Hypothesis**: A `bootstrapping.Evaluator` can be constructed from deserialized bootstrap evaluation keys, without the secret key present on the server side.

**What this proves**: That the most complex key type (bootstrap keys — generated via `btpParams.GenEvaluationKeys(sk)`) can be serialized, transferred, and used independently. This is critical because bootstrap is the operation that refreshes ciphertext noise budget.

- [x] Create `experiments/02_bootstrap_imported/` with `go.mod`
- [x] Investigate `bootstrapping.EvaluationKeys` struct — does it implement `MarshalBinary` directly, or must component keys be serialized individually?
- [x] Write `main.go` with client-server simulation:
  - Client: create CKKS params + bootstrap params, generate sk, call `btpParams.GenEvaluationKeys(sk)` to get bootstrap eval keys, serialize the key bundle (top-level or component-wise depending on investigation above)
  - Server: deserialize bootstrap keys, construct `bootstrapping.Evaluator` via `bootstrapping.NewEvaluator(btpParams, btpKeys)` — verify it accepts deserialized keys without needing sk
  - Client: encrypt a low-level ciphertext and send to server
  - Server: bootstrap the ciphertext (refresh level), send back
  - Client: decrypt, verify values are approximately correct (bootstrap introduces small error)
- [x] If `bootstrapping.NewEvaluator` doesn't directly accept deserialized keys, document what wrapper or workaround is needed
- [x] Document the serialized size of bootstrap keys (these are large — important for transfer planning)
- [x] Run experiment — must produce correct results within acceptable error bounds

### Task 3: Experiment 3 — Python keyless compilation

**Hypothesis**: Orion's `compile()` pipeline (FX tracing → NetworkDAG → LevelDAG → bootstrap placement → diagonal packing) can run without generating any cryptographic keys, and can emit a **key requirements manifest** listing all required Galois elements, bootstrap slot counts, and relinearization key flag.

**What this proves**: That we can cleanly split Orion's compile phase into "planning" (server, no keys) and "key generation" (client, has sk). This is the core Python-side architectural experiment.

**Known coupling points to bypass** (from code analysis — not discovery work):

1. `init_scheme()` (`orion.py:69-75`) — creates keygen + all evaluators after params. Must split into params-only init.
2. `NewEvaluator()` (`evaluator.go:14-23`) — calls `AddPo2RotationKeys()` from sk. Server needs a constructor that accepts pre-built keys.
3. `lt_evaluator.generate_rotation_keys()` (`lt_evaluator.py:58-83`) — called from `generate_transforms()` during `module.compile()`. Must be replaced with manifest collection.
4. `bootstrapper.generate_bootstrapper()` (`bootstrapper.go:46`) — calls `GenEvaluationKeys(sk)`. Must be skipped, slot count collected instead.
5. Go `Encoder` (`encoder.go`) — uses `scheme.Params` only, **no keys needed**. Safe to initialize without keygen.

**Approach**: Modify library internals directly (document all changes):

- Add `init_params_only()` path to `Scheme` that creates backend + encoder + params but skips keygen/encryptor/evaluators
- Modify `lt_evaluator` to support a `collect_keys_only` mode where `generate_rotation_keys()` appends Galois elements to a manifest list instead of generating keys
- Skip `bootstrapper.generate_bootstrapper()` calls, collect slot counts into manifest
- Skip `NewEvaluator()` call (not needed for compilation planning)

- [x] Create `experiments/03_keyless_compile/` directory
- [x] Add `init_params_only()` method to `Scheme` class — creates backend + encoder, skips keygen/encryptor/evaluators. Document the change.
- [x] Add `collect_keys_only` mode to `lt_evaluator.NewEvaluator` — when enabled, `generate_rotation_keys()` appends to a `required_galois_elements` set instead of calling Go keygen. Document the change.
- [x] Modify `compile()` flow to skip `bootstrapper.generate_bootstrapper()` when in keyless mode, collecting `bootstrapper_slots` into the manifest instead. Document the change.
- [x] Collect **all** required Galois elements, including:
  - Linear transform rotation keys (from `GetLinearTransformRotationKeys`)
  - Power-of-2 rotation keys (from `AddPo2RotationKeys` — `1, 2, 4, ...` up to `MaxSlots`)
  - Hybrid method output rotations (from `linear.py:71-72` — `slots // (2**i)` for each linear layer's `output_rotations`)
- [x] Write experiment script that: inits params only → loads MLP model → fits → compiles in keyless mode → prints manifest
- [x] Verify Go `Encoder` works without keygen (expected: yes, it only needs `scheme.Params`)
- [x] Document all library modifications made (file, function, what changed, why)
- [x] Run experiment — compilation must complete and produce a valid manifest

### Task 4: Experiment 4 — Server-side evaluator from imported keys via FFI

**Hypothesis**: New Go FFI exports can load externally-provided eval keys (pk, rlk, galois keys, bootstrap keys) and construct working evaluators without keygen or sk on the Go side.

**What this proves**: That the Python-Go FFI boundary supports the key import pattern proven in Experiments 1-2, and that Orion's Go evaluators can be initialized from imported keys.

**Required new Go exports** (based on Experiments 1-2 findings):

1. `NewEvaluatorFromKeys()` — like `NewEvaluator()` but takes a pre-built `MemEvaluationKeySet`, does NOT call `AddPo2RotationKeys()`
2. `LoadPublicKey(data, len)` — deserialize pk into `scheme.PublicKey`
3. `LoadRelinKey(data, len)` — deserialize rlk into `scheme.RelinKey`
4. `LoadGaloisKey(data, len, galEl)` — deserialize a single galois key (extends existing `LoadRotationKey`)
5. `LoadBootstrapKeys(data, len, slots)` — deserialize bootstrap eval keys, construct `bootstrapping.Evaluator`
6. Modify `Rotate()`/`RotateNew()` — remove lazy `AddRotationKey()` call; panic if key is missing instead of generating from sk

- [x] Create `experiments/04_key_import_ffi/` directory
- [x] Add new Go exports listed above to the Lattigo backend. Document each change.
- [x] Add corresponding Python FFI bindings in `bindings.py`. Document each change.
- [x] Modify `Rotate()`/`RotateNew()` in `evaluator.go` to NOT lazy-generate missing keys — fail explicitly instead. Document the change.
- [x] Write Python test script that:
  - Inits scheme normally (with keys) as the "client"
  - Serializes pk, rlk, a few galois keys via existing `SerializeSecretKey`-style pattern
  - Creates a second scheme instance (params only) as the "server"
  - Loads serialized keys via new FFI functions
  - Constructs evaluator via `NewEvaluatorFromKeys`
  - Performs a simple ct-ct multiply + rotate on server side
  - Decrypts on client side, verifies correctness
- [x] Document all Go and Python changes made (file, function, what changed, why)
- [x] Run experiment — operations must produce correct results

### Task 5: Experiment 5 — Full client-server inference roundtrip

**Hypothesis**: End-to-end flow works: server compiles keylessly → exports key manifest → client generates keys from manifest → server imports keys via FFI → neural network inference on encrypted data produces correct results.

**What this proves**: That the entire pipeline can be split across client and server with only eval keys crossing the boundary. Integrates all previous experiments into Orion's actual inference flow.

**Depends on**: Task 3 (keyless compilation + manifest) and Task 4 (key import FFI).

- [x] Create `experiments/05_full_roundtrip/` directory
- [x] Write a Python script simulating the full flow:
  - **Server phase 1**: `init_params_only()`, load MLP model, fit, compile in keyless mode → get key manifest
  - **Client phase**: `init_scheme()` normally, generate all keys from manifest (sk + pk + rlk + galois keys for all listed elements + bootstrap keys for listed slots), serialize eval keys (everything except sk)
  - **Server phase 2**: load eval keys via Task 4's FFI functions, construct evaluators via `NewEvaluatorFromKeys`, construct bootstrappers via `LoadBootstrapKeys`
  - **Client**: encode + encrypt test input
  - **Server**: `net.he()` → forward pass on ciphertext
  - **Client**: decrypt + decode, compare with cleartext reference
- [x] Verify ALL required Galois elements from manifest are loaded before inference (no lazy generation possible after Task 4's `Rotate` change)
- [x] Measure and document: key manifest size, serialized eval key sizes, inference accuracy vs. monolithic baseline
- [x] Document all additional changes needed beyond Tasks 3-4
- [x] Run experiment — inference results must match monolithic Orion within acceptable FHE error tolerance

### Task 6: Document experiment results in REPORT.md

- [x] Create `experiments/REPORT.md` documenting:
  - Each experiment's hypothesis, approach, and result (confirmed/refuted/modified)
  - Lattigo API patterns discovered for key serialization/deserialization
  - Serialized key sizes for each key type (important for transfer protocol design)
  - Complete list of coupling points found in Orion's current architecture
  - All library modifications made during experiments (file, function, change, purpose) — to be reverted before Phase 2
  - Minimal set of new Go FFI functions needed for production client-server split
  - Minimal set of Python-side changes needed
  - Recommended architecture for Phase 2 production implementation
  - Open questions and risks for Phase 2
- [x] Include code snippets for key patterns discovered (e.g., how to construct evaluator from deserialized keys, how to serialize bootstrap keys)
- [x] Format and review report

## Technical Details

### Key types and their roles

| Key                      | Generated from | Needed by                 | Serializable?       |
| ------------------------ | -------------- | ------------------------- | ------------------- |
| SecretKey (sk)           | KeyGen         | Decrypt, key generation   | Yes (MarshalBinary) |
| PublicKey (pk)           | KeyGen + sk    | Encrypt                   | Yes                 |
| RelinearizationKey (rlk) | KeyGen + sk    | Ciphertext multiplication | Yes                 |
| GaloisKey (per element)  | KeyGen + sk    | Ciphertext rotation       | Yes                 |
| BootstrapEvalKeys        | btpParams + sk | Bootstrap (level refresh) | TBD (Experiment 2)  |

### Expected manifest format

```python
{
    "ckks_params": { ... },            # So client can create matching params
    "galois_elements": [               # ALL required rotation keys (union of all sources):
        # 1. Power-of-2 keys: params.GaloisElement(1), (2), (4), ... up to MaxSlots
        # 2. Linear transform keys: from GetLinearTransformRotationKeys() per transform
        # 3. Hybrid output rotations: params.GaloisElement(slots // 2^i) per layer
        # 4. Bootstrap-internal rotations (if determinable from params)
    ],
    "bootstrap_slots": [4096, 8192],   # Required bootstrap configurations
    "needs_rlk": True,                 # Relinearization key needed (always true for ct-ct mul)
    "bootstrap_params": { ... }        # Bootstrap-specific CKKS params (LogP etc.)
}
```

**Note on completeness**: The manifest must contain ALL Galois elements the server will ever need. After Task 4's modification to `Rotate()`, any missing key causes an explicit failure — no silent fallback. This is by design: the server must never need the secret key.

### Client-server boundary

- **Crosses the boundary** (client → server): pk, rlk, galois keys, bootstrap keys, encrypted data
- **Never crosses** (stays on client): sk
- **Server-only**: compiled model, diagonal matrices, level assignments

## Post-Completion

**Phase 2 planning** (after experiments):

- Design production API based on experiment findings
- Refactor Go singleton into client/server components
- Add proper key serialization format (possibly replacing HDF5 for keys)
- Design key manifest protocol
- Integration tests with real models
