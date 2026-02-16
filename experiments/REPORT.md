# Client-Server Key Separation Experiments: Report

## Executive Summary

Five experiments were conducted to validate that Orion's FHE inference pipeline can be split into a client-server architecture where the server never touches the secret key. All five hypotheses were confirmed. The experiments demonstrate a complete end-to-end flow: server compiles a neural network without keys, emits a key manifest, client generates keys from the manifest, server imports evaluation keys via FFI, and FHE inference on encrypted data produces correct results (MAE 0.000599 vs cleartext, well within the 0.005 tolerance).

## Experiment Results

### Experiment 1: Go Client-Server Eval Key Roundtrip

- **Hypothesis**: Keys (pk, rlk, galois keys) serialized from one Lattigo instance can be deserialized and used for ciphertext evaluation on another instance that never had the secret key.
- **Result**: CONFIRMED
- **Approach**: Pure Go program simulating client and server as separate function scopes. Client generates all keys and serializes them via `MarshalBinary`. Server deserializes via `UnmarshalBinary`, constructs `rlwe.MemEvaluationKeySet`, creates `ckks.Evaluator` -- without ever calling `AddPo2RotationKeys` or any keygen function. Server performs ct-ct multiply + relinearize + rotate using only pre-loaded keys.
- **Accuracy**: Max error 9.32e-07 across 4096 slots (CKKS with LogScale=40)
- **Missing key behavior**: Rotation with a missing galois key returns an error: `"cannot RotateNew: cannot Automorphism: GaloisKey[X] is nil"`. No silent corruption.

### Experiment 2: Go Bootstrap with Imported Keys

- **Hypothesis**: A `bootstrapping.Evaluator` can be constructed from deserialized bootstrap evaluation keys, without the secret key present on the server side.
- **Result**: CONFIRMED
- **Approach**: Client generates bootstrap eval keys via `btpParams.GenEvaluationKeys(sk)`, serializes the entire `bootstrapping.EvaluationKeys` struct via `MarshalBinary`. Server deserializes via `UnmarshalBinary` and constructs `bootstrapping.NewEvaluator(btpParams, btpKeys)` -- no sk, no keygen. Server bootstraps a level-0 ciphertext back to max level.
- **Accuracy**: Max error 2.89e-08 (25 bits precision) at LogN=10
- **Key finding**: `bootstrapping.EvaluationKeys` implements `MarshalBinary`/`UnmarshalBinary` directly. Component keys do NOT need to be serialized individually. No workaround needed.
- **Bootstrap key size**: 36.13 MB at LogN=10. At production LogN=13-16 these will be significantly larger.

### Experiment 3: Python Keyless Compilation

- **Hypothesis**: Orion's `compile()` pipeline (FX tracing -> NetworkDAG -> LevelDAG -> bootstrap placement -> diagonal packing) can run without generating any cryptographic keys, and can emit a key requirements manifest.
- **Result**: CONFIRMED
- **Approach**: Added `init_params_only()` to the Scheme class, `keyless` mode to `lt_evaluator`, and manifest collection to `compile()`. Go Encoder works without keygen (encode/decode roundtrip error: 2.09e-06). Full compile pipeline runs in keyless mode with MLP on MNIST config.
- **Manifest produced**: 211 unique Galois elements from three sources:
  - Linear transform rotations: 233 elements (from `GetLinearTransformRotationKeys`)
  - Power-of-2 rotations: 13 elements (1, 2, 4, ..., MaxSlots/2)
  - Hybrid output rotations: 1 element (from `linear.py` output rotation logic)
  - Bootstrap slots: [4096]
- **Backward compatibility**: All 9 existing tests pass with the modifications.

### Experiment 4: Server-Side Evaluator from Imported Keys via FFI

- **Hypothesis**: New Go FFI exports can load externally-provided eval keys and construct working evaluators without keygen or sk on the Go side.
- **Result**: CONFIRMED
- **Approach**: Added Go FFI exports for key serialization/loading and a `NewEvaluatorFromKeys()` constructor. Modified `Rotate()`/`RotateNew()` to not lazy-generate keys when sk is nil. Python test performs full client-server roundtrip: serialize keys on client -> load on server -> multiply + rotate -> decrypt on client.
- **Accuracy**: Max error 3.70e-03 (CKKS with LogScale=26 after mul+rescale+rotate)
- **Missing key behavior**: When `scheme.SecretKey` is nil (server mode), rotation with a missing galois key causes an explicit Go panic: `"cannot Rotate: cannot apply Automorphism: GaloisKey[X] is nil"`. Via CGO this causes process abort, not a Python exception.
- **Backward compatibility**: All 9 existing tests pass. The `Rotate`/`RotateNew` modification uses a runtime check (`scheme.KeyGen != nil && scheme.SecretKey != nil`) so existing code with keys is unaffected.

### Experiment 5: Full Client-Server Inference Roundtrip

- **Hypothesis**: End-to-end flow works: server compiles keylessly -> exports key manifest -> client generates keys from manifest -> server imports keys via FFI -> neural network inference on encrypted data produces correct results.
- **Result**: CONFIRMED
- **Approach**: Full 4-phase pipeline on MLP with MNIST data. Server phase 1: keyless compile. Client phase: keygen from manifest + encrypt. Server phase 2: load eval keys + FHE inference. Client phase 2: decrypt + verify.
- **Accuracy**: MAE 0.000599 vs cleartext (tolerance: 0.005, same as monolithic `test_mlp.py`)
- **Key manifest**: 31 galois elements, no bootstrap needed for this MLP/MNIST config
- **Eval key transfer**: ~97 MB total
- **No additional library modifications** beyond Tasks 3-4

## Lattigo API Patterns for Key Serialization

### Standard Key Serialization (Client Side)

```go
// Public key
pkBytes, err := pk.MarshalBinary()

// Relinearization key
rlkBytes, err := rlk.MarshalBinary()

// Galois key (one per rotation)
galEl := params.GaloisElement(rotation)
gk := kgen.GenGaloisKeyNew(galEl, sk)
gkBytes, err := gk.MarshalBinary()

// Ciphertext
ctBytes, err := ct.MarshalBinary()

// Bootstrap evaluation keys (entire bundle)
btpKeys, _, err := btpParams.GenEvaluationKeys(sk)
btpBytes, err := btpKeys.MarshalBinary()
```

### Standard Key Deserialization (Server Side)

```go
// Public key
pk := rlwe.NewPublicKey(params)
err := pk.UnmarshalBinary(pkBytes)

// Relinearization key
rlk := rlwe.NewRelinearizationKey(params)
err := rlk.UnmarshalBinary(rlkBytes)

// Galois key
gk := rlwe.NewGaloisKey(params)
err := gk.UnmarshalBinary(gkBytes)

// Ciphertext
ct := rlwe.NewCiphertext(params, degree, level)
err := ct.UnmarshalBinary(ctBytes)

// Bootstrap evaluation keys
btpKeys := &bootstrapping.EvaluationKeys{}
err := btpKeys.UnmarshalBinary(btpBytes)
```

Note: `NewPublicKey`, `NewRelinearizationKey`, `NewGaloisKey` require params to allocate correctly-sized ring elements before `UnmarshalBinary`.

### Evaluator Construction from Deserialized Keys (Server Side)

```go
// Standard evaluator (no sk, no keygen)
evk := rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...)
eval := ckks.NewEvaluator(params, evk)

// Bootstrap evaluator (no sk, no keygen)
btpEval, err := bootstrapping.NewEvaluator(btpParams, btpKeys)
```

### Python FFI Key Import Workflow

```python
# Server loads keys from serialized bytes
backend.LoadRelinKey(rlk_data)
backend.GenerateEvaluationKeys()   # creates MemEvaluationKeySet from loaded rlk
for galEl, gk_bytes in galois_keys.items():
    backend.LoadRotationKey(gk_bytes, galEl)
backend.NewEvaluatorFromKeys()     # creates evaluator from loaded EvalKeys
                                   # (no AddPo2RotationKeys, no sk)
# For bootstrap
backend.LoadBootstrapKeys(btp_bytes, slots, logPs)
```

## Serialized Key Sizes

| Key Type                  | Size (LogN=13, ConjugateInvariant) | Size (LogN=10, Standard) |
| ------------------------- | ---------------------------------- | ------------------------ |
| Secret Key (sk)           | 0.50 MB                            | --                       |
| Public Key (pk)           | 1.00 MB                            | --                       |
| Relinearization Key (rlk) | 3.00 MB                            | --                       |
| Galois Key (each)         | 3.00 MB                            | --                       |
| Ciphertext                | ~768 KB                            | --                       |
| Bootstrap Keys            | --                                 | 36.13 MB                 |

For the MLP/MNIST experiment (LogN=13, 31 galois keys): total eval key transfer was ~97 MB.

Note: Bootstrap key size at LogN=10 is 36 MB. At production LogN=13-16, expect hundreds of MB to several GB for bootstrap keys. This is a major consideration for the transfer protocol.

## Coupling Points in Orion's Architecture

The following coupling points between key generation and compilation/inference were identified and addressed:

| #   | Location                                                | Coupling                                                                 | Resolution                                                                                          |
| --- | ------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| 1   | `scheme.go`: `Scheme` struct                            | Holds Params + KeyGen + SK + PK + all evaluators together                | Added `EvalKeys` field; `NewScheme` initializes `KeyGen` but leaves all keys nil                    |
| 2   | `orion.py`: `init_scheme()`                             | Creates params AND keys AND evaluators in one call                       | Added `init_params_only()` that creates only backend + encoder                                      |
| 3   | `orion.py`: `compile()`                                 | Calls `bootstrapper.generate_bootstrapper()` which needs sk              | In keyless mode, skips bootstrap key generation, records slot counts in manifest                    |
| 4   | `lineartransform.go`                                    | `GetLinearTransformRotationKeys` + `GenGaloisKeyNew` during compile      | In keyless mode, `lt_evaluator` collects Galois elements into a set instead                         |
| 5   | `evaluator.go`: `NewEvaluator()`                        | Calls `AddPo2RotationKeys()` which generates all po2 keys from sk        | Added `NewEvaluatorFromKeys()` that takes pre-loaded `EvalKeys` without key generation              |
| 6   | `evaluator.go`: `Rotate()`/`RotateNew()`                | Lazy-generates missing rotation keys from sk                             | Modified to check `scheme.KeyGen != nil && scheme.SecretKey != nil` -- server panics on missing key |
| 7   | `linear.py`: `compile()`                                | Calls `lt_evaluator.generate_transforms()` -> `generate_rotation_keys()` | In keyless mode, rotation key generation is replaced by manifest collection                         |
| 8   | `linear.py`: `evaluate_transforms()`                    | `out.roll()` calls Go `Rotate()` -> lazy key generation                  | After Exp 4, `Rotate()` no longer lazy-generates when sk is nil                                     |
| 9   | Key persistence (`key_generator.py`, `lt_evaluator.py`) | Single HDF5 file mixes sk with eval keys                                 | Not addressed in experiments; noted for Phase 2                                                     |

## Library Modifications Made During Experiments

All modifications below are experimental and should be reverted before Phase 2 production implementation.

### Go Backend

| File                 | Function                            | Change                                           | Purpose                                                                                      |
| -------------------- | ----------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `scheme.go`          | `Scheme` struct                     | Added `EvalKeys *rlwe.MemEvaluationKeySet` field | Holds imported evaluation keys for server-side evaluator construction                        |
| `scheme.go`          | `GetGaloisElement()`                | New export                                       | Python can compute Galois elements for rotations via Lattigo's ring-specific formula         |
| `scheme.go`          | `GetMaxSlots()`                     | New export                                       | Python can compute po2 rotation key range without hardcoding                                 |
| `evaluator.go`       | `NewEvaluatorFromKeys()`            | New export                                       | Constructs evaluator from pre-loaded `EvalKeys` without calling `AddPo2RotationKeys()`       |
| `evaluator.go`       | `Rotate()`                          | Modified                                         | Conditional lazy key generation: only when `scheme.KeyGen != nil && scheme.SecretKey != nil` |
| `evaluator.go`       | `RotateNew()`                       | Modified                                         | Same conditional check as `Rotate()`                                                         |
| `keygenerator.go`    | `SerializePublicKey()`              | New export                                       | Serializes pk via `MarshalBinary`                                                            |
| `keygenerator.go`    | `LoadPublicKey()`                   | New export                                       | Deserializes pk via `UnmarshalBinary`, sets `scheme.PublicKey`                               |
| `keygenerator.go`    | `SerializeRelinKey()`               | New export                                       | Serializes rlk via `MarshalBinary`                                                           |
| `keygenerator.go`    | `LoadRelinKey()`                    | New export                                       | Deserializes rlk via `UnmarshalBinary`, sets `scheme.RelinKey`                               |
| `lineartransform.go` | `GenerateAndSerializeRotationKey()` | New export                                       | Generates a single galois key from sk and serializes it                                      |
| `lineartransform.go` | `LoadRotationKey()`                 | New export                                       | Deserializes a galois key and loads it into `scheme.EvalKeys`                                |
| `bootstrapper.go`    | `SerializeBootstrapKeys()`          | New export                                       | Generates bootstrap eval keys from sk and serializes via `MarshalBinary`                     |
| `bootstrapper.go`    | `LoadBootstrapKeys()`               | New export                                       | Deserializes bootstrap keys and constructs `bootstrapping.Evaluator`                         |
| `tensors.go`         | `SerializeCiphertext()`             | New export                                       | Serializes a ciphertext from the heap                                                        |
| `tensors.go`         | `LoadCiphertext()`                  | New export                                       | Deserializes a ciphertext and pushes to heap                                                 |

### Python Backend

| File              | Function/Class             | Change                               | Purpose                                                                    |
| ----------------- | -------------------------- | ------------------------------------ | -------------------------------------------------------------------------- |
| `bindings.py`     | --                         | Added FFI bindings                   | For all new Go exports listed above                                        |
| `lt_evaluator.py` | `NewEvaluator.__init__()`  | Added `keyless` parameter            | Skips `NewLinearTransformEvaluator()` and evaluator access in keyless mode |
| `lt_evaluator.py` | `NewEvaluator`             | Added `required_galois_elements` set | Accumulates Galois elements during keyless compilation                     |
| `lt_evaluator.py` | `generate_rotation_keys()` | Modified                             | In keyless mode, appends to manifest set instead of generating keys        |

### Core

| File                             | Function/Class                         | Change                    | Purpose                                                                                      |
| -------------------------------- | -------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------- |
| `orion.py`                       | `Scheme`                               | Added `keyless` attribute | Boolean flag, defaults to `False`                                                            |
| `orion.py`                       | `Scheme.init_params_only()`            | New method                | Creates backend + encoder, skips keygen/encryptor/evaluators                                 |
| `orion.py`                       | `Scheme.compile()`                     | Modified                  | In keyless mode: skips bootstrap key gen, builds manifest, returns `(input_level, manifest)` |
| `orion.py`                       | `Scheme._build_key_manifest()`         | New method                | Collects all required Galois elements from 3 sources + bootstrap slots                       |
| `orion.py`                       | `Scheme._get_po2_galois_elements()`    | New method                | Computes power-of-2 Galois elements via Go backend                                           |
| `orion.py`                       | `Scheme._rotation_to_galois_element()` | New method                | Delegates to Go `GetGaloisElement`                                                           |
| `__init__.py` (core + top-level) | --                                     | Added exports             | Exposes `init_params_only` at flat API level                                                 |

## Minimal New Go FFI Functions for Production

The following FFI functions are the minimal set needed for a production client-server split:

1. `GetGaloisElement(rotation) -> galEl` -- compute Galois element from rotation amount
2. `GetMaxSlots() -> int` -- get max slots for po2 key range
3. `NewEvaluatorFromKeys()` -- construct evaluator from pre-loaded keys (no sk)
4. `LoadPublicKey(data, len)` -- import pk
5. `LoadRelinKey(data, len)` -- import rlk
6. `LoadRotationKey(data, len, galEl)` -- import a single galois key
7. `LoadBootstrapKeys(data, len, slots, LogPs, lenLogPs)` -- import bootstrap keys + construct evaluator
8. `GenerateAndSerializeRotationKey(galEl) -> bytes` -- client-side key generation + serialization
9. `SerializePublicKey() -> bytes` -- client-side pk export
10. `SerializeRelinKey() -> bytes` -- client-side rlk export
11. `SerializeBootstrapKeys(slots, LogPs, lenLogPs) -> bytes` -- client-side bootstrap key export
12. `SerializeCiphertext(ctID) -> bytes` -- ciphertext transfer
13. `LoadCiphertext(data, len) -> ctID` -- ciphertext import

The `Rotate()`/`RotateNew()` conditional check (no lazy key generation when sk is nil) must also be retained.

## Minimal Python-Side Changes for Production

1. `init_params_only()` on `Scheme` -- params-only initialization path
2. `keyless` mode on `lt_evaluator` -- manifest collection instead of key generation
3. `compile()` modification -- skip bootstrap key gen, build manifest in keyless mode
4. `_build_key_manifest()` -- collect all Galois elements from linear transforms, po2 rotations, and hybrid output rotations
5. Server-side evaluator wrapper (like Experiment 5's `ServerEvaluator`) -- wraps the Go backend evaluator initialized via `NewEvaluatorFromKeys()`

## Recommended Architecture for Phase 2

### Refactored Go Singleton

The Go `Scheme` struct should be split into:

- `SchemeParams` -- holds `Params` only, shared by client and server
- `ClientScheme` -- extends with `KeyGen`, `SecretKey`, encryption/decryption
- `ServerScheme` -- extends with imported `EvalKeys`, evaluators, linear transforms

Alternatively, the current singleton can be retained with clear mode flags, but the struct should enforce that server mode never allows sk access.

### Key Manifest Protocol

The manifest should be a serializable JSON object containing:

- CKKS parameter specification (so client can create matching params)
- Complete list of required Galois elements (uint64)
- Bootstrap slot configurations
- RLK requirement flag
- Bootstrap params (LogP values)

### Key Transfer Format

Replace HDF5 for key storage with a streaming binary format:

- Individual key blobs via `MarshalBinary` (already proven to work)
- A container format that bundles pk + rlk + N galois keys + M bootstrap key sets
- Consider compression -- galois keys at 3 MB each, 30+ keys = 90+ MB uncompressed

### Production API (Proposed)

```python
# Server
server = orion.Server()
server.init_params(config)
server.fit(net, dataloader)
manifest = server.compile(net)
server.save_manifest("manifest.json")

# Client (separate process/machine)
client = orion.Client()
client.load_manifest("manifest.json")
client.generate_keys()
client.save_eval_keys("eval_keys.bin")  # everything except sk
ct = client.encrypt(input_data)
client.save_ciphertext(ct, "input.bin")

# Server (after receiving keys + ciphertext)
server.load_eval_keys("eval_keys.bin")
result_ct = server.infer(net, "input.bin")
server.save_ciphertext(result_ct, "output.bin")

# Client (after receiving result)
result = client.decrypt("output.bin")
```

## Open Questions and Risks for Phase 2

1. **Bootstrap key sizes at production LogN**: At LogN=10, bootstrap keys are 36 MB. At LogN=15-16, they could be multiple GB. Need to measure and plan for streaming transfer.

2. **Go singleton lifetime management**: Experiment 5 relies on NOT calling `DeleteScheme` between server phase 1 (compile) and server phase 2 (inference) so that linear transforms persist in the Go heap. Production needs to make this lifecycle explicit.

3. **Error handling for missing keys**: Currently a missing galois key causes a Go panic via CGO, which aborts the process. Production should catch this more gracefully -- either pre-validate key completeness against the manifest before inference, or convert Go panics to recoverable errors.

4. **HDF5 key persistence**: Current key persistence mixes sk with eval keys in HDF5 files. Phase 2 needs separate persistence paths for client keys (sk) and server keys (eval keys).

5. **Multi-client support**: Current architecture uses a Go global singleton. Supporting multiple clients with different keys requires either per-client state management or multiple Go instances.

6. **Key manifest completeness**: The manifest must be provably complete -- every Galois element the server will ever need during inference must be listed. If any code path generates a rotation not captured by the manifest, the server will crash. Need comprehensive testing across all model architectures.

7. **Hybrid method output rotations**: These are computed during `evaluate_transforms()` at inference time (`slots // (2**i)`). The manifest currently captures them during compile, but if the slot count or packing changes, the rotations would differ. Need to verify these are deterministic from compile-time information.

8. **Revert strategy**: All experiment modifications are in the main library. Before Phase 2, these should be reverted and reimplemented properly with clean abstractions, tests, and documentation.
