# Lattigo Bindings Refactor

**Goal:** Make `python/lattigo` and `js/lattigo` pure Lattigo bindings with zero Orion
imports. Both expose the same Lattigo surface area. No JSON across FFI boundaries.

## Problem

`python/lattigo/bridge/` imports `github.com/baahl-nyu/orion` for:

1. `orion.Params` — Orion-specific JSON schema → `ckks.Parameters` conversion
2. `orion.GenerateMonomial/GenerateChebyshev` — thin wrappers around `bignum.NewPolynomial`
3. `orion.GenerateMinimaxSignCoeffs` — Lattigo API + Orion-specific caching + sign→\[0,1\] rescaling

This violates the stated design: lattigo bindings should depend only on `github.com/baahl-nyu/lattigo/v6`.

## Architecture

Three layers, same in both Python and JS:

1. **Wrapper layer** (Python classes / TS classes) — caller-facing API, accepts native types
2. **FFI layer** (ffi.py / bridge.ts) — translates native types ↔ C types
3. **Go bridge** (CGO exports / WASM exports) — thin C functions, imports only Lattigo

## Naming conventions

Each layer follows its language's idioms. The Go bridge is the shared contract.

- **Go bridge (CGO):** `PascalCase` — `NewCKKSParams`, `NewEncoder`, `KeyGenGenRelinKey`
- **Go bridge (WASM):** `camelCase` — `newCKKSParams`, `newEncoder`, `keyGenGenRelinKey`
- **Python FFI:** `snake_case` — `new_ckks_params`, `new_encoder`, `keygen_gen_relin_key`
- **Python wrapper:** Pythonic classes — `Parameters(...)`, `Encoder(params)`, `KeyGenerator(params)`
- **JS/TS wrapper:** JS classes — `new CKKSParameters({...})`, `new Encoder(params)`, `new KeyGenerator(params)`

### Naming fixes (current inconsistencies)

- Drop redundant `CKKS` prefix from Encoder/Encryptor/Decryptor (library is CKKS-only): `NewCKKSEncoder` → `NewEncoder`, `NewCKKSEncryptor` → `NewEncryptor`, `NewCKKSDecryptor` → `NewDecryptor`
- Shorten `RelinearizationKey` → `RelinKey` in bridge (align Python CGO with JS WASM): `KeyGenGenRelinearizationKey` → `KeyGenGenRelinKey`
- Rename `GeneratePolynomialMonomial` → `NewPolynomialMonomial` (match `New*` convention)
- Drop redundant `paramsH` arg from `EncryptorEncryptNew`/`DecryptorDecryptNew` (encryptor/decryptor already holds params from construction)

### Construction pattern

All wrapper classes use direct constructors — no `.new()` factory methods, no `.from_*()` classmethods.

Python:

```python
params = ckks.Parameters(logn=14, logq=[...], logp=[...], log_default_scale=40)
kg = rlwe.KeyGenerator(params)
encoder = ckks.Encoder(params)
encryptor = rlwe.Encryptor(params, pk)
```

JS:

```typescript
const params = new CKKSParameters({ logN: 14, logQ: [...], logP: [...], logDefaultScale: 40 });
const kg = new KeyGenerator(params);
const encoder = new Encoder(params);
const encryptor = new Encryptor(params, pk);
```

### Cleanup pattern

- Python: `close()` + `__del__` (idiomatic — matches file handles, context managers)
- JS: `free()` + `FinalizationRegistry` (idiomatic — matches WASM/C convention)

Both are correct for their language. No change needed.

## Changes

### 1. Parameters: flat C args, not JSON

**Go bridge** accepts individual C arguments:

```go
//export NewCKKSParams
func NewCKKSParams(
    logn C.int,
    logqPtr *C.int, logqLen C.int,
    logpPtr *C.int, logpLen C.int,
    logDefaultScale C.int,
    h C.int,
    ringType *C.char,
    logNthRoot C.int,
    errOut **C.char,
) C.uintptr_t
```

**Python wrapper** — constructor takes plain args:

```python
from typing import Literal

RingType = Literal["standard", "conjugate_invariant"]

class Parameters:
    def __init__(
        self,
        logn: int,
        logq: list[int],
        logp: list[int],
        log_default_scale: int,
        ring_type: RingType,
        *,
        h: int = 0,
        log_nth_root: int = 0,
    ):
        self._handle = ffi.new_ckks_params(
            logn, logq, logp, log_default_scale, h, ring_type, log_nth_root
        )
```

`ring_type` is required (no default). ConjugateInvariant gives 2x slots but panics with
bootstrap; Standard is bootstrap-compatible but halves slots. A thin binding should not
hide this decision. `orion_compiler.CKKSParams` can add compile-time validation that
rejects CI + bootstrap combinations with a clear error.

**JS wrapper** — same pattern:

```typescript
type RingType = "standard" | "conjugate_invariant";

class CKKSParameters {
  constructor(opts: {
    logN: number;
    logQ: number[];
    logP: number[];
    logDefaultScale: number;
    ringType: RingType;
    h?: number;
    logNthRoot?: number;
  }) {
    // bridge flattens internally
  }
}
```

### 2. Polynomials: inline Lattigo calls, expose interval

```go
//export NewPolynomialMonomial
func NewPolynomialMonomial(coeffs *C.double, n C.int, errOut **C.char) C.uintptr_t {
    goCoeffs := cDoublesToGoFloat64s(coeffs, n)
    poly := bignum.NewPolynomial(bignum.Monomial, goCoeffs, nil)
    return C.uintptr_t(cgo.NewHandle(&poly))
}

//export NewPolynomialChebyshev
func NewPolynomialChebyshev(
    coeffs *C.double, n C.int,
    intervalA C.double, intervalB C.double,
    errOut **C.char,
) C.uintptr_t {
    goCoeffs := cDoublesToGoFloat64s(coeffs, n)
    poly := bignum.NewPolynomial(
        bignum.Chebyshev, goCoeffs, [2]float64{float64(intervalA), float64(intervalB)},
    )
    return C.uintptr_t(cgo.NewHandle(&poly))
}
```

Chebyshev takes explicit interval (Lattigo API accepts it), not hardcoded `[-1, 1]`.

### 3. Minimax: expose raw Lattigo function, no Orion logic

```go
//export GenMinimaxCompositePolynomial
func GenMinimaxCompositePolynomial(
    prec C.uint,
    logAlpha C.int, logErr C.int,
    degreesPtr *C.int, numDegrees C.int,
    debug C.int,
    outCoeffs **C.double, outLen *C.int,
    outSeps **C.int, outNumPolys *C.int,
    errOut **C.char,
)
```

Returns raw `big.Float` coefficients converted to `float64`, with separator indices so
caller can split per polynomial. No caching, no sign→\[0,1\] rescaling.

### 4. Bootstrap params: flat args (align Python with JS)

```go
//export NewBootstrapParams
func NewBootstrapParams(
    paramsH C.uintptr_t,
    logn C.int,
    logpPtr *C.int, logpLen C.int,
    h C.int,
    logSlots C.int,
    errOut **C.char,
) C.uintptr_t
```

Python bridge currently lacks bootstrap support — add it for parity with JS.

### 5. Remove orion dependency from Go bridge

After all above, `python/lattigo/bridge/go.mod` drops `require`/`replace` for
`github.com/baahl-nyu/orion`. Depends only on `github.com/baahl-nyu/lattigo/v6`.

## Orion-specific logic moves to `orion_compiler`

| Logic                           | Currently in           | Moves to                                             |
| ------------------------------- | ---------------------- | ---------------------------------------------------- |
| `orion.Params` struct           | `params.go` (root)     | Deleted. Each wrapper has native `Parameters` class  |
| `CKKSParams` → bridge args      | Implicit (same JSON)   | `orion_compiler/params.py` — `CKKSParams` translates |
| Minimax sign→\[0,1\] rescaling  | `minimax.go` (root)    | `orion_compiler/core/compiler_backend.py`            |
| Minimax caching                 | `minimax.go` (root)    | `orion_compiler/core/compiler_backend.py`            |
| `orion.Polynomial` wrapper type | `polynomial.go` (root) | Deleted. Bridge stores `bignum.Polynomial` directly  |

## Parity checklist

Both bridges expose the same Lattigo surface area. CGO exports use `PascalCase`, WASM
exports use `camelCase` — same functions, different casing.

| Lattigo API                                  | CGO export (Python)                 | WASM export (JS)                    |
| -------------------------------------------- | ----------------------------------- | ----------------------------------- |
| `ckks.NewParametersFromLiteral`              | `NewCKKSParams`                     | `newCKKSParams`                     |
| `ckks.Parameters` accessors                  | `CKKSParamsMaxSlots` etc.           | `ckksMaxSlots` etc.                 |
| `rlwe.NewKeyGenerator`                       | `NewKeyGenerator`                   | `newKeyGenerator`                   |
| `rlwe.KeyGenerator.GenSecretKeyNew`          | `KeyGenGenSecretKey`                | `keyGenGenSecretKey`                |
| `rlwe.KeyGenerator.GenPublicKeyNew`          | `KeyGenGenPublicKey`                | `keyGenGenPublicKey`                |
| `rlwe.KeyGenerator.GenRelinKeyNew`           | `KeyGenGenRelinKey`                 | `keyGenGenRelinKey`                 |
| `rlwe.KeyGenerator.GenGaloisKeyNew`          | `KeyGenGenGaloisKey`                | `keyGenGenGaloisKey`                |
| `ckks.NewEncoder`                            | `NewEncoder`                        | `newEncoder`                        |
| `ckks.Encoder.Encode`                        | `EncoderEncode`                     | `encoderEncode`                     |
| `ckks.Encoder.Decode`                        | `EncoderDecode`                     | `encoderDecode`                     |
| `ckks.NewEncryptor`                          | `NewEncryptor`                      | `newEncryptor`                      |
| `rlwe.Encryptor.Encrypt`                     | `EncryptorEncryptNew`               | `encryptorEncryptNew`               |
| `ckks.NewDecryptor`                          | `NewDecryptor`                      | `newDecryptor`                      |
| `rlwe.Decryptor.Decrypt`                     | `DecryptorDecryptNew`               | `decryptorDecryptNew`               |
| `rlwe.SecretKey` marshal/unmarshal           | `SecretKeyMarshal/Unmarshal`        | `secretKeyMarshal/Unmarshal`        |
| `rlwe.PublicKey` marshal/unmarshal           | `PublicKeyMarshal/Unmarshal`        | `publicKeyMarshal/Unmarshal`        |
| `rlwe.RelinearizationKey` marshal/unmarshal  | `RelinKeyMarshal/Unmarshal`         | `relinKeyMarshal/Unmarshal`         |
| `rlwe.GaloisKey` marshal/unmarshal           | `GaloisKeyMarshal/Unmarshal`        | `galoisKeyMarshal/Unmarshal`        |
| `rlwe.Ciphertext` marshal/unmarshal/level    | `CiphertextMarshal/Unmarshal/Level` | `ciphertextMarshal/Unmarshal/Level` |
| `rlwe.Plaintext` marshal/unmarshal/level     | `PlaintextMarshal/Unmarshal/Level`  | `plaintextMarshal/Unmarshal/Level`  |
| `rlwe.NewMemEvaluationKeySet`                | `NewMemEvalKeySet`                  | `newMemEvalKeySet`                  |
| `rlwe.MemEvaluationKeySet` marshal/unmarshal | `MemEvalKeySetMarshal/Unmarshal`    | `memEvalKeySetMarshal/Unmarshal`    |
| `bootstrapping.NewParametersFromLiteral`     | `NewBootstrapParams`                | `newBootstrapParams`                |
| `bootstrapping.GenEvaluationKeys`            | `BootstrapParamsGenEvalKeys`        | `bootstrapParamsGenEvalKeys`        |
| `bootstrapping.EvaluationKeys` marshal       | `BootstrapEvalKeysMarshal`          | `bootstrapEvalKeysMarshal`          |
| `bignum.NewPolynomial` (Monomial)            | `NewPolynomialMonomial`             | `newPolynomialMonomial`             |
| `bignum.NewPolynomial` (Chebyshev)           | `NewPolynomialChebyshev`            | `newPolynomialChebyshev`            |
| `minimax.GenMinimaxCompositePolynomial`      | `GenMinimaxCompositePolynomial`     | `genMinimaxCompositePolynomial`     |
| Handle lifecycle                             | `DeleteHandle`                      | `deleteHandle`                      |

## Not in scope

- `evaluator/model.go` and `python/orion-evaluator/bridge/evaluator.go` also import
  `orion.Params` / `orion.Manifest`. Separate refactor.
- Root `params.go`, `polynomial.go`, `minimax.go` — can be deleted once evaluator also
  stops importing them.

## Files changed

- `python/lattigo/bridge/lattigo.go` — flat args, drop `CKKS` prefix from Encoder/Encryptor/Decryptor, drop redundant params arg from encrypt/decrypt, remove `orion` import
- `python/lattigo/bridge/types.go` — inline Lattigo calls, raw minimax, rename to `NewPolynomial*`
- `python/lattigo/bridge/go.mod` — remove orion dependency
- `python/lattigo/lattigo/ffi.py` — update FFI signatures to match renamed exports
- `python/lattigo/lattigo/ckks.py` — `Parameters.__init__` takes plain args, `Encoder.__init__` takes params
- `python/lattigo/lattigo/rlwe.py` — `KeyGenerator.__init__` takes params, etc.
- `python/orion-compiler/orion_compiler/params.py` — `CKKSParams` → bridge args translation
- `python/orion-compiler/orion_compiler/core/compiler_backend.py` — absorb minimax rescaling + caching
- `js/lattigo/bridge/params.go` — flat args (drop JSON)
- `js/lattigo/bridge/` — add polynomial + minimax bridge functions, rename bootstrap exports for consistency
- `js/lattigo/src/ckks.ts` — `CKKSParameters` constructor takes plain args
- `js/lattigo/src/` — add polynomial wrapper types
- `js/lattigo/src/types.ts` — update `WasmBridge` interface for renamed/added functions
