# Experiment 04: Library Modifications

All changes made to support server-side evaluator construction from imported keys via FFI.

## Go Changes

### `orion/backend/lattigo/keygenerator.go`

- **Added `SerializePublicKey()`**: Serializes `scheme.PublicKey` via `MarshalBinary`, returns byte array. Mirrors existing `SerializeSecretKey` pattern.
- **Added `LoadPublicKey(dataPtr, lenData)`**: Deserializes a public key via `UnmarshalBinary`, sets `scheme.PublicKey`. Creates a properly-sized key via `rlwe.NewPublicKey(scheme.Params)` before unmarshaling.
- **Added `SerializeRelinKey()`**: Serializes `scheme.RelinKey` via `MarshalBinary`, returns byte array.
- **Added `LoadRelinKey(dataPtr, lenData)`**: Deserializes a relinearization key via `UnmarshalBinary`, sets `scheme.RelinKey`.

### `orion/backend/lattigo/evaluator.go`

- **Added `NewEvaluatorFromKeys()`**: Constructs a `ckks.Evaluator` from `scheme.EvalKeys` without calling `AddPo2RotationKeys()`. This is the server-side constructor — all required galois keys must already be loaded into `scheme.EvalKeys` via `LoadRotationKey` before calling this.
- **Modified `Rotate()`**: No longer unconditionally calls `AddRotationKey()`. Now checks `scheme.KeyGen != nil && scheme.SecretKey != nil` first. On the server side (where sk is nil), rotation with a missing galois key causes an explicit Lattigo panic: `"cannot Rotate: cannot apply Automorphism: GaloisKey[X] is nil"`. This prevents silent fallback to key generation.
- **Modified `RotateNew()`**: Same change as `Rotate()` — conditional `AddRotationKey()` call.

### `orion/backend/lattigo/bootstrapper.go`

- **Added `SerializeBootstrapKeys(numSlots, LogPs, lenLogPs)`**: Generates bootstrap evaluation keys from `scheme.SecretKey` and serializes them via `MarshalBinary`. Returns the full serialized bootstrap key bundle.
- **Added `LoadBootstrapKeys(dataPtr, lenData, numSlots, LogPs, lenLogPs)`**: Deserializes bootstrap evaluation keys, creates bootstrap parameters, constructs a `bootstrapping.Evaluator`, and stores it in `bootstrapperMap`. This is the server-side bootstrap key import path.

### `orion/backend/lattigo/tensors.go`

- **Added `SerializeCiphertext(ciphertextID)`**: Serializes a ciphertext from the heap via `MarshalBinary`, returns byte array. Needed for ciphertext transfer between client and server.
- **Added `LoadCiphertext(dataPtr, lenData)`**: Deserializes a ciphertext via `UnmarshalBinary`, pushes to heap, returns new ID.

## Python Changes

### `orion/backend/lattigo/bindings.py`

- **Added `SerializePublicKey`**: FFI binding, no args, returns `ArrayResultByte`.
- **Added `LoadPublicKey`**: FFI binding, takes `(POINTER(c_ubyte), c_ulong)`.
- **Added `SerializeRelinKey`**: FFI binding, no args, returns `ArrayResultByte`.
- **Added `LoadRelinKey`**: FFI binding, takes `(POINTER(c_ubyte), c_ulong)`.
- **Added `NewEvaluatorFromKeys`**: FFI binding, no args, no return.
- **Added `SerializeBootstrapKeys`**: FFI binding, takes `(c_int, POINTER(c_int), c_int)`, returns `ArrayResultByte`.
- **Added `LoadBootstrapKeys`**: FFI binding, takes `(POINTER(c_ubyte), c_ulong, c_int, POINTER(c_int), c_int)`.
- **Added `SerializeCiphertext`**: FFI binding, takes `(c_int)`, returns `ArrayResultByte`.
- **Added `LoadCiphertext`**: FFI binding, takes `(POINTER(c_ubyte), c_ulong)`, returns `c_int`.

## Server-side Key Import Workflow

```
1. NewScheme(params)           -- creates scheme with Params + KeyGen, no keys
2. LoadRelinKey(rlk_data)      -- sets scheme.RelinKey
3. GenerateEvaluationKeys()    -- creates scheme.EvalKeys from rlk
4. LoadRotationKey(gk_data, galEl)  -- for each galois key
5. NewEvaluatorFromKeys()      -- creates evaluator from loaded EvalKeys
6. LoadCiphertext(ct_data)     -- load ciphertext from client
7. MulRelinCiphertextNew(...)  -- evaluate on server
8. RotateNew(...)              -- uses pre-loaded galois key (no lazy gen)
9. SerializeCiphertext(ct_id)  -- serialize result for client
```

## Key Sizes (LogN=13, ConjugateInvariant)

| Key Type | Size |
|----------|------|
| Secret Key (sk) | 0.50 MB |
| Public Key (pk) | 1.00 MB |
| Relinearization Key (rlk) | 3.00 MB |
| Galois Key (each) | 3.00 MB |
| Ciphertext | ~768 KB |

## Backwards Compatibility

All changes are backwards compatible. The `Rotate`/`RotateNew` modification uses a runtime check (`scheme.KeyGen != nil && scheme.SecretKey != nil`) to decide whether to lazily generate keys. Existing code that initializes keys normally is unaffected — all 9 existing tests pass.
