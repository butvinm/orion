# orion-v2-lattigo

Python bindings for [Lattigo](https://github.com/tuneinsight/lattigo) CKKS homomorphic encryption primitives.

Part of the [Orion](https://github.com/butvinm/orion) FHE framework.

## Modules

- `lattigo.ckks` — `Parameters`, `Encoder`
- `lattigo.rlwe` — `SecretKey`, `PublicKey`, `RelinearizationKey`, `GaloisKey`, `Ciphertext`, `Plaintext`, `KeyGenerator`, `Encryptor`, `Decryptor`, `MemEvaluationKeySet`
- `lattigo.gohandle` — `GoHandle` RAII wrapper for cgo.Handle values

## Requirements

Requires the CGO shared library built from the Go bridge. See the [main repo](https://github.com/butvinm/orion) for build instructions.
