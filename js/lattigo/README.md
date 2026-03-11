# @orion/lattigo

TypeScript bindings for Lattigo's CKKS operations via Go WASM. Enables browser-side key generation, encryption, and decryption — the secret key never leaves the client.

## Prerequisites

The WASM binary must be built before use:

```bash
python tools/build_lattigo_wasm.py
```

This compiles `js/lattigo/bridge/` to `js/lattigo/wasm/lattigo.wasm` and copies `wasm_exec.js`.

## Installation

```bash
cd js/lattigo
npm install
npm run build
```

## Usage

### Node.js

```typescript
import {
  loadLattigo,
  CKKSParameters,
  KeyGenerator,
  Encoder,
  Encryptor,
  Decryptor,
} from "@orion/lattigo";

// Load the WASM bridge — required before calling any API
const bridge = await loadLattigo("wasm/lattigo.wasm");

// Create CKKS parameters
const params = CKKSParameters.fromLogn({
  logN: 14,
  logQ: [55, 40, 40, 40, 40, 40, 40, 40],
  logP: [55, 55],
  logDefaultScale: 40,
});

// Key generation
const kg = KeyGenerator.new(params);
const sk = kg.genSecretKey();
const pk = kg.genPublicKey(sk);

// Encode → encrypt → decrypt → decode
const encoder = Encoder.new(params);
const encryptor = Encryptor.new(params, pk);
const decryptor = Decryptor.new(params, sk);

const values = new Array(params.maxSlots()).fill(0.5);
const pt = encoder.encode(values, params.maxLevel(), params.defaultScale());
const ct = encryptor.encryptNew(pt);
const decPt = decryptor.decryptNew(ct);
const output = encoder.decode(decPt, params.maxSlots());

// Release handles when done
ct.close();
pt.close();
decPt.close();
encoder.close();
encryptor.close();
decryptor.close();
kg.close();
sk.close();
pk.close();
params.close();
```

### Browser

```html
<script src="wasm/wasm_exec.js"></script>
<script type="module">
  import { loadLattigo, CKKSParameters } from "./dist/index.js";

  // WASM loads from the URL "wasm/lattigo.wasm" by default in the browser
  const bridge = await loadLattigo();
  const params = CKKSParameters.fromLogn({ logN: 14, ... });
  // ...
</script>
```

## Memory Management

Every class has a `.close()` method that releases the underlying Go handle. Call it when done with an object.

```typescript
const sk = kg.genSecretKey();
// ... use sk ...
sk.close(); // explicit cleanup
```

A `FinalizationRegistry` is registered as a safety net to catch forgotten `.close()` calls, but GC timing is not deterministic — prefer explicit cleanup for long-lived sessions.

## API

- `loadLattigo(wasmPath?)` — loads the WASM bridge. Must be called first.
- `CKKSParameters` — CKKS scheme parameters. `fromLogn()`, `fromJSON()`, accessors.
- `KeyGenerator` — generates secret, public, relinearization, and Galois keys.
- `Encoder` — encodes/decodes float64 values to/from plaintexts.
- `Encryptor` — encrypts plaintexts to ciphertexts.
- `Decryptor` — decrypts ciphertexts to plaintexts.
- `MemEvaluationKeySet` — bundles RLK + Galois keys for serialization.
- Key types: `SecretKey`, `PublicKey`, `RelinearizationKey`, `GaloisKey`, `Ciphertext`, `Plaintext`.

All key and ciphertext types support `.marshalBinary()` / `static unmarshalBinary(bytes)` for Lattigo-compatible serialization.

## Examples

See `js/examples/node/` for full usage examples:

- `roundtrip.ts` — keygen, encode, encrypt, decrypt, decode with timing
- `eval-keys.ts` — full key generation from a `KeyManifest` (RLK + Galois keys + bootstrap keys)

## Tests

```bash
npm test
```

Tests require the WASM binary to be built first.
