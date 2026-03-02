import type { Handle } from "./types.js";
import { isError } from "./types.js";
import { getBridge, registry } from "./bridge.js";
import type { CKKSParameters } from "./ckks.js";

// =========================================================================
// Key types
// =========================================================================

/** Wraps Lattigo's rlwe.SecretKey. */
export class SecretKey {
  private _handle: Handle;

  constructor(handle: Handle) {
    this._handle = handle;
    registry.register(this, handle, this);
  }

  get handle(): Handle {
    return this._handle;
  }

  marshalBinary(): Uint8Array {
    const result = getBridge().secretKeyMarshal(this._handle);
    if (isError(result)) throw new Error(`SecretKey marshal: ${result.error}`);
    return result;
  }

  static unmarshalBinary(bytes: Uint8Array): SecretKey {
    const result = getBridge().secretKeyUnmarshal(bytes);
    if (isError(result))
      throw new Error(`SecretKey unmarshal: ${result.error}`);
    return new SecretKey(result.handle);
  }

  free(): void {
    if (this._handle !== 0) {
      registry.unregister(this);
      getBridge().deleteHandle(this._handle);
      this._handle = 0;
    }
  }
}

/** Wraps Lattigo's rlwe.PublicKey. */
export class PublicKey {
  private _handle: Handle;

  constructor(handle: Handle) {
    this._handle = handle;
    registry.register(this, handle, this);
  }

  get handle(): Handle {
    return this._handle;
  }

  marshalBinary(): Uint8Array {
    const result = getBridge().publicKeyMarshal(this._handle);
    if (isError(result)) throw new Error(`PublicKey marshal: ${result.error}`);
    return result;
  }

  static unmarshalBinary(bytes: Uint8Array): PublicKey {
    const result = getBridge().publicKeyUnmarshal(bytes);
    if (isError(result))
      throw new Error(`PublicKey unmarshal: ${result.error}`);
    return new PublicKey(result.handle);
  }

  free(): void {
    if (this._handle !== 0) {
      registry.unregister(this);
      getBridge().deleteHandle(this._handle);
      this._handle = 0;
    }
  }
}

/** Wraps Lattigo's rlwe.RelinearizationKey. */
export class RelinearizationKey {
  private _handle: Handle;

  constructor(handle: Handle) {
    this._handle = handle;
    registry.register(this, handle, this);
  }

  get handle(): Handle {
    return this._handle;
  }

  marshalBinary(): Uint8Array {
    const result = getBridge().relinKeyMarshal(this._handle);
    if (isError(result))
      throw new Error(`RelinearizationKey marshal: ${result.error}`);
    return result;
  }

  static unmarshalBinary(bytes: Uint8Array): RelinearizationKey {
    const result = getBridge().relinKeyUnmarshal(bytes);
    if (isError(result))
      throw new Error(`RelinearizationKey unmarshal: ${result.error}`);
    return new RelinearizationKey(result.handle);
  }

  free(): void {
    if (this._handle !== 0) {
      registry.unregister(this);
      getBridge().deleteHandle(this._handle);
      this._handle = 0;
    }
  }
}

/** Wraps Lattigo's rlwe.GaloisKey. */
export class GaloisKey {
  private _handle: Handle;

  constructor(handle: Handle) {
    this._handle = handle;
    registry.register(this, handle, this);
  }

  get handle(): Handle {
    return this._handle;
  }

  marshalBinary(): Uint8Array {
    const result = getBridge().galoisKeyMarshal(this._handle);
    if (isError(result)) throw new Error(`GaloisKey marshal: ${result.error}`);
    return result;
  }

  static unmarshalBinary(bytes: Uint8Array): GaloisKey {
    const result = getBridge().galoisKeyUnmarshal(bytes);
    if (isError(result))
      throw new Error(`GaloisKey unmarshal: ${result.error}`);
    return new GaloisKey(result.handle);
  }

  free(): void {
    if (this._handle !== 0) {
      registry.unregister(this);
      getBridge().deleteHandle(this._handle);
      this._handle = 0;
    }
  }
}

// =========================================================================
// Ciphertext / Plaintext
// =========================================================================

/** Wraps a raw Lattigo rlwe.Ciphertext. */
export class Ciphertext {
  private _handle: Handle;

  constructor(handle: Handle) {
    this._handle = handle;
    registry.register(this, handle, this);
  }

  get handle(): Handle {
    return this._handle;
  }

  level(): number {
    const result = getBridge().ciphertextLevel(this._handle);
    if (isError(result))
      throw new Error(`Ciphertext level: ${result.error}`);
    return result;
  }

  marshalBinary(): Uint8Array {
    const result = getBridge().ciphertextMarshal(this._handle);
    if (isError(result))
      throw new Error(`Ciphertext marshal: ${result.error}`);
    return result;
  }

  static unmarshalBinary(bytes: Uint8Array): Ciphertext {
    const result = getBridge().ciphertextUnmarshal(bytes);
    if (isError(result))
      throw new Error(`Ciphertext unmarshal: ${result.error}`);
    return new Ciphertext(result.handle);
  }

  free(): void {
    if (this._handle !== 0) {
      registry.unregister(this);
      getBridge().deleteHandle(this._handle);
      this._handle = 0;
    }
  }
}

/** Wraps a raw Lattigo rlwe.Plaintext. */
export class Plaintext {
  private _handle: Handle;

  constructor(handle: Handle) {
    this._handle = handle;
    registry.register(this, handle, this);
  }

  get handle(): Handle {
    return this._handle;
  }

  level(): number {
    const result = getBridge().plaintextLevel(this._handle);
    if (isError(result))
      throw new Error(`Plaintext level: ${result.error}`);
    return result;
  }

  marshalBinary(): Uint8Array {
    const result = getBridge().plaintextMarshal(this._handle);
    if (isError(result))
      throw new Error(`Plaintext marshal: ${result.error}`);
    return result;
  }

  static unmarshalBinary(bytes: Uint8Array): Plaintext {
    const result = getBridge().plaintextUnmarshal(bytes);
    if (isError(result))
      throw new Error(`Plaintext unmarshal: ${result.error}`);
    return new Plaintext(result.handle);
  }

  free(): void {
    if (this._handle !== 0) {
      registry.unregister(this);
      getBridge().deleteHandle(this._handle);
      this._handle = 0;
    }
  }
}

// =========================================================================
// KeyGenerator
// =========================================================================

/** Wraps Lattigo's rlwe.KeyGenerator. */
export class KeyGenerator {
  private _handle: Handle;

  constructor(handle: Handle) {
    this._handle = handle;
    registry.register(this, handle, this);
  }

  get handle(): Handle {
    return this._handle;
  }

  static new(params: CKKSParameters): KeyGenerator {
    const result = getBridge().newKeyGenerator(params.handle);
    if (isError(result))
      throw new Error(`KeyGenerator create: ${result.error}`);
    return new KeyGenerator(result.handle);
  }

  genSecretKey(): SecretKey {
    const result = getBridge().keyGenGenSecretKey(this._handle);
    if (isError(result))
      throw new Error(`genSecretKey: ${result.error}`);
    return new SecretKey(result.handle);
  }

  genPublicKey(sk: SecretKey): PublicKey {
    const result = getBridge().keyGenGenPublicKey(this._handle, sk.handle);
    if (isError(result))
      throw new Error(`genPublicKey: ${result.error}`);
    return new PublicKey(result.handle);
  }

  genRelinKey(sk: SecretKey): RelinearizationKey {
    const result = getBridge().keyGenGenRelinKey(this._handle, sk.handle);
    if (isError(result))
      throw new Error(`genRelinKey: ${result.error}`);
    return new RelinearizationKey(result.handle);
  }

  genGaloisKey(sk: SecretKey, galoisElement: number): GaloisKey {
    const result = getBridge().keyGenGenGaloisKey(
      this._handle,
      sk.handle,
      galoisElement,
    );
    if (isError(result))
      throw new Error(`genGaloisKey: ${result.error}`);
    return new GaloisKey(result.handle);
  }

  free(): void {
    if (this._handle !== 0) {
      registry.unregister(this);
      getBridge().deleteHandle(this._handle);
      this._handle = 0;
    }
  }
}

// =========================================================================
// Encryptor / Decryptor
// =========================================================================

/** Wraps Lattigo's rlwe.Encryptor (public-key encryption). */
export class Encryptor {
  private _handle: Handle;

  constructor(handle: Handle) {
    this._handle = handle;
    registry.register(this, handle, this);
  }

  get handle(): Handle {
    return this._handle;
  }

  static new(params: CKKSParameters, pk: PublicKey): Encryptor {
    const result = getBridge().newEncryptor(params.handle, pk.handle);
    if (isError(result))
      throw new Error(`Encryptor create: ${result.error}`);
    return new Encryptor(result.handle);
  }

  encryptNew(pt: Plaintext): Ciphertext {
    const result = getBridge().encryptorEncryptNew(this._handle, pt.handle);
    if (isError(result))
      throw new Error(`encryptNew: ${result.error}`);
    return new Ciphertext(result.handle);
  }

  free(): void {
    if (this._handle !== 0) {
      registry.unregister(this);
      getBridge().deleteHandle(this._handle);
      this._handle = 0;
    }
  }
}

/** Wraps Lattigo's rlwe.Decryptor. */
export class Decryptor {
  private _handle: Handle;

  constructor(handle: Handle) {
    this._handle = handle;
    registry.register(this, handle, this);
  }

  get handle(): Handle {
    return this._handle;
  }

  static new(params: CKKSParameters, sk: SecretKey): Decryptor {
    const result = getBridge().newDecryptor(params.handle, sk.handle);
    if (isError(result))
      throw new Error(`Decryptor create: ${result.error}`);
    return new Decryptor(result.handle);
  }

  decryptNew(ct: Ciphertext): Plaintext {
    const result = getBridge().decryptorDecryptNew(this._handle, ct.handle);
    if (isError(result))
      throw new Error(`decryptNew: ${result.error}`);
    return new Plaintext(result.handle);
  }

  free(): void {
    if (this._handle !== 0) {
      registry.unregister(this);
      getBridge().deleteHandle(this._handle);
      this._handle = 0;
    }
  }
}

// =========================================================================
// MemEvaluationKeySet
// =========================================================================

/** Wraps Lattigo's rlwe.MemEvaluationKeySet. */
export class MemEvaluationKeySet {
  private _handle: Handle;

  constructor(handle: Handle) {
    this._handle = handle;
    registry.register(this, handle, this);
  }

  get handle(): Handle {
    return this._handle;
  }

  /**
   * Create a MemEvaluationKeySet from optional RLK + Galois keys.
   * Pass null for rlk if no relinearization key is needed.
   */
  static new(
    rlk: RelinearizationKey | null,
    galoisKeys: GaloisKey[] = [],
  ): MemEvaluationKeySet {
    const result = getBridge().newMemEvalKeySet(
      rlk?.handle ?? null,
      galoisKeys.map((gk) => gk.handle),
    );
    if (isError(result))
      throw new Error(`MemEvaluationKeySet create: ${result.error}`);
    return new MemEvaluationKeySet(result.handle);
  }

  marshalBinary(): Uint8Array {
    const result = getBridge().memEvalKeySetMarshal(this._handle);
    if (isError(result))
      throw new Error(`MemEvaluationKeySet marshal: ${result.error}`);
    return result;
  }

  static unmarshalBinary(bytes: Uint8Array): MemEvaluationKeySet {
    const result = getBridge().memEvalKeySetUnmarshal(bytes);
    if (isError(result))
      throw new Error(`MemEvaluationKeySet unmarshal: ${result.error}`);
    return new MemEvaluationKeySet(result.handle);
  }

  free(): void {
    if (this._handle !== 0) {
      registry.unregister(this);
      getBridge().deleteHandle(this._handle);
      this._handle = 0;
    }
  }
}
