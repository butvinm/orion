/** Opaque numeric ID referencing a Go-side object in the handle map. */
export type Handle = number;

/** Returned by bridge functions that create a Go-side object. */
export interface HandleResult {
  handle: Handle;
}

/** Returned by bridge functions on failure. */
export interface ErrorResult {
  error: string;
}

/** Ring type for CKKS parameters. */
export type RingType = "standard" | "conjugate_invariant";

/**
 * Typed interface for all functions registered on `globalThis.lattigo`
 * by the Go WASM bridge (js/lattigo/bridge/main.go).
 *
 * Return types mirror Go's dynamic `any` returns:
 * - Object-creating functions return HandleResult | ErrorResult
 * - Accessor functions return primitives directly (or null on bad handle)
 * - Marshal functions return Uint8Array | ErrorResult
 * - encoderDecode returns Float64Array | ErrorResult
 * - bootstrapParamsGenEvalKeys returns Promise
 */
export interface WasmBridge {
  /** Readiness signal — true once all functions are registered. */
  __ready: boolean;

  // --- Handle lifecycle ---

  /** Free a Go-side handle. Idempotent (no-op if already deleted). */
  deleteHandle(handleID: Handle): void;

  // --- CKKS Parameters ---

  /** Create CKKS parameters from flat arguments. */
  newCKKSParams(
    logN: number,
    logQ: number[],
    logP: number[],
    logDefaultScale: number,
    ringType: string,
    h?: number,
    logNthRoot?: number,
  ): HandleResult | ErrorResult;

  /** Max number of plaintext slots for these parameters. */
  ckksMaxSlots(paramsHID: Handle): number | null;

  /** Max ciphertext level for these parameters. */
  ckksMaxLevel(paramsHID: Handle): number | null;

  /** Default scale (float64) for these parameters. */
  ckksDefaultScale(paramsHID: Handle): number | null;

  /** Galois element for a given rotation. */
  ckksGaloisElement(paramsHID: Handle, rotation: number): number | null;

  /** Q moduli chain as array of numbers (float64, safe up to 2^53). */
  ckksModuliChain(paramsHID: Handle): number[] | null;

  /** P auxiliary moduli chain as array of numbers. */
  ckksAuxModuliChain(paramsHID: Handle): number[] | null;

  // --- Key Generation ---

  /** Create a new key generator from CKKS parameters. */
  newKeyGenerator(paramsHID: Handle): HandleResult | ErrorResult;

  /** Generate a new secret key. */
  keyGenGenSecretKey(kgHID: Handle): HandleResult | ErrorResult;

  /** Generate a public key from a secret key. */
  keyGenGenPublicKey(kgHID: Handle, skHID: Handle): HandleResult | ErrorResult;

  /** Generate a relinearization key. */
  keyGenGenRelinKey(kgHID: Handle, skHID: Handle): HandleResult | ErrorResult;

  /** Generate a Galois key for a specific Galois element. */
  keyGenGenGaloisKey(
    kgHID: Handle,
    skHID: Handle,
    galoisElement: number,
  ): HandleResult | ErrorResult;

  // --- Encoder ---

  /** Create a new encoder from CKKS parameters. */
  newEncoder(paramsHID: Handle): HandleResult | ErrorResult;

  /** Encode float64 values into a plaintext at given level and scale. */
  encoderEncode(
    encHID: Handle,
    values: Float64Array | number[],
    level: number,
    scale: number,
  ): HandleResult | ErrorResult;

  /** Decode a plaintext into float64 values. */
  encoderDecode(
    encHID: Handle,
    ptHID: Handle,
    numSlots: number,
  ): Float64Array | ErrorResult;

  // --- Encryptor ---

  /** Create an encryptor from CKKS parameters and a public key. */
  newEncryptor(
    paramsHID: Handle,
    pkHID: Handle,
  ): HandleResult | ErrorResult;

  /** Encrypt a plaintext, returning a new ciphertext handle. */
  encryptorEncryptNew(
    encHID: Handle,
    ptHID: Handle,
  ): HandleResult | ErrorResult;

  // --- Decryptor ---

  /** Create a decryptor from CKKS parameters and a secret key. */
  newDecryptor(
    paramsHID: Handle,
    skHID: Handle,
  ): HandleResult | ErrorResult;

  /** Decrypt a ciphertext, returning a new plaintext handle. */
  decryptorDecryptNew(
    decHID: Handle,
    ctHID: Handle,
  ): HandleResult | ErrorResult;

  // --- Serialization: SecretKey ---

  secretKeyMarshal(skHID: Handle): Uint8Array | ErrorResult;
  secretKeyUnmarshal(bytes: Uint8Array): HandleResult | ErrorResult;

  // --- Serialization: PublicKey ---

  publicKeyMarshal(pkHID: Handle): Uint8Array | ErrorResult;
  publicKeyUnmarshal(bytes: Uint8Array): HandleResult | ErrorResult;

  // --- Serialization: RelinearizationKey ---

  relinKeyMarshal(rlkHID: Handle): Uint8Array | ErrorResult;
  relinKeyUnmarshal(bytes: Uint8Array): HandleResult | ErrorResult;

  // --- Serialization: GaloisKey ---

  galoisKeyMarshal(gkHID: Handle): Uint8Array | ErrorResult;
  galoisKeyUnmarshal(bytes: Uint8Array): HandleResult | ErrorResult;

  // --- Serialization: Ciphertext ---

  ciphertextMarshal(ctHID: Handle): Uint8Array | ErrorResult;
  ciphertextUnmarshal(bytes: Uint8Array): HandleResult | ErrorResult;
  ciphertextLevel(ctHID: Handle): number | ErrorResult;

  // --- Serialization: Plaintext ---

  plaintextMarshal(ptHID: Handle): Uint8Array | ErrorResult;
  plaintextUnmarshal(bytes: Uint8Array): HandleResult | ErrorResult;
  plaintextLevel(ptHID: Handle): number | ErrorResult;

  // --- MemEvaluationKeySet ---

  /**
   * Create a MemEvaluationKeySet from optional RLK + Galois key handles.
   * rlkHID=0 or null means no relinearization key.
   */
  newMemEvalKeySet(
    rlkHID: Handle | null,
    galoisKeyHIDs: Handle[],
  ): HandleResult | ErrorResult;

  memEvalKeySetMarshal(evkHID: Handle): Uint8Array | ErrorResult;
  memEvalKeySetUnmarshal(bytes: Uint8Array): HandleResult | ErrorResult;

  // --- Bootstrap ---

  /** Construct bootstrap parameters from flat arguments. */
  newBootstrapParams(
    paramsHID: Handle,
    logN?: number | null,
    logP?: number[] | null,
    h?: number | null,
    logSlots?: number | null,
  ): HandleResult | ErrorResult;

  /**
   * Generate bootstrap evaluation keys. Async (5-30s).
   * Returns both the MemEvaluationKeySet handle and the full bootstrap EvaluationKeys handle.
   */
  bootstrapParamsGenEvalKeys(
    btpParamsHID: Handle,
    skHID: Handle,
  ): Promise<{ evkHID: Handle; btpEvkHID: Handle }>;

  /** Marshal bootstrap evaluation keys to binary. */
  bootstrapEvalKeysMarshal(btpEvkHID: Handle): Uint8Array | ErrorResult;

  // --- Bootstrap (streaming) ---

  /** Extend SK to bootstrap ring. Returns {skN2HID, kgN2HID, needsRingSwitch, isConjugateInvariant}. */
  bootstrapExtendSK(
    btpParamsHID: Handle,
    skHID: Handle,
  ): { skN2HID: Handle; kgN2HID: Handle; needsRingSwitch: boolean; isConjugateInvariant: boolean } | ErrorResult;

  /** Get Galois elements required by the bootstrap circuit. */
  bootstrapGaloisElements(btpParamsHID: Handle): number[] | ErrorResult;

  /** Generate ring-switching and encapsulation keys. Returns {keys: [{hid, name}...]}. */
  bootstrapGenSwitchingKeys(
    btpParamsHID: Handle,
    skN1HID: Handle,
    skN2HID: Handle,
  ): { keys: Array<{ hid: Handle; name: string }> } | ErrorResult;

  /** Marshal an individual evaluation key by handle. */
  evalKeyMarshal(hid: Handle): Uint8Array | ErrorResult;

  // --- Polynomial ---

  /** Create a monomial-basis polynomial from coefficients. */
  newPolynomialMonomial(
    coeffs: Float64Array | number[],
  ): HandleResult | ErrorResult;

  /** Create a Chebyshev-basis polynomial from coefficients with explicit interval. */
  newPolynomialChebyshev(
    coeffs: Float64Array | number[],
    intervalA: number,
    intervalB: number,
  ): HandleResult | ErrorResult;

  /**
   * Generate minimax composite polynomial coefficients for sign approximation.
   * Returns raw coefficients (no caching, no sign→[0,1] rescaling).
   */
  genMinimaxCompositePolynomial(
    prec: number,
    logAlpha: number,
    logErr: number,
    degrees: number[],
  ): { coeffs: Float64Array; seps: number[] } | ErrorResult;
}

/** Type guard: check if a bridge result is an error. */
export function isError(result: unknown): result is ErrorResult {
  return (
    typeof result === "object" &&
    result !== null &&
    "error" in result &&
    typeof (result as ErrorResult).error === "string"
  );
}

/** Type guard: check if a bridge result is a handle result. */
export function isHandle(result: unknown): result is HandleResult {
  return (
    typeof result === "object" &&
    result !== null &&
    "handle" in result &&
    typeof (result as HandleResult).handle === "number"
  );
}
