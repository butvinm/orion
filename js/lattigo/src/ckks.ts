import type { Handle } from "./types.js";
import { isError } from "./types.js";
import { getBridge, registry } from "./bridge.js";

/** CKKS scheme parameters. Wraps Lattigo's ckks.Parameters. */
export class CKKSParameters {
  private _handle: Handle;

  constructor(handle: Handle) {
    this._handle = handle;
    registry.register(this, handle, this);
  }

  /** @internal Access the raw handle ID. */
  get handle(): Handle {
    return this._handle;
  }

  /** Create CKKS parameters from log-scale specifications. */
  static fromLogn(opts: {
    logN: number;
    logQ: number[];
    logP: number[];
    logDefaultScale: number;
    h?: number;
    ringType?: string;
  }): CKKSParameters {
    const json = JSON.stringify({
      LogN: opts.logN,
      LogQ: opts.logQ,
      LogP: opts.logP,
      LogDefaultScale: opts.logDefaultScale,
      H: opts.h ?? 192,
      RingType: opts.ringType ?? "ConjugateInvariant",
    });
    return CKKSParameters.fromJSON(json);
  }

  /** Create from a JSON string (Lattigo parameter format). */
  static fromJSON(paramsJson: string): CKKSParameters {
    const result = getBridge().newCKKSParams(paramsJson);
    if (isError(result)) {
      throw new Error(`Failed to create CKKS params: ${result.error}`);
    }
    return new CKKSParameters(result.handle);
  }

  /** Maximum number of plaintext slots. */
  maxSlots(): number {
    const v = getBridge().ckksMaxSlots(this._handle);
    if (v === null) throw new Error("Invalid params handle");
    return v;
  }

  /** Maximum ciphertext level (len(logQ) - 1). */
  maxLevel(): number {
    const v = getBridge().ckksMaxLevel(this._handle);
    if (v === null) throw new Error("Invalid params handle");
    return v;
  }

  /** Default scale (float64, typically 2^logDefaultScale). */
  defaultScale(): number {
    const v = getBridge().ckksDefaultScale(this._handle);
    if (v === null) throw new Error("Invalid params handle");
    return v;
  }

  /** Galois element for a given rotation step. */
  galoisElement(rotation: number): number {
    const v = getBridge().ckksGaloisElement(this._handle, rotation);
    if (v === null) throw new Error("Invalid params handle");
    return v;
  }

  /** Q primes (ciphertext moduli chain). */
  moduliChain(): number[] {
    const v = getBridge().ckksModuliChain(this._handle);
    if (v === null) throw new Error("Invalid params handle");
    return v;
  }

  /** P primes (auxiliary moduli chain for key switching). */
  auxModuliChain(): number[] {
    const v = getBridge().ckksAuxModuliChain(this._handle);
    if (v === null) throw new Error("Invalid params handle");
    return v;
  }

  /** Free the underlying Go handle. Idempotent. */
  free(): void {
    if (this._handle !== 0) {
      registry.unregister(this);
      getBridge().deleteHandle(this._handle);
      this._handle = 0;
    }
  }
}
