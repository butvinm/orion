import type { Handle, RingType } from "./types.js";
import { isError } from "./types.js";
import { getBridge, registry } from "./bridge.js";

/** CKKS scheme parameters. Wraps Lattigo's ckks.Parameters. */
export class CKKSParameters {
  private _handle: Handle;

  constructor(opts: {
    logN: number;
    logQ: number[];
    logP: number[];
    logDefaultScale: number;
    ringType: RingType;
    h?: number;
    logNthRoot?: number;
  }) {
    const result = getBridge().newCKKSParams(
      opts.logN,
      opts.logQ,
      opts.logP,
      opts.logDefaultScale,
      opts.ringType,
      opts.h,
      opts.logNthRoot,
    );
    if (isError(result)) {
      throw new Error(`Failed to create CKKS params: ${result.error}`);
    }
    this._handle = result.handle;
    registry.register(this, this._handle, this);
  }

  /** @internal Wrap an existing handle (for deserialization etc). */
  static _fromHandle(handle: Handle): CKKSParameters {
    const obj = Object.create(CKKSParameters.prototype) as CKKSParameters;
    obj._handle = handle;
    registry.register(obj, handle, obj);
    return obj;
  }

  /** @internal Access the raw handle ID. */
  get handle(): Handle {
    return this._handle;
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
