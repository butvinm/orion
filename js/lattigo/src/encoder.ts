import { isError } from "./types.js";
import { getBridge, registry } from "./bridge.js";
import type { CKKSParameters } from "./ckks.js";
import { Plaintext } from "./rlwe.js";
import type { Handle } from "./types.js";

/** CKKS encoder. Wraps Lattigo's ckks.Encoder. */
export class Encoder {
  private _handle: Handle;

  constructor(params: CKKSParameters) {
    const result = getBridge().newEncoder(params.handle);
    if (isError(result))
      throw new Error(`Encoder create: ${result.error}`);
    this._handle = result.handle;
    registry.register(this, this._handle, this);
  }

  /** @internal Wrap an existing handle. */
  static _fromHandle(handle: Handle): Encoder {
    const obj = Object.create(Encoder.prototype) as Encoder;
    obj._handle = handle;
    registry.register(obj, handle, obj);
    return obj;
  }

  get handle(): Handle {
    return this._handle;
  }

  /** Encode float64 values into a Plaintext at given level and scale. */
  encode(values: Float64Array | number[], level: number, scale: number): Plaintext {
    const result = getBridge().encoderEncode(this._handle, values, level, scale);
    if (isError(result))
      throw new Error(`encode: ${result.error}`);
    return new Plaintext(result.handle);
  }

  /** Decode a Plaintext back to float64 values. */
  decode(plaintext: Plaintext, numSlots: number): Float64Array {
    const result = getBridge().encoderDecode(this._handle, plaintext.handle, numSlots);
    if (isError(result))
      throw new Error(`decode: ${result.error}`);
    return result;
  }

  /** Release the underlying Go handle. Idempotent. */
  close(): void {
    if (this._handle !== 0) {
      registry.unregister(this);
      getBridge().deleteHandle(this._handle);
      this._handle = 0;
    }
  }
}
