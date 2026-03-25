import { describe, it, expect, beforeAll } from "vitest";
import { ensureWasmLoaded } from "./helpers.js";
import {
  SecretKey,
  PublicKey,
  RelinearizationKey,
  GaloisKey,
  Ciphertext,
  Plaintext,
  MemEvaluationKeySet,
} from "../src/index.js";

beforeAll(async () => {
  await ensureWasmLoaded();
});

describe("Error handling", () => {
  it.each([
    ["SecretKey", () => SecretKey.unmarshalBinary(new Uint8Array([1, 2, 3]))],
    ["PublicKey", () => PublicKey.unmarshalBinary(new Uint8Array([1, 2, 3]))],
    [
      "RelinearizationKey",
      () => RelinearizationKey.unmarshalBinary(new Uint8Array([1, 2, 3])),
    ],
    ["GaloisKey", () => GaloisKey.unmarshalBinary(new Uint8Array([1, 2, 3]))],
    ["Ciphertext", () => Ciphertext.unmarshalBinary(new Uint8Array([1, 2, 3]))],
    ["Plaintext", () => Plaintext.unmarshalBinary(new Uint8Array([1, 2, 3]))],
    [
      "MemEvaluationKeySet",
      () => MemEvaluationKeySet.unmarshalBinary(new Uint8Array([1, 2, 3])),
    ],
  ])("%s.unmarshalBinary throws on invalid bytes", (_name, fn) => {
    expect(fn).toThrow();
  });
});
