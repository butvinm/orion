import { describe, it, expect, beforeAll } from "vitest";
import { ensureWasmLoaded } from "./helpers.js";
import {
  CKKSParameters,
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
  describe("CKKSParameters", () => {
    it("fromJSON throws on invalid JSON string", () => {
      expect(() => CKKSParameters.fromJSON("not json")).toThrow();
    });

    it("fromJSON throws on empty JSON object", () => {
      expect(() => CKKSParameters.fromJSON("{}")).toThrow();
    });

    it("fromJSON throws on invalid ring type", () => {
      const json = JSON.stringify({
        LogN: 13,
        LogQ: [29, 26],
        LogP: [29],
        LogDefaultScale: 26,
        RingType: "InvalidRingType",
      });
      expect(() => CKKSParameters.fromJSON(json)).toThrow(/ring type/i);
    });
  });

  describe("Key unmarshal", () => {
    it("SecretKey.unmarshalBinary throws on invalid bytes", () => {
      expect(() =>
        SecretKey.unmarshalBinary(new Uint8Array([1, 2, 3])),
      ).toThrow();
    });

    it("PublicKey.unmarshalBinary throws on invalid bytes", () => {
      expect(() =>
        PublicKey.unmarshalBinary(new Uint8Array([1, 2, 3])),
      ).toThrow();
    });

    it("RelinearizationKey.unmarshalBinary throws on invalid bytes", () => {
      expect(() =>
        RelinearizationKey.unmarshalBinary(new Uint8Array([1, 2, 3])),
      ).toThrow();
    });

    it("GaloisKey.unmarshalBinary throws on invalid bytes", () => {
      expect(() =>
        GaloisKey.unmarshalBinary(new Uint8Array([1, 2, 3])),
      ).toThrow();
    });

    it("Ciphertext.unmarshalBinary throws on invalid bytes", () => {
      expect(() =>
        Ciphertext.unmarshalBinary(new Uint8Array([1, 2, 3])),
      ).toThrow();
    });

    it("Plaintext.unmarshalBinary throws on invalid bytes", () => {
      expect(() =>
        Plaintext.unmarshalBinary(new Uint8Array([1, 2, 3])),
      ).toThrow();
    });

    it("MemEvaluationKeySet.unmarshalBinary throws on invalid bytes", () => {
      expect(() =>
        MemEvaluationKeySet.unmarshalBinary(new Uint8Array([1, 2, 3])),
      ).toThrow();
    });
  });

  describe("Empty byte arrays", () => {
    it("SecretKey.unmarshalBinary throws on empty bytes", () => {
      expect(() =>
        SecretKey.unmarshalBinary(new Uint8Array([])),
      ).toThrow();
    });

    it("Ciphertext.unmarshalBinary throws on empty bytes", () => {
      expect(() =>
        Ciphertext.unmarshalBinary(new Uint8Array([])),
      ).toThrow();
    });
  });
});
