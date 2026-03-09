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
    it("constructor throws on invalid ring type", () => {
      expect(
        () =>
          new CKKSParameters({
            logN: 13,
            logQ: [29, 26],
            logP: [29],
            logDefaultScale: 26,
            ringType: "InvalidRingType" as any,
          }),
      ).toThrow();
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
