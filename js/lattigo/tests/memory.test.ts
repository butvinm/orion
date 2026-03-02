import { describe, it, expect, beforeAll } from "vitest";
import { ensureWasmLoaded, TEST_PARAMS } from "./helpers.js";
import {
  CKKSParameters,
  KeyGenerator,
  Encoder,
  Encryptor,
  Decryptor,
  MemEvaluationKeySet,
} from "../src/index.js";

beforeAll(async () => {
  await ensureWasmLoaded();
});

describe("Memory cleanup", () => {
  it("CKKSParameters.free() is idempotent (double free)", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    expect(() => params.free()).not.toThrow();
    expect(() => params.free()).not.toThrow();
    expect(() => params.free()).not.toThrow();
  });

  it("KeyGenerator.free() is idempotent", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const kg = KeyGenerator.new(params);
    expect(() => kg.free()).not.toThrow();
    expect(() => kg.free()).not.toThrow();
    params.free();
  });

  it("SecretKey.free() is idempotent", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const kg = KeyGenerator.new(params);
    const sk = kg.genSecretKey();
    expect(() => sk.free()).not.toThrow();
    expect(() => sk.free()).not.toThrow();
    kg.free();
    params.free();
  });

  it("all key types .free() are idempotent", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const kg = KeyGenerator.new(params);
    const sk = kg.genSecretKey();
    const pk = kg.genPublicKey(sk);
    const rlk = kg.genRelinKey(sk);
    const ge = params.galoisElement(1);
    const gk = kg.genGaloisKey(sk, ge);

    // Free everything twice
    expect(() => gk.free()).not.toThrow();
    expect(() => gk.free()).not.toThrow();
    expect(() => rlk.free()).not.toThrow();
    expect(() => rlk.free()).not.toThrow();
    expect(() => pk.free()).not.toThrow();
    expect(() => pk.free()).not.toThrow();
    expect(() => sk.free()).not.toThrow();
    expect(() => sk.free()).not.toThrow();
    expect(() => kg.free()).not.toThrow();
    expect(() => kg.free()).not.toThrow();
    expect(() => params.free()).not.toThrow();
    expect(() => params.free()).not.toThrow();
  });

  it("Encoder.free() is idempotent", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const encoder = Encoder.new(params);
    expect(() => encoder.free()).not.toThrow();
    expect(() => encoder.free()).not.toThrow();
    params.free();
  });

  it("Encryptor and Decryptor .free() are idempotent", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const kg = KeyGenerator.new(params);
    const sk = kg.genSecretKey();
    const pk = kg.genPublicKey(sk);
    const encryptor = Encryptor.new(params, pk);
    const decryptor = Decryptor.new(params, sk);

    expect(() => encryptor.free()).not.toThrow();
    expect(() => encryptor.free()).not.toThrow();
    expect(() => decryptor.free()).not.toThrow();
    expect(() => decryptor.free()).not.toThrow();

    sk.free();
    pk.free();
    kg.free();
    params.free();
  });

  it("Ciphertext and Plaintext .free() are idempotent", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const kg = KeyGenerator.new(params);
    const sk = kg.genSecretKey();
    const pk = kg.genPublicKey(sk);
    const encoder = Encoder.new(params);
    const encryptor = Encryptor.new(params, pk);

    const pt = encoder.encode([1.0], params.maxLevel(), params.defaultScale());
    const ct = encryptor.encryptNew(pt);

    expect(() => pt.free()).not.toThrow();
    expect(() => pt.free()).not.toThrow();
    expect(() => ct.free()).not.toThrow();
    expect(() => ct.free()).not.toThrow();

    encoder.free();
    encryptor.free();
    sk.free();
    pk.free();
    kg.free();
    params.free();
  });

  it("MemEvaluationKeySet.free() is idempotent", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const kg = KeyGenerator.new(params);
    const sk = kg.genSecretKey();
    const rlk = kg.genRelinKey(sk);
    const evk = MemEvaluationKeySet.new(rlk, []);

    expect(() => evk.free()).not.toThrow();
    expect(() => evk.free()).not.toThrow();

    rlk.free();
    sk.free();
    kg.free();
    params.free();
  });
});
