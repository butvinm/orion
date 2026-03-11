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
  it("CKKSParameters.close() is idempotent (double close)", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    expect(() => params.close()).not.toThrow();
    expect(() => params.close()).not.toThrow();
    expect(() => params.close()).not.toThrow();
  });

  it("KeyGenerator.close() is idempotent", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    const kg = new KeyGenerator(params);
    expect(() => kg.close()).not.toThrow();
    expect(() => kg.close()).not.toThrow();
    params.close();
  });

  it("SecretKey.close() is idempotent", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    const kg = new KeyGenerator(params);
    const sk = kg.genSecretKey();
    expect(() => sk.close()).not.toThrow();
    expect(() => sk.close()).not.toThrow();
    kg.close();
    params.close();
  });

  it("all key types .close() are idempotent", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    const kg = new KeyGenerator(params);
    const sk = kg.genSecretKey();
    const pk = kg.genPublicKey(sk);
    const rlk = kg.genRelinKey(sk);
    const ge = params.galoisElement(1);
    const gk = kg.genGaloisKey(sk, ge);

    // Free everything twice
    expect(() => gk.close()).not.toThrow();
    expect(() => gk.close()).not.toThrow();
    expect(() => rlk.close()).not.toThrow();
    expect(() => rlk.close()).not.toThrow();
    expect(() => pk.close()).not.toThrow();
    expect(() => pk.close()).not.toThrow();
    expect(() => sk.close()).not.toThrow();
    expect(() => sk.close()).not.toThrow();
    expect(() => kg.close()).not.toThrow();
    expect(() => kg.close()).not.toThrow();
    expect(() => params.close()).not.toThrow();
    expect(() => params.close()).not.toThrow();
  });

  it("Encoder.close() is idempotent", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    const encoder = new Encoder(params);
    expect(() => encoder.close()).not.toThrow();
    expect(() => encoder.close()).not.toThrow();
    params.close();
  });

  it("Encryptor and Decryptor .close() are idempotent", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    const kg = new KeyGenerator(params);
    const sk = kg.genSecretKey();
    const pk = kg.genPublicKey(sk);
    const encryptor = new Encryptor(params, pk);
    const decryptor = new Decryptor(params, sk);

    expect(() => encryptor.close()).not.toThrow();
    expect(() => encryptor.close()).not.toThrow();
    expect(() => decryptor.close()).not.toThrow();
    expect(() => decryptor.close()).not.toThrow();

    sk.close();
    pk.close();
    kg.close();
    params.close();
  });

  it("Ciphertext and Plaintext .close() are idempotent", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    const kg = new KeyGenerator(params);
    const sk = kg.genSecretKey();
    const pk = kg.genPublicKey(sk);
    const encoder = new Encoder(params);
    const encryptor = new Encryptor(params, pk);

    const pt = encoder.encode([1.0], params.maxLevel(), params.defaultScale());
    const ct = encryptor.encryptNew(pt);

    expect(() => pt.close()).not.toThrow();
    expect(() => pt.close()).not.toThrow();
    expect(() => ct.close()).not.toThrow();
    expect(() => ct.close()).not.toThrow();

    encoder.close();
    encryptor.close();
    sk.close();
    pk.close();
    kg.close();
    params.close();
  });

  it("MemEvaluationKeySet.close() is idempotent", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    const kg = new KeyGenerator(params);
    const sk = kg.genSecretKey();
    const rlk = kg.genRelinKey(sk);
    const evk = new MemEvaluationKeySet(rlk, []);

    expect(() => evk.close()).not.toThrow();
    expect(() => evk.close()).not.toThrow();

    rlk.close();
    sk.close();
    kg.close();
    params.close();
  });
});
