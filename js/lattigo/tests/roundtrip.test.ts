import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { ensureWasmLoaded, TEST_PARAMS } from "./helpers.js";
import {
  CKKSParameters,
  KeyGenerator,
  Encoder,
  Encryptor,
  Decryptor,
} from "../src/index.js";
import type { SecretKey, PublicKey } from "../src/index.js";

let params: CKKSParameters;
let kg: KeyGenerator;
let sk: SecretKey;
let pk: PublicKey;
let encoder: Encoder;
let encryptor: Encryptor;
let decryptor: Decryptor;

beforeAll(async () => {
  await ensureWasmLoaded();
  params = new CKKSParameters(TEST_PARAMS);
  kg = new KeyGenerator(params);
  sk = kg.genSecretKey();
  pk = kg.genPublicKey(sk);
  encoder = new Encoder(params);
  encryptor = new Encryptor(params, pk);
  decryptor = new Decryptor(params, sk);
});

afterAll(() => {
  decryptor?.free();
  encryptor?.free();
  encoder?.free();
  pk?.free();
  sk?.free();
  kg?.free();
  params?.free();
});

describe("Encryption roundtrip", () => {
  it("encode -> encrypt -> decrypt -> decode preserves values (number[])", () => {
    const input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    const level = params.maxLevel();
    const scale = params.defaultScale();

    const pt = encoder.encode(input, level, scale);
    const ct = encryptor.encryptNew(pt);
    const decPt = decryptor.decryptNew(ct);
    const decoded = encoder.decode(decPt, params.maxSlots());

    expect(decoded).toBeInstanceOf(Float64Array);
    const tolerance = 1e-3;
    for (let i = 0; i < input.length; i++) {
      expect(Math.abs(decoded[i] - input[i])).toBeLessThan(tolerance);
    }

    pt.free();
    ct.free();
    decPt.free();
  });

  it("encode -> encrypt -> decrypt -> decode preserves values (Float64Array)", () => {
    const input = new Float64Array([1.5, -2.3, 0.0, 42.0]);
    const level = params.maxLevel();
    const scale = params.defaultScale();

    const pt = encoder.encode(input, level, scale);
    const ct = encryptor.encryptNew(pt);
    const decPt = decryptor.decryptNew(ct);
    const decoded = encoder.decode(decPt, params.maxSlots());

    const tolerance = 1e-3;
    for (let i = 0; i < input.length; i++) {
      expect(Math.abs(decoded[i] - input[i])).toBeLessThan(tolerance);
    }

    pt.free();
    ct.free();
    decPt.free();
  });

  it("encode -> encrypt -> decrypt -> decode with sin values", () => {
    const numValues = 16;
    const input: number[] = [];
    for (let i = 0; i < numValues; i++) {
      input.push(Math.sin(i));
    }
    const level = params.maxLevel();
    const scale = params.defaultScale();

    const pt = encoder.encode(input, level, scale);
    const ct = encryptor.encryptNew(pt);
    const decPt = decryptor.decryptNew(ct);
    const decoded = encoder.decode(decPt, params.maxSlots());

    const tolerance = 1e-3;
    for (let i = 0; i < numValues; i++) {
      expect(Math.abs(decoded[i] - input[i])).toBeLessThan(tolerance);
    }

    pt.free();
    ct.free();
    decPt.free();
  });

  it("ciphertext level matches encoding level", () => {
    const level = params.maxLevel();
    const scale = params.defaultScale();
    const pt = encoder.encode([1.0], level, scale);
    const ct = encryptor.encryptNew(pt);

    expect(ct.level()).toBe(level);

    pt.free();
    ct.free();
  });

  it("plaintext level matches encoding level", () => {
    const level = params.maxLevel();
    const scale = params.defaultScale();
    const pt = encoder.encode([1.0], level, scale);

    expect(pt.level()).toBe(level);

    pt.free();
  });
});
