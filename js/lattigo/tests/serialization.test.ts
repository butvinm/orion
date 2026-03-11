import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { ensureWasmLoaded, TEST_PARAMS } from "./helpers.js";
import {
  CKKSParameters,
  KeyGenerator,
  SecretKey,
  PublicKey,
  RelinearizationKey,
  GaloisKey,
  Ciphertext,
  Plaintext,
  MemEvaluationKeySet,
  Encoder,
  Encryptor,
  Decryptor,
} from "../src/index.js";

let params: CKKSParameters;
let kg: KeyGenerator;

beforeAll(async () => {
  await ensureWasmLoaded();
  params = new CKKSParameters(TEST_PARAMS);
  kg = new KeyGenerator(params);
});

afterAll(() => {
  kg?.close();
  params?.close();
});

describe("Serialization roundtrip", () => {
  it("SecretKey: marshal -> unmarshal -> remarshal produces identical bytes", () => {
    const sk = kg.genSecretKey();
    const bytes1 = sk.marshalBinary();
    const sk2 = SecretKey.unmarshalBinary(bytes1);
    const bytes2 = sk2.marshalBinary();

    expect(bytes1).toEqual(bytes2);

    sk.close();
    sk2.close();
  });

  it("PublicKey: marshal -> unmarshal -> remarshal produces identical bytes", () => {
    const sk = kg.genSecretKey();
    const pk = kg.genPublicKey(sk);
    const bytes1 = pk.marshalBinary();
    const pk2 = PublicKey.unmarshalBinary(bytes1);
    const bytes2 = pk2.marshalBinary();

    expect(bytes1).toEqual(bytes2);

    sk.close();
    pk.close();
    pk2.close();
  });

  it("RelinearizationKey: marshal -> unmarshal -> remarshal produces identical bytes", () => {
    const sk = kg.genSecretKey();
    const rlk = kg.genRelinKey(sk);
    const bytes1 = rlk.marshalBinary();
    const rlk2 = RelinearizationKey.unmarshalBinary(bytes1);
    const bytes2 = rlk2.marshalBinary();

    expect(bytes1).toEqual(bytes2);

    sk.close();
    rlk.close();
    rlk2.close();
  });

  it("GaloisKey: marshal -> unmarshal -> remarshal produces identical bytes", () => {
    const sk = kg.genSecretKey();
    const ge = params.galoisElement(1);
    const gk = kg.genGaloisKey(sk, ge);
    const bytes1 = gk.marshalBinary();
    const gk2 = GaloisKey.unmarshalBinary(bytes1);
    const bytes2 = gk2.marshalBinary();

    expect(bytes1).toEqual(bytes2);

    sk.close();
    gk.close();
    gk2.close();
  });

  it("Ciphertext: marshal -> unmarshal -> decrypt produces same values", () => {
    const sk = kg.genSecretKey();
    const pk = kg.genPublicKey(sk);
    const encoder = new Encoder(params);
    const encryptor = new Encryptor(params, pk);
    const decryptor = new Decryptor(params, sk);

    const input = [1.0, 2.0, 3.0];
    const pt = encoder.encode(input, params.maxLevel(), params.defaultScale());
    const ct = encryptor.encryptNew(pt);

    const ctBytes = ct.marshalBinary();
    const ct2 = Ciphertext.unmarshalBinary(ctBytes);

    const decPt = decryptor.decryptNew(ct2);
    const decoded = encoder.decode(decPt, params.maxSlots());

    const tolerance = 1e-3;
    for (let i = 0; i < input.length; i++) {
      expect(Math.abs(decoded[i] - input[i])).toBeLessThan(tolerance);
    }

    pt.close();
    ct.close();
    ct2.close();
    decPt.close();
    encoder.close();
    encryptor.close();
    decryptor.close();
    sk.close();
    pk.close();
  });

  it("Plaintext: marshal -> unmarshal -> decode produces same values", () => {
    const encoder = new Encoder(params);
    const input = [4.0, 5.0, 6.0];
    const pt = encoder.encode(input, params.maxLevel(), params.defaultScale());

    const ptBytes = pt.marshalBinary();
    const pt2 = Plaintext.unmarshalBinary(ptBytes);

    const decoded = encoder.decode(pt2, params.maxSlots());
    const tolerance = 1e-4;
    for (let i = 0; i < input.length; i++) {
      expect(Math.abs(decoded[i] - input[i])).toBeLessThan(tolerance);
    }

    pt.close();
    pt2.close();
    encoder.close();
  });

  it("MemEvaluationKeySet: create from RLK + Galois keys -> marshal -> unmarshal", () => {
    const sk = kg.genSecretKey();
    const rlk = kg.genRelinKey(sk);
    const ge1 = params.galoisElement(1);
    const ge2 = params.galoisElement(2);
    const gk1 = kg.genGaloisKey(sk, ge1);
    const gk2 = kg.genGaloisKey(sk, ge2);

    const evk = new MemEvaluationKeySet(rlk, [gk1, gk2]);
    expect(evk.handle).toBeGreaterThan(0);

    const evkBytes = evk.marshalBinary();
    expect(evkBytes.length).toBeGreaterThan(0);

    const evk2 = MemEvaluationKeySet.unmarshalBinary(evkBytes);
    expect(evk2.handle).toBeGreaterThan(0);

    // Remarshal and compare
    const evkBytes2 = evk2.marshalBinary();
    expect(evkBytes).toEqual(evkBytes2);

    evk.close();
    evk2.close();
    gk1.close();
    gk2.close();
    rlk.close();
    sk.close();
  });

  it("MemEvaluationKeySet: create with null RLK", () => {
    const sk = kg.genSecretKey();
    const ge = params.galoisElement(1);
    const gk = kg.genGaloisKey(sk, ge);

    const evk = new MemEvaluationKeySet(null, [gk]);
    expect(evk.handle).toBeGreaterThan(0);

    const evkBytes = evk.marshalBinary();
    expect(evkBytes.length).toBeGreaterThan(0);

    evk.close();
    gk.close();
    sk.close();
  });
});
