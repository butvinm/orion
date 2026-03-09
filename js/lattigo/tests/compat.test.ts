// Cross-platform serialization test: encrypt in WASM, marshal bytes,
// verify native Go can unmarshal + decrypt.
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { ensureWasmLoaded, TEST_PARAMS } from "./helpers.js";
import {
  CKKSParameters,
  KeyGenerator,
  Encoder,
  Encryptor,
} from "../src/index.js";
import type { SecretKey, PublicKey } from "../src/index.js";
import { execSync } from "node:child_process";
import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const COMPAT_DIR = join(__dirname, "compat");

let params: CKKSParameters;
let kg: KeyGenerator;
let sk: SecretKey;
let pk: PublicKey;
let encoder: Encoder;
let encryptor: Encryptor;

beforeAll(async () => {
  await ensureWasmLoaded();
  params = new CKKSParameters(TEST_PARAMS);
  kg = new KeyGenerator(params);
  sk = kg.genSecretKey();
  pk = kg.genPublicKey(sk);
  encoder = new Encoder(params);
  encryptor = new Encryptor(params, pk);
});

afterAll(() => {
  encryptor?.free();
  encoder?.free();
  pk?.free();
  sk?.free();
  kg?.free();
  params?.free();
});

describe("Cross-platform serialization", () => {
  it("WASM-encrypted ciphertext can be decrypted by native Go", () => {
    const input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    const level = params.maxLevel();
    const scale = params.defaultScale();

    // Encrypt in WASM
    const pt = encoder.encode(input, level, scale);
    const ct = encryptor.encryptNew(pt);

    // Marshal SK and CT to bytes
    const skBytes = sk.marshalBinary();
    const ctBytes = ct.marshalBinary();
    const paramsJson = JSON.stringify({
      LogN: TEST_PARAMS.logN,
      LogQ: TEST_PARAMS.logQ,
      LogP: TEST_PARAMS.logP,
      LogDefaultScale: TEST_PARAMS.logDefaultScale,
      H: 192,
      RingType: TEST_PARAMS.ringType,
    });

    // Write to temp files
    const tmpDir = mkdtempSync(join(tmpdir(), "lattigo-compat-"));
    const paramsFile = join(tmpDir, "params.json");
    const skFile = join(tmpDir, "sk.bin");
    const ctFile = join(tmpDir, "ct.bin");

    try {
      writeFileSync(paramsFile, paramsJson);
      writeFileSync(skFile, skBytes);
      writeFileSync(ctFile, ctBytes);

      // Run native Go program to unmarshal + decrypt
      const output = execSync(
        `go run . "${paramsFile}" "${skFile}" "${ctFile}"`,
        {
          cwd: COMPAT_DIR,
          encoding: "utf-8",
          timeout: 60000,
        },
      ).trim();

      // Parse output (JSON array of decoded values)
      const decoded: number[] = JSON.parse(output);
      expect(decoded).toHaveLength(8);

      // Verify values match within CKKS tolerance
      const tolerance = 1e-3;
      for (let i = 0; i < input.length; i++) {
        expect(Math.abs(decoded[i] - input[i])).toBeLessThan(tolerance);
      }
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
      pt.free();
      ct.free();
    }
  });
});
