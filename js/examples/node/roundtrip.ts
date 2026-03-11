/**
 * Roundtrip example: keygen -> encode -> encrypt -> decrypt -> decode.
 *
 * Demonstrates the full CKKS homomorphic encryption lifecycle using
 * the @orion/lattigo WASM bindings in Node.js.
 */
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

// Load Go WASM runtime (sets globalThis.Go)
require(join(__dirname, "../../lattigo/wasm/wasm_exec.js"));

import {
  loadLattigo,
  CKKSParameters,
  KeyGenerator,
  Encoder,
  Encryptor,
  Decryptor,
} from "@orion/lattigo";

async function main() {
  console.log("=== Lattigo CKKS Roundtrip Example ===\n");

  // 1. Load WASM
  const t0 = performance.now();
  const wasmPath = join(__dirname, "../../lattigo/wasm/lattigo.wasm");
  await loadLattigo(wasmPath);
  console.log(`WASM loaded in ${(performance.now() - t0).toFixed(1)}ms`);

  // 2. Create CKKS parameters
  const t1 = performance.now();
  const params = CKKSParameters.fromLogn({
    logN: 13,
    logQ: [29, 26, 26, 26, 26, 26],
    logP: [29, 29],
    logDefaultScale: 26,
    ringType: "ConjugateInvariant",
  });
  console.log(`Parameters created in ${(performance.now() - t1).toFixed(1)}ms`);
  console.log(`  maxSlots: ${params.maxSlots()}`);
  console.log(`  maxLevel: ${params.maxLevel()}`);
  console.log(`  defaultScale: ${params.defaultScale().toExponential(4)}`);

  // 3. Key generation
  const t2 = performance.now();
  const kg = KeyGenerator.new(params);
  const sk = kg.genSecretKey();
  const pk = kg.genPublicKey(sk);
  console.log(`Key generation in ${(performance.now() - t2).toFixed(1)}ms`);

  // 4. Create encoder, encryptor, decryptor
  const encoder = Encoder.new(params);
  const encryptor = Encryptor.new(params, pk);
  const decryptor = Decryptor.new(params, sk);

  // 5. Prepare input values
  const numSlots = params.maxSlots();
  const input = new Float64Array(numSlots);
  for (let i = 0; i < numSlots; i++) {
    input[i] = Math.sin((2 * Math.PI * i) / numSlots);
  }
  console.log(
    `\nInput (first 8 values): [${Array.from(input.slice(0, 8)).map((v) => v.toFixed(6)).join(", ")}]`,
  );

  // 6. Encode
  const t3 = performance.now();
  const level = params.maxLevel();
  const scale = params.defaultScale();
  const pt = encoder.encode(input, level, scale);
  console.log(`Encode in ${(performance.now() - t3).toFixed(1)}ms`);

  // 7. Encrypt
  const t4 = performance.now();
  const ct = encryptor.encryptNew(pt);
  console.log(`Encrypt in ${(performance.now() - t4).toFixed(1)}ms`);

  // 8. Decrypt
  const t5 = performance.now();
  const ptResult = decryptor.decryptNew(ct);
  console.log(`Decrypt in ${(performance.now() - t5).toFixed(1)}ms`);

  // 9. Decode
  const t6 = performance.now();
  const output = encoder.decode(ptResult, numSlots);
  console.log(`Decode in ${(performance.now() - t6).toFixed(1)}ms`);

  // 10. Verify
  console.log(
    `\nOutput (first 8 values): [${Array.from(output.slice(0, 8)).map((v) => v.toFixed(6)).join(", ")}]`,
  );

  let maxError = 0;
  for (let i = 0; i < numSlots; i++) {
    const err = Math.abs(output[i] - input[i]);
    if (err > maxError) maxError = err;
  }
  console.log(`\nMax error: ${maxError.toExponential(4)}`);

  const tolerance = 1e-4;
  if (maxError < tolerance) {
    console.log(`PASS: max error < ${tolerance}`);
  } else {
    console.error(`FAIL: max error ${maxError} >= ${tolerance}`);
    process.exit(1);
  }

  // 11. Cleanup
  ptResult.close();
  ct.close();
  pt.close();
  decryptor.close();
  encryptor.close();
  encoder.close();
  pk.close();
  sk.close();
  kg.close();
  params.close();

  console.log("\nAll resources closed. Done.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
