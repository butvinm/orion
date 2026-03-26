/**
 * Evaluation key generation from a KeyManifest JSON.
 *
 * This is the reference implementation that users copy from. It shows:
 * 1. Parsing a KeyManifest (produced by orion-compiler)
 * 2. Generating RLK (if needs_rlk)
 * 3. Generating each Galois key (from galois_elements)
 * 4. Constructing bootstrap ParametersLiteral (handling ConjugateInvariant LogN+1)
 * 5. Calling btpParamsGenEvaluationKeys (async, heavy)
 * 6. Assembling final MemEvaluationKeySet and printing sizes
 */
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

// Load Go WASM runtime (sets globalThis.Go)
require(join(__dirname, "node_modules/orion-v2-lattigo/wasm/wasm_exec.js"));

import {
  loadLattigo,
  CKKSParameters,
  KeyGenerator,
  RelinearizationKey,
  GaloisKey,
  MemEvaluationKeySet,
  isError,
} from "orion-v2-lattigo";
import type { WasmBridge } from "orion-v2-lattigo";

// ---------------------------------------------------------------------------
// Types matching orion-compiler's KeyManifest.to_dict() output
// ---------------------------------------------------------------------------

interface KeyManifest {
  galois_elements: number[];
  bootstrap_slots: number[];
  boot_logp: number[] | null;
  needs_rlk: boolean;
}

interface CKKSParamsJSON {
  LogN: number;
  LogQ: number[];
  LogP: number[];
  LogDefaultScale: number;
  H: number;
  RingType: "ConjugateInvariant" | "Standard";
}

// ---------------------------------------------------------------------------
// Example data — matches a small compiled model
// ---------------------------------------------------------------------------

const CKKS_PARAMS: CKKSParamsJSON = {
  LogN: 13,
  LogQ: [29, 26, 26, 26, 26, 26],
  LogP: [29, 29],
  LogDefaultScale: 26,
  H: 192,
  RingType: "ConjugateInvariant",
};

// A manifest with a few galois elements and bootstrap
const KEY_MANIFEST: KeyManifest = {
  galois_elements: [5, 25, 125],
  bootstrap_slots: [8], // small bootstrap slot count
  boot_logp: [46, 46, 46],
  needs_rlk: true,
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function main() {
  console.log("=== Evaluation Key Generation from KeyManifest ===\n");

  // 1. Load WASM
  const t0 = performance.now();
  const wasmPath = join(__dirname, "../../lattigo/wasm/lattigo.wasm");
  const bridge = await loadLattigo(wasmPath);
  console.log(`WASM loaded in ${(performance.now() - t0).toFixed(1)}ms\n`);

  // 2. Create CKKS parameters
  const params = CKKSParameters.fromLogn({
    logN: CKKS_PARAMS.LogN,
    logQ: CKKS_PARAMS.LogQ,
    logP: CKKS_PARAMS.LogP,
    logDefaultScale: CKKS_PARAMS.LogDefaultScale,
    h: CKKS_PARAMS.H,
    ringType: CKKS_PARAMS.RingType,
  });
  console.log(
    `CKKS params: LogN=${CKKS_PARAMS.LogN}, levels=${params.maxLevel()}, slots=${params.maxSlots()}`,
  );
  console.log(`KeyManifest: needs_rlk=${KEY_MANIFEST.needs_rlk}, ` +
    `galois_elements=[${KEY_MANIFEST.galois_elements.join(", ")}], ` +
    `bootstrap_slots=[${KEY_MANIFEST.bootstrap_slots.join(", ")}]\n`);

  // 3. Generate secret key
  const t1 = performance.now();
  const kg = KeyGenerator.new(params);
  const sk = kg.genSecretKey();
  console.log(`Secret key generated in ${(performance.now() - t1).toFixed(1)}ms`);

  // 4. Generate RLK if needed
  let rlk: RelinearizationKey | null = null;
  if (KEY_MANIFEST.needs_rlk) {
    const t2 = performance.now();
    rlk = kg.genRelinKey(sk);
    const rlkBytes = rlk.marshalBinary();
    console.log(
      `RLK generated in ${(performance.now() - t2).toFixed(1)}ms (${formatBytes(rlkBytes.length)})`,
    );
  }

  // 5. Generate Galois keys
  const galoisKeys: GaloisKey[] = [];
  const t3 = performance.now();
  for (let i = 0; i < KEY_MANIFEST.galois_elements.length; i++) {
    const ge = KEY_MANIFEST.galois_elements[i];
    const gk = kg.genGaloisKey(sk, ge);
    galoisKeys.push(gk);
    const gkBytes = gk.marshalBinary();
    console.log(
      `  Galois key ${i + 1}/${KEY_MANIFEST.galois_elements.length} ` +
        `(element=${ge}): ${formatBytes(gkBytes.length)}`,
    );
  }
  console.log(
    `All ${galoisKeys.length} Galois keys generated in ${(performance.now() - t3).toFixed(1)}ms`,
  );

  // 6. Assemble non-bootstrap MemEvaluationKeySet
  const t4 = performance.now();
  const evk = MemEvaluationKeySet.new(rlk, galoisKeys);
  const evkBytes = evk.marshalBinary();
  console.log(
    `\nMemEvaluationKeySet assembled in ${(performance.now() - t4).toFixed(1)}ms (${formatBytes(evkBytes.length)})`,
  );

  // 7. Bootstrap key generation (if manifest specifies bootstrap_slots)
  if (KEY_MANIFEST.bootstrap_slots.length > 0 && KEY_MANIFEST.boot_logp) {
    console.log("\n--- Bootstrap Key Generation ---");

    // Construct bootstrap ParametersLiteral JSON
    // For ConjugateInvariant ring type, bootstrap operates at LogN+1
    const btpLogN =
      CKKS_PARAMS.RingType === "ConjugateInvariant"
        ? CKKS_PARAMS.LogN + 1
        : CKKS_PARAMS.LogN;

    // LogSlots = log2(bootstrap_slots[0]) for the first bootstrap config
    const btpLogSlots = Math.round(Math.log2(KEY_MANIFEST.bootstrap_slots[0]));

    const btpLiteral = {
      LogN: btpLogN,
      LogP: KEY_MANIFEST.boot_logp,
      H: CKKS_PARAMS.H,
      LogSlots: btpLogSlots,
    };

    console.log(
      `Bootstrap literal: LogN=${btpLogN}, LogP=[${btpLiteral.LogP.join(", ")}], ` +
        `H=${btpLiteral.H}, LogSlots=${btpLogSlots}`,
    );

    // Create bootstrap parameters
    const t5 = performance.now();
    const btpParamsResult = bridge.newBootstrapParametersFromLiteral(
      params.handle,
      JSON.stringify(btpLiteral),
    );
    if (isError(btpParamsResult)) {
      console.error(`Bootstrap params error: ${btpParamsResult.error}`);
      process.exit(1);
    }
    console.log(
      `Bootstrap params created in ${(performance.now() - t5).toFixed(1)}ms`,
    );

    // Generate bootstrap evaluation keys (async — heavy operation)
    const t6 = performance.now();
    console.log("Generating bootstrap evaluation keys (this may take a while)...");
    const btpKeysResult = await bridge.btpParamsGenEvaluationKeys(
      btpParamsResult.handle,
      sk.handle,
    );
    const btpDuration = performance.now() - t6;
    console.log(
      `Bootstrap eval keys generated in ${(btpDuration / 1000).toFixed(1)}s`,
    );
    console.log(
      `  evkHID=${btpKeysResult.evkHID}, btpEvkHID=${btpKeysResult.btpEvkHID}`,
    );

    // Clean up bootstrap handles
    bridge.deleteHandle(btpParamsResult.handle);
    bridge.deleteHandle(btpKeysResult.evkHID);
    bridge.deleteHandle(btpKeysResult.btpEvkHID);
  }

  // 8. Summary
  console.log("\n=== Summary ===");
  console.log(`Total evaluation key set size: ${formatBytes(evkBytes.length)}`);
  console.log(`  RLK: ${rlk ? "yes" : "no"}`);
  console.log(`  Galois keys: ${galoisKeys.length}`);
  console.log(
    `  Bootstrap: ${KEY_MANIFEST.bootstrap_slots.length > 0 ? "yes" : "no"}`,
  );

  // 9. Cleanup
  evk.close();
  for (const gk of galoisKeys) gk.close();
  if (rlk) rlk.close();
  sk.close();
  kg.close();
  params.close();

  console.log("\nAll resources closed. Done.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
