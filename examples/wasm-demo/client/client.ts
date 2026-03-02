/**
 * Browser client for the Orion WASM encrypted inference demo.
 *
 * Full manual Lattigo usage — no convenience wrappers beyond the TS classes.
 * Secret key is generated and kept in browser memory; never sent to server.
 *
 * Flow:
 *   Initialize Keys:
 *     1. GET /params  → CKKS params, key manifest, input level
 *     2. Load Lattigo WASM
 *     3. Create CKKSParameters from params JSON
 *     4. Generate SK, PK via KeyGenerator
 *     5. Generate RLK if manifest.needs_rlk
 *     6. Generate Galois keys from manifest.galois_elements (with progress)
 *     7. If manifest.bootstrap_slots non-empty: construct ParametersLiteral,
 *        call btpParamsGenEvaluationKeys (async, may take 5–30s)
 *     8. Assemble MemEvaluationKeySet, marshal, POST /session → session ID
 *
 *   Run Inference:
 *     9.  Encode + encrypt user input
 *     10. POST /session/{id}/infer → result ciphertext bytes
 *     11. Decrypt + decode → display values with timing breakdown
 */

import {
  loadLattigo,
  CKKSParameters,
  KeyGenerator,
  Encoder,
  Encryptor,
  Decryptor,
  MemEvaluationKeySet,
  Ciphertext,
  isError,
} from "@orion/lattigo";
import type { WasmBridge, Handle } from "@orion/lattigo";
import type { RelinearizationKey, GaloisKey, SecretKey } from "@orion/lattigo";

// ============================================================
// Types matching the /params response from the Go server
// ============================================================

interface OrionParams {
  logn: number;
  logq: number[];
  logp: number[];
  logscale: number;
  h: number;
  ring_type: string;
  boot_logp?: number[];
}

interface KeyManifest {
  galois_elements: number[];
  bootstrap_slots: number[];
  boot_logp?: number[];
  needs_rlk: boolean;
}

interface ParamsResponse {
  ckks_params: OrionParams;
  key_manifest: KeyManifest;
  input_level: number;
}

// ============================================================
// Application state (module-level singletons)
// ============================================================

let bridge: WasmBridge | null = null;
let params: CKKSParameters | null = null;
let sk: SecretKey | null = null;
let encryptor: Encryptor | null = null;
let decryptor: Decryptor | null = null;
let encoder: Encoder | null = null;
let sessionId: string | null = null;
let inputLevel = 0;

// ============================================================
// UI helpers
// ============================================================

function setStatus(msg: string): void {
  const el = document.getElementById("status");
  if (el) el.textContent = msg;
}

function appendLine(msg: string): void {
  const el = document.getElementById("status");
  if (el) el.textContent += "\n" + msg;
}

function setOutput(msg: string): void {
  const el = document.getElementById("output");
  if (el) el.textContent = msg;
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

// ============================================================
// Initialize keys
// ============================================================

async function initializeKeys(): Promise<void> {
  const btnInit = document.getElementById("btn-init") as HTMLButtonElement;
  const btnInfer = document.getElementById("btn-infer") as HTMLButtonElement;
  btnInit.disabled = true;
  btnInfer.disabled = true;
  setStatus("Fetching parameters from server...");

  const tTotal = performance.now();

  // 1. GET /params
  let serverParams: ParamsResponse;
  try {
    const resp = await fetch("/params");
    if (!resp.ok) {
      setStatus(`Error fetching /params: ${resp.status} ${resp.statusText}`);
      btnInit.disabled = false;
      return;
    }
    serverParams = (await resp.json()) as ParamsResponse;
  } catch (err) {
    setStatus(`Network error: ${(err as Error).message}`);
    btnInit.disabled = false;
    return;
  }

  const { ckks_params: ckksData, key_manifest: manifest } = serverParams;
  inputLevel = serverParams.input_level;

  setStatus(
    `Server params: LogN=${ckksData.logn}, levels=${ckksData.logq.length - 1}, ` +
      `input_level=${inputLevel}, needs_rlk=${manifest.needs_rlk}, ` +
      `galois_keys=${manifest.galois_elements.length}`,
  );

  // 2. Load Lattigo WASM
  appendLine("Loading Lattigo WASM...");
  const t1 = performance.now();
  bridge = await loadLattigo("wasm/lattigo.wasm");
  appendLine(`WASM loaded in ${formatDuration(performance.now() - t1)}`);

  // 3. Create CKKS params (convert orion format → bridge format)
  const bridgeParamsJSON = JSON.stringify({
    LogN: ckksData.logn,
    LogQ: ckksData.logq,
    LogP: ckksData.logp,
    LogDefaultScale: ckksData.logscale,
    H: ckksData.h,
    RingType: ckksData.ring_type, // bridge handles "conjugate_invariant" directly
  });
  params = CKKSParameters.fromJSON(bridgeParamsJSON);
  appendLine(
    `CKKS params: ${params.maxSlots()} slots, max level ${params.maxLevel()}`,
  );

  // 4. Keygen
  appendLine("Generating secret key...");
  const t2 = performance.now();
  const kg = KeyGenerator.new(params);
  sk = kg.genSecretKey();
  const pk = kg.genPublicKey(sk);
  appendLine(`Keygen in ${formatDuration(performance.now() - t2)}`);

  // 5. Create encoder, encryptor, decryptor
  encoder = Encoder.new(params);
  encryptor = Encryptor.new(params, pk);
  decryptor = Decryptor.new(params, sk);

  // 6. Generate RLK if needed
  let rlk: RelinearizationKey | null = null;
  if (manifest.needs_rlk) {
    appendLine("Generating relinearization key...");
    const t3 = performance.now();
    rlk = kg.genRelinKey(sk);
    appendLine(
      `RLK in ${formatDuration(performance.now() - t3)} (${formatBytes(rlk.marshalBinary().length)})`,
    );
  }

  // 7. Generate Galois keys
  const galoisKeys: GaloisKey[] = [];
  if (manifest.galois_elements.length > 0) {
    const total = manifest.galois_elements.length;
    appendLine(`Generating ${total} Galois key(s)...`);
    const t4 = performance.now();
    for (let i = 0; i < total; i++) {
      const ge = manifest.galois_elements[i];
      const gk = kg.genGaloisKey(sk, ge);
      galoisKeys.push(gk);
      const pct = Math.round(((i + 1) / total) * 100);
      setStatus(
        `Galois keys: ${i + 1}/${total} [${pct}%] — element=${ge} — ` +
          `${formatDuration(performance.now() - t4)} elapsed`,
      );
    }
    appendLine(
      `${total} Galois key(s) in ${formatDuration(performance.now() - t4)}`,
    );
  }

  // 8. Bootstrap keys (if needed)
  if (
    manifest.bootstrap_slots.length > 0 &&
    manifest.boot_logp &&
    manifest.boot_logp.length > 0
  ) {
    appendLine(
      "Generating bootstrap evaluation keys (may take 5–30s)...",
    );

    // For ConjugateInvariant, bootstrap operates at LogN+1
    const btpLogN =
      ckksData.ring_type === "conjugate_invariant" ||
      ckksData.ring_type === "ConjugateInvariant"
        ? ckksData.logn + 1
        : ckksData.logn;

    const btpLogSlots = Math.round(
      Math.log2(manifest.bootstrap_slots[0]),
    );

    const btpLiteral = {
      LogN: btpLogN,
      LogP: manifest.boot_logp,
      H: ckksData.h,
      LogSlots: btpLogSlots,
    };

    const btpParamsResult = bridge.newBootstrapParametersFromLiteral(
      params.handle,
      JSON.stringify(btpLiteral),
    );
    if (isError(btpParamsResult)) {
      setStatus(`Bootstrap params error: ${btpParamsResult.error}`);
      btnInit.disabled = false;
      return;
    }

    const t5 = performance.now();
    const btpKeys = await bridge.btpParamsGenEvaluationKeys(
      btpParamsResult.handle,
      sk.handle,
    );
    appendLine(
      `Bootstrap eval keys in ${formatDuration(performance.now() - t5)}`,
    );

    // Free bootstrap handles — this demo only needs the standard eval keys
    bridge.deleteHandle(btpParamsResult.handle);
    bridge.deleteHandle(btpKeys.evkHID);
    bridge.deleteHandle(btpKeys.btpEvkHID);
  }

  // 9. Assemble MemEvaluationKeySet and POST to /session
  appendLine("Assembling evaluation key set...");
  const t6 = performance.now();
  const evk = MemEvaluationKeySet.new(rlk, galoisKeys);
  const evkBytes = evk.marshalBinary();
  appendLine(
    `EVK assembled in ${formatDuration(performance.now() - t6)} (${formatBytes(evkBytes.length)})`,
  );

  appendLine("Uploading keys to server...");
  const t7 = performance.now();
  let sessData: { session_id: string };
  try {
    const sessResp = await fetch("/session", {
      method: "POST",
      // Cast: Go WASM bridge always returns ArrayBuffer (never SharedArrayBuffer)
      body: evkBytes as unknown as ArrayBuffer,
      headers: { "Content-Type": "application/octet-stream" },
    });
    if (!sessResp.ok) {
      const msg = await sessResp.text();
      setStatus(`Session creation failed: ${msg}`);
      btnInit.disabled = false;
      return;
    }
    sessData = (await sessResp.json()) as { session_id: string };
  } catch (err) {
    setStatus(`Session error: ${(err as Error).message}`);
    btnInit.disabled = false;
    return;
  }
  sessionId = sessData.session_id;
  appendLine(
    `Session created in ${formatDuration(performance.now() - t7)}: ${sessionId}`,
  );

  // Cleanup (PK, EVK, individual key handles — SK/encoder/encryptor/decryptor kept for inference)
  evk.free();
  for (const gk of galoisKeys) gk.free();
  if (rlk) rlk.free();
  pk.free();
  kg.free();

  const total = performance.now() - tTotal;
  appendLine(`\nTotal initialization: ${formatDuration(total)}`);
  appendLine("Ready — enter values and click Run Inference.");

  setOutput("Keys initialized.");
  btnInfer.disabled = false;
}

// ============================================================
// Run inference
// ============================================================

async function runInference(): Promise<void> {
  if (!sessionId || !encryptor || !decryptor || !encoder || !params) {
    setStatus("Error: not initialized");
    return;
  }

  const btnInfer = document.getElementById("btn-infer") as HTMLButtonElement;
  btnInfer.disabled = true;

  const inputEl = document.getElementById("input-values") as HTMLInputElement;
  const values = inputEl.value
    .split(",")
    .map((s) => parseFloat(s.trim()))
    .filter((v) => !isNaN(v));

  if (values.length === 0) {
    setStatus("Error: no valid values entered");
    btnInfer.disabled = false;
    return;
  }

  // Pad to maxSlots
  const maxSlots = params.maxSlots();
  const padded = new Array<number>(maxSlots).fill(0);
  for (let i = 0; i < Math.min(values.length, maxSlots); i++) {
    padded[i] = values[i];
  }

  setStatus("Encoding and encrypting input...");
  const tEnc = performance.now();

  const defaultScale = params.defaultScale();
  const pt = encoder.encode(padded, inputLevel, defaultScale);
  const ct = encryptor.encryptNew(pt);
  const ctBytes = ct.marshalBinary();
  pt.free();
  ct.free();

  const encTime = performance.now() - tEnc;

  // POST to server
  setStatus(
    `Sending ciphertext (${formatBytes(ctBytes.length)}) to server for inference...`,
  );
  const tInfer = performance.now();

  let resultBytes: Uint8Array;
  try {
    const inferResp = await fetch(`/session/${sessionId}/infer`, {
      method: "POST",
      // Cast: Go WASM bridge always returns ArrayBuffer (never SharedArrayBuffer)
      body: ctBytes as unknown as ArrayBuffer,
      headers: { "Content-Type": "application/octet-stream" },
    });
    if (!inferResp.ok) {
      const msg = await inferResp.text();
      setStatus(`Inference error: ${msg}`);
      btnInfer.disabled = false;
      return;
    }
    resultBytes = new Uint8Array(await inferResp.arrayBuffer());
  } catch (err) {
    setStatus(`Network error: ${(err as Error).message}`);
    btnInfer.disabled = false;
    return;
  }

  const inferTime = performance.now() - tInfer;

  // Decrypt and decode
  setStatus("Decrypting result...");
  const tDec = performance.now();

  const resultCt = Ciphertext.unmarshalBinary(resultBytes);
  const resultPt = decryptor.decryptNew(resultCt);
  const outputValues = encoder.decode(resultPt, maxSlots);
  resultCt.free();
  resultPt.free();

  const decTime = performance.now() - tDec;

  // Display first N values (up to original input count, capped at 10)
  const displayCount = Math.min(Math.max(values.length, 1), 10);
  const inputSlice = values.slice(0, displayCount).join(", ");
  const outputSlice = Array.from(outputValues.slice(0, displayCount))
    .map((v) => v.toFixed(6))
    .join(", ");

  setOutput(
    `Input:  [${inputSlice}]\n` +
      `Output: [${outputSlice}]\n\n` +
      `Timing:\n` +
      `  Encode + Encrypt : ${formatDuration(encTime)}\n` +
      `  Server inference : ${formatDuration(inferTime)}\n` +
      `  Decrypt + Decode : ${formatDuration(decTime)}\n` +
      `  Ciphertext size  : ${formatBytes(ctBytes.length)}\n` +
      `  Result size      : ${formatBytes(resultBytes.length)}`,
  );
  setStatus("Inference complete.");
  btnInfer.disabled = false;
}

// ============================================================
// Entry point
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
  const btnInit = document.getElementById("btn-init") as HTMLButtonElement;
  const btnInfer = document.getElementById("btn-infer") as HTMLButtonElement;

  btnInit.addEventListener("click", () => {
    initializeKeys().catch((err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err);
      appendLine(`\nError: ${msg}`);
      btnInit.disabled = false;
    });
  });

  btnInfer.addEventListener("click", () => {
    runInference().catch((err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(`Inference error: ${msg}`);
      btnInfer.disabled = false;
    });
  });
});
