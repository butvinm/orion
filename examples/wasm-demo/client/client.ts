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
 *        call bootstrapParamsGenEvalKeys (async, may take 5–30s)
 *     8. POST /session (no body) → session ID
 *     9. Stream keys: generate-marshal-upload-close loop (one key in memory at a time)
 *    10. POST /session/{id}/keys/finalize
 *
 *   Run Inference:
 *     11. Encode + encrypt user input
 *     12. POST /session/{id}/infer → result ciphertext bytes
 *     13. Decrypt + decode → display values with timing breakdown
 */

import {
  loadLattigo,
  CKKSParameters,
  KeyGenerator,
  Encoder,
  Encryptor,
  Decryptor,
  Ciphertext,
} from "orion-v2-lattigo";
import type { WasmBridge, SecretKey } from "orion-v2-lattigo";

// ============================================================
// Types matching the /params response from the Go server
// ============================================================

interface OrionParams {
  logn: number;
  logq: number[];
  logp: number[];
  log_default_scale: number;
  h: number;
  ring_type: string;
  boot_logp?: number[];
  btp_logn?: number;
}

interface KeyManifest {
  galois_elements: number[];
  bootstrap_slots: number[];
  boot_logp?: number[];
  btp_logn?: number;
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

  // Free previously allocated Go handles before overwriting the module-level
  // variables. Without this, each re-initialization leaks the old handles.
  params?.close();
  params = null;
  sk?.close();
  sk = null;
  encoder?.close();
  encoder = null;
  encryptor?.close();
  encryptor = null;
  decryptor?.close();
  decryptor = null;

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

  // 3. Create CKKS params (flat args, no JSON)
  // When bootstrap is enabled, set logNthRoot = btp_logn + 1 so that generated
  // primes satisfy q = 1 mod 2^(btp_logn+1), required by Lattigo's bootstrap.
  let logNthRoot: number | undefined;
  if (ckksData.boot_logp && ckksData.boot_logp.length > 0) {
    const btpLogN = ckksData.btp_logn ?? ckksData.logn;
    logNthRoot = btpLogN + 1;
  }
  params = new CKKSParameters({
    logN: ckksData.logn,
    logQ: ckksData.logq,
    logP: ckksData.logp,
    logDefaultScale: ckksData.log_default_scale,
    ringType: ckksData.ring_type as "standard" | "conjugate_invariant",
    h: ckksData.h,
    logNthRoot,
  });
  appendLine(
    `CKKS params: ${params.maxSlots()} slots, max level ${params.maxLevel()}`,
  );

  // 4. Keygen
  appendLine("Generating secret key...");
  const t2 = performance.now();
  const kg = new KeyGenerator(params);
  sk = kg.genSecretKey();
  const pk = kg.genPublicKey(sk);
  appendLine(`Keygen in ${formatDuration(performance.now() - t2)}`);

  // 5. Create encoder, encryptor, decryptor
  encoder = new Encoder(params);
  encryptor = new Encryptor(params, pk);
  decryptor = new Decryptor(params, sk);

  // 6. Create session (no body — keys will be streamed individually)
  appendLine("Creating session...");
  let sessData: { session_id: string };
  try {
    const sessResp = await fetch("/session", { method: "POST" });
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
  appendLine(`Session created: ${sessionId}`);

  // 7. Upload RLK if needed (generate → marshal → upload → close)
  if (manifest.needs_rlk) {
    appendLine("Generating and uploading relinearization key...");
    const t3 = performance.now();
    const rlk = kg.genRelinKey(sk);
    const rlkBytes = rlk.marshalBinary();
    rlk.close();
    try {
      const resp = await fetch(`/session/${sessionId}/keys/relin`, {
        method: "POST",
        body: rlkBytes as unknown as ArrayBuffer,
        headers: { "Content-Type": "application/octet-stream" },
      });
      if (!resp.ok) {
        const msg = await resp.text();
        setStatus(`RLK upload failed: ${msg}`);
        btnInit.disabled = false;
        return;
      }
    } catch (err) {
      setStatus(`RLK upload error: ${(err as Error).message}`);
      btnInit.disabled = false;
      return;
    }
    appendLine(
      `RLK uploaded in ${formatDuration(performance.now() - t3)} (${formatBytes(rlkBytes.length)})`,
    );
  }

  // 8. Stream Galois keys: generate → marshal → upload → close (one key in memory at a time)
  if (manifest.galois_elements.length > 0) {
    const total = manifest.galois_elements.length;
    appendLine(`Uploading ${total} Galois key(s)...`);
    const t4 = performance.now();
    for (let i = 0; i < total; i++) {
      const ge = manifest.galois_elements[i];
      const gk = kg.genGaloisKey(sk, ge);
      const gkBytes = gk.marshalBinary();
      gk.close();

      try {
        const resp = await fetch(
          `/session/${sessionId}/keys/galois/${ge}`,
          {
            method: "POST",
            body: gkBytes as unknown as ArrayBuffer,
            headers: { "Content-Type": "application/octet-stream" },
          },
        );
        if (!resp.ok) {
          const msg = await resp.text();
          setStatus(`Galois key upload failed (element=${ge}): ${msg}`);
          btnInit.disabled = false;
          return;
        }
      } catch (err) {
        setStatus(
          `Galois key upload error (element=${ge}): ${(err as Error).message}`,
        );
        btnInit.disabled = false;
        return;
      }

      const pct = Math.round(((i + 1) / total) * 100);
      setStatus(
        `Galois keys: ${i + 1}/${total} [${pct}%] — element=${ge} — ` +
          `${formatDuration(performance.now() - t4)} elapsed`,
      );
    }
    appendLine(
      `${total} Galois key(s) uploaded in ${formatDuration(performance.now() - t4)}`,
    );
  }

  // 8b. Upload bootstrap keys if model requires bootstrap
  if (manifest.bootstrap_slots.length > 0 && bridge) {
    appendLine("Generating bootstrap evaluation keys (this may take a while)...");
    const t5 = performance.now();

    // Construct bootstrap parameters (flat args)
    const btpLogN = ckksData.btp_logn ?? ckksData.logn;
    const btpLogP = (manifest.boot_logp && manifest.boot_logp.length > 0)
      ? manifest.boot_logp : null;
    const btpH = ckksData.h ?? null;
    const minSlots = Math.min(...manifest.bootstrap_slots);
    const btpLogSlots = Math.round(Math.log2(minSlots));

    const btpParamsResult = bridge.newBootstrapParams(
      params!.handle,
      btpLogN,
      btpLogP,
      btpH,
      btpLogSlots,
    );
    if ("error" in btpParamsResult) {
      setStatus(`Bootstrap params error: ${btpParamsResult.error}`);
      btnInit.disabled = false;
      return;
    }
    const btpParamsHID = btpParamsResult.handle;

    // Generate bootstrap evaluation keys (async — heavy operation)
    const btpResult = await bridge.bootstrapParamsGenEvalKeys(
      btpParamsHID,
      sk!.handle,
    );
    bridge.deleteHandle(btpParamsHID);

    // Marshal bootstrap keys
    const btpBytes = bridge.bootstrapEvalKeysMarshal(btpResult.btpEvkHID);
    bridge.deleteHandle(btpResult.evkHID);
    bridge.deleteHandle(btpResult.btpEvkHID);

    if ("error" in btpBytes) {
      setStatus(`Bootstrap keys marshal error: ${btpBytes.error}`);
      btnInit.disabled = false;
      return;
    }

    appendLine(
      `Bootstrap keys generated in ${formatDuration(performance.now() - t5)} (${formatBytes(btpBytes.length)})`,
    );

    // Upload bootstrap keys
    appendLine("Uploading bootstrap keys...");
    const t6 = performance.now();
    try {
      const resp = await fetch(
        `/session/${sessionId}/keys/bootstrap`,
        {
          method: "POST",
          body: btpBytes as unknown as ArrayBuffer,
          headers: { "Content-Type": "application/octet-stream" },
        },
      );
      if (!resp.ok) {
        const msg = await resp.text();
        setStatus(`Bootstrap key upload failed: ${msg}`);
        btnInit.disabled = false;
        return;
      }
    } catch (err) {
      setStatus(
        `Bootstrap key upload error: ${(err as Error).message}`,
      );
      btnInit.disabled = false;
      return;
    }
    appendLine(
      `Bootstrap keys uploaded in ${formatDuration(performance.now() - t6)}`,
    );
  }

  // 9. Finalize session — server validates all keys present and creates evaluator
  appendLine("Finalizing session...");
  try {
    const finalizeResp = await fetch(
      `/session/${sessionId}/keys/finalize`,
      { method: "POST" },
    );
    if (!finalizeResp.ok) {
      const msg = await finalizeResp.text();
      setStatus(`Finalize failed: ${msg}`);
      btnInit.disabled = false;
      return;
    }
  } catch (err) {
    setStatus(`Finalize error: ${(err as Error).message}`);
    btnInit.disabled = false;
    return;
  }
  appendLine("Session finalized — evaluator ready.");

  // Cleanup (PK, KG — SK/encoder/encryptor/decryptor kept for inference)
  pk.close();
  kg.close();

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
  pt.close();
  ct.close();

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
  resultCt.close();
  resultPt.close();

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
