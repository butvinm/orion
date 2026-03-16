/**
 * Browser client for C3AE age verification FHE demo.
 *
 * Flow:
 *   1. Initialize keys (same as wasm-demo)
 *   2. User uploads face image (64x64 RGB)
 *   3. Image is preprocessed, encoded, encrypted in browser
 *   4. Ciphertext sent to server for FHE inference
 *   5. Result decrypted in browser → age classification
 *
 * Secret key never leaves the browser.
 */

import {
  loadLattigo,
  CKKSParameters,
  KeyGenerator,
  Encoder,
  Encryptor,
  Decryptor,
  Ciphertext,
} from "@orion/lattigo";
import type { WasmBridge, SecretKey } from "@orion/lattigo";

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

// Application state
let bridge: WasmBridge | null = null;
let params: CKKSParameters | null = null;
let sk: SecretKey | null = null;
let encryptor: Encryptor | null = null;
let decryptor: Decryptor | null = null;
let encoder: Encoder | null = null;
let sessionId: string | null = null;
let inputLevel = 0;

// UI helpers
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

/**
 * Preprocess a 64x64 image from canvas RGBA data to CHW float array normalized to [-1, 1].
 */
function preprocessImage(
  canvas: HTMLCanvasElement,
): { values: number[]; width: number; height: number } {
  const ctx = canvas.getContext("2d")!;
  const imageData = ctx.getImageData(0, 0, 64, 64);
  const rgba = imageData.data; // Uint8ClampedArray [R,G,B,A, R,G,B,A, ...]

  // Convert RGBA HWC -> CHW float [-1, 1]
  const C = 3,
    H = 64,
    W = 64;
  const values = new Array<number>(C * H * W);

  for (let c = 0; c < C; c++) {
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        const pixelIdx = (h * W + w) * 4 + c; // RGBA offset
        const normalized = rgba[pixelIdx] / 255.0;
        const scaled = (normalized - 0.5) / 0.5; // [-1, 1]
        values[c * H * W + h * W + w] = scaled;
      }
    }
  }

  return { values, width: W, height: H };
}

// ============================================================
// Initialize keys (same pattern as wasm-demo)
// ============================================================

async function initializeKeys(): Promise<void> {
  const btnInit = document.getElementById("btn-init") as HTMLButtonElement;
  const btnInfer = document.getElementById("btn-infer") as HTMLButtonElement;
  btnInit.disabled = true;
  btnInfer.disabled = true;

  // Free previous handles
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

  setStatus("Fetching parameters from server...");
  const tTotal = performance.now();

  // 1. GET /params
  let serverParams: ParamsResponse;
  try {
    const resp = await fetch("/params");
    if (!resp.ok) {
      setStatus(`Error fetching /params: ${resp.status}`);
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
      `input_level=${inputLevel}, galois_keys=${manifest.galois_elements.length}, ` +
      `bootstrap=${manifest.bootstrap_slots.length > 0 ? "yes" : "no"}`,
  );

  // 2. Load WASM
  appendLine("Loading Lattigo WASM...");
  const t1 = performance.now();
  bridge = await loadLattigo("wasm/lattigo.wasm");
  appendLine(`WASM loaded in ${formatDuration(performance.now() - t1)}`);

  // 3. Create CKKS params
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
  appendLine("Generating keys...");
  const t2 = performance.now();
  const kg = new KeyGenerator(params);
  sk = kg.genSecretKey();
  const pk = kg.genPublicKey(sk);
  appendLine(`Keygen in ${formatDuration(performance.now() - t2)}`);

  encoder = new Encoder(params);
  encryptor = new Encryptor(params, pk);
  decryptor = new Decryptor(params, sk);

  // 5. Create session
  appendLine("Creating session...");
  let sessData: { session_id: string };
  try {
    const sessResp = await fetch("/session", { method: "POST" });
    if (!sessResp.ok) {
      setStatus(`Session creation failed: ${await sessResp.text()}`);
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
  appendLine(`Session: ${sessionId}`);

  // 6. Upload RLK
  if (manifest.needs_rlk) {
    appendLine("Uploading relinearization key...");
    const t3 = performance.now();
    const rlk = kg.genRelinKey(sk);
    const rlkBytes = rlk.marshalBinary();
    rlk.close();
    const resp = await fetch(`/session/${sessionId}/keys/relin`, {
      method: "POST",
      body: rlkBytes as unknown as ArrayBuffer,
      headers: { "Content-Type": "application/octet-stream" },
    });
    if (!resp.ok) {
      setStatus(`RLK upload failed: ${await resp.text()}`);
      btnInit.disabled = false;
      return;
    }
    appendLine(
      `RLK uploaded in ${formatDuration(performance.now() - t3)} (${formatBytes(rlkBytes.length)})`,
    );
  }

  // 7. Stream Galois keys
  if (manifest.galois_elements.length > 0) {
    const total = manifest.galois_elements.length;
    appendLine(`Uploading ${total} Galois key(s)...`);
    const t4 = performance.now();
    for (let i = 0; i < total; i++) {
      const ge = manifest.galois_elements[i];
      const gk = kg.genGaloisKey(sk, ge);
      const gkBytes = gk.marshalBinary();
      gk.close();
      const resp = await fetch(`/session/${sessionId}/keys/galois/${ge}`, {
        method: "POST",
        body: gkBytes as unknown as ArrayBuffer,
        headers: { "Content-Type": "application/octet-stream" },
      });
      if (!resp.ok) {
        setStatus(`Galois key upload failed (${ge}): ${await resp.text()}`);
        btnInit.disabled = false;
        return;
      }
      setStatus(
        `Galois keys: ${i + 1}/${total} [${Math.round(((i + 1) / total) * 100)}%] — ` +
          `${formatDuration(performance.now() - t4)} elapsed`,
      );
    }
    appendLine(
      `${total} Galois key(s) uploaded in ${formatDuration(performance.now() - t4)}`,
    );
  }

  // 8. Bootstrap keys
  if (manifest.bootstrap_slots.length > 0 && bridge) {
    appendLine(
      "Generating bootstrap keys (this may take a while)...",
    );
    const t5 = performance.now();

    const btpLogN = ckksData.btp_logn ?? ckksData.logn;
    const btpLogP =
      manifest.boot_logp && manifest.boot_logp.length > 0
        ? manifest.boot_logp
        : null;
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

    const btpResult = await bridge.bootstrapParamsGenEvalKeys(
      btpParamsHID,
      sk!.handle,
    );
    bridge.deleteHandle(btpParamsHID);

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

    appendLine("Uploading bootstrap keys...");
    const t6 = performance.now();
    const resp = await fetch(`/session/${sessionId}/keys/bootstrap`, {
      method: "POST",
      body: btpBytes as unknown as ArrayBuffer,
      headers: { "Content-Type": "application/octet-stream" },
    });
    if (!resp.ok) {
      setStatus(`Bootstrap key upload failed: ${await resp.text()}`);
      btnInit.disabled = false;
      return;
    }
    appendLine(
      `Bootstrap keys uploaded in ${formatDuration(performance.now() - t6)}`,
    );
  }

  // 9. Finalize
  appendLine("Finalizing session...");
  try {
    const finalizeResp = await fetch(
      `/session/${sessionId}/keys/finalize`,
      { method: "POST" },
    );
    if (!finalizeResp.ok) {
      setStatus(`Finalize failed: ${await finalizeResp.text()}`);
      btnInit.disabled = false;
      return;
    }
  } catch (err) {
    setStatus(`Finalize error: ${(err as Error).message}`);
    btnInit.disabled = false;
    return;
  }

  // Cleanup
  pk.close();
  kg.close();

  const total = performance.now() - tTotal;
  appendLine(`\nTotal initialization: ${formatDuration(total)}`);
  appendLine("Ready — upload a face image and click Encrypt & Infer.");
  setOutput("Keys initialized. Upload a face image to begin.");
  btnInfer.disabled = false;
}

// ============================================================
// Image handling
// ============================================================

function loadImageToCanvas(file: File): Promise<void> {
  return new Promise((resolve, reject) => {
    const canvas = document.getElementById("preview") as HTMLCanvasElement;
    const ctx = canvas.getContext("2d")!;
    const img = new Image();
    img.onload = () => {
      // Resize to 64x64
      canvas.width = 64;
      canvas.height = 64;
      ctx.drawImage(img, 0, 0, 64, 64);
      URL.revokeObjectURL(img.src);
      resolve();
    };
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = URL.createObjectURL(file);
  });
}

// ============================================================
// Run inference
// ============================================================

async function runInference(): Promise<void> {
  if (!sessionId || !encryptor || !decryptor || !encoder || !params) {
    setStatus("Error: not initialized");
    return;
  }

  const canvas = document.getElementById("preview") as HTMLCanvasElement;
  if (canvas.width === 0) {
    setStatus("Error: no image loaded");
    return;
  }

  const btnInfer = document.getElementById("btn-infer") as HTMLButtonElement;
  btnInfer.disabled = true;

  // Preprocess image
  setStatus("Preprocessing image...");
  const { values } = preprocessImage(canvas);

  // Pad to maxSlots
  const maxSlots = params.maxSlots();
  const padded = new Array<number>(maxSlots).fill(0);
  for (let i = 0; i < Math.min(values.length, maxSlots); i++) {
    padded[i] = values[i];
  }

  // Encode + encrypt
  setStatus("Encoding and encrypting...");
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
    `Sending ciphertext (${formatBytes(ctBytes.length)}) for FHE inference...`,
  );
  const tInfer = performance.now();

  let resultBytes: Uint8Array;
  let inferenceTimeHeader: string | null = null;
  try {
    const inferResp = await fetch(`/session/${sessionId}/infer`, {
      method: "POST",
      body: ctBytes as unknown as ArrayBuffer,
      headers: { "Content-Type": "application/octet-stream" },
    });
    if (!inferResp.ok) {
      setStatus(`Inference error: ${await inferResp.text()}`);
      btnInfer.disabled = false;
      return;
    }
    inferenceTimeHeader = inferResp.headers.get("X-Inference-Time");
    resultBytes = new Uint8Array(await inferResp.arrayBuffer());
  } catch (err) {
    setStatus(`Network error: ${(err as Error).message}`);
    btnInfer.disabled = false;
    return;
  }

  const inferTime = performance.now() - tInfer;

  // Decrypt
  setStatus("Decrypting result...");
  const tDec = performance.now();

  const resultCt = Ciphertext.unmarshalBinary(resultBytes);
  const resultPt = decryptor.decryptNew(resultCt);
  const outputValues = encoder.decode(resultPt, maxSlots);
  resultCt.close();
  resultPt.close();

  const decTime = performance.now() - tDec;

  // Interpret result
  const logit = outputValues[0];
  const prob = 1 / (1 + Math.exp(-logit));
  const prediction = prob >= 0.5 ? "ADULT (18+)" : "MINOR (<18)";

  setOutput(
    `Prediction: ${prediction}\n` +
      `P(adult):   ${(prob * 100).toFixed(1)}%\n` +
      `Raw logit:  ${logit.toFixed(6)}\n\n` +
      `Timing:\n` +
      `  Encode + Encrypt : ${formatDuration(encTime)}\n` +
      `  Server inference : ${formatDuration(inferTime)}` +
      (inferenceTimeHeader ? ` (server-side: ${inferenceTimeHeader})` : "") +
      `\n` +
      `  Decrypt + Decode : ${formatDuration(decTime)}\n\n` +
      `Sizes:\n` +
      `  Ciphertext : ${formatBytes(ctBytes.length)}\n` +
      `  Result     : ${formatBytes(resultBytes.length)}`,
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
  const fileInput = document.getElementById("file-input") as HTMLInputElement;

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

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    try {
      await loadImageToCanvas(file);
      appendLine(`Image loaded: ${file.name}`);
    } catch (err) {
      setStatus(`Image error: ${(err as Error).message}`);
    }
  });
});
