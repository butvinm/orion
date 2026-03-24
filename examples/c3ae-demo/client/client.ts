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

  // 8. Bootstrap keys (streamed one at a time to minimize memory)
  if (manifest.bootstrap_slots.length > 0 && bridge) {
    const t5 = performance.now();

    // 8a. Create bootstrap params
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
    const btpParamsHID = (btpParamsResult as { handle: number }).handle;

    // 8b. Extend SK to bootstrap ring
    appendLine("Extending secret key to bootstrap parameters...");
    const extResult = bridge.bootstrapExtendSK(btpParamsHID, sk!.handle);
    if ("error" in extResult) {
      setStatus(`Bootstrap extend SK error: ${(extResult as { error: string }).error}`);
      btnInit.disabled = false;
      return;
    }
    const skN2HID = extResult.skN2HID as number;
    const kgN2HID = extResult.kgN2HID as number;

    // 8c. Generate + upload switching keys (small, 2-6 keys)
    appendLine("Generating bootstrap switching keys...");
    const switchResult = bridge.bootstrapGenSwitchingKeys(btpParamsHID, sk!.handle, skN2HID);
    if ("error" in switchResult) {
      setStatus(`Bootstrap switching keys error: ${(switchResult as { error: string }).error}`);
      btnInit.disabled = false;
      return;
    }
    const switchKeys = switchResult.keys as Array<{ hid: number; name: string }>;
    for (const { hid, name } of switchKeys) {
      const keyBytes = bridge.evalKeyMarshal(hid);
      bridge.deleteHandle(hid);
      if ("error" in keyBytes) {
        setStatus(`Bootstrap switching key marshal error: ${(keyBytes as { error: string }).error}`);
        btnInit.disabled = false;
        return;
      }
      const resp = await fetch(`/session/${sessionId}/keys/bootstrap/switching/${name}`, {
        method: "POST",
        body: keyBytes as unknown as ArrayBuffer,
        headers: { "Content-Type": "application/octet-stream" },
      });
      if (!resp.ok) {
        setStatus(`Bootstrap switching key upload failed: ${await resp.text()}`);
        btnInit.disabled = false;
        return;
      }
      appendLine(`  Switching key "${name}" uploaded (${formatBytes((keyBytes as Uint8Array).length)})`);
    }

    // 8d. Generate + upload bootstrap RLK
    appendLine("Generating bootstrap relinearization key...");
    const btpRlk = bridge.keyGenGenRelinKey(kgN2HID, skN2HID);
    if ("error" in btpRlk) {
      setStatus(`Bootstrap RLK error: ${(btpRlk as { error: string }).error}`);
      btnInit.disabled = false;
      return;
    }
    const btpRlkHID = (btpRlk as { handle: number }).handle;
    const btpRlkBytes = bridge.relinKeyMarshal(btpRlkHID);
    bridge.deleteHandle(btpRlkHID);
    {
      const resp = await fetch(`/session/${sessionId}/keys/bootstrap/relin`, {
        method: "POST",
        body: btpRlkBytes as unknown as ArrayBuffer,
        headers: { "Content-Type": "application/octet-stream" },
      });
      if (!resp.ok) {
        setStatus(`Bootstrap RLK upload failed: ${await resp.text()}`);
        btnInit.disabled = false;
        return;
      }
    }
    appendLine(`  Bootstrap RLK uploaded (${formatBytes((btpRlkBytes as Uint8Array).length)})`);

    // 8e. Stream bootstrap Galois keys one at a time
    const btpGaloisElements = bridge.bootstrapGaloisElements(btpParamsHID) as number[];
    const btpTotal = btpGaloisElements.length;
    appendLine(`Streaming ${btpTotal} bootstrap Galois key(s)...`);
    const t6 = performance.now();
    for (let i = 0; i < btpTotal; i++) {
      const ge = btpGaloisElements[i];
      const gkResult = bridge.keyGenGenGaloisKey(kgN2HID, skN2HID, ge);
      if ("error" in gkResult) {
        setStatus(`Bootstrap Galois key error (${ge}): ${(gkResult as { error: string }).error}`);
        btnInit.disabled = false;
        return;
      }
      const gkHID = (gkResult as { handle: number }).handle;
      const gkBytes = bridge.galoisKeyMarshal(gkHID);
      bridge.deleteHandle(gkHID);

      const resp = await fetch(`/session/${sessionId}/keys/bootstrap/galois/${ge}`, {
        method: "POST",
        body: gkBytes as unknown as ArrayBuffer,
        headers: { "Content-Type": "application/octet-stream" },
      });
      if (!resp.ok) {
        setStatus(`Bootstrap Galois key upload failed (${ge}): ${await resp.text()}`);
        btnInit.disabled = false;
        return;
      }
      setStatus(
        `Bootstrap Galois keys: ${i + 1}/${btpTotal} [${Math.round(((i + 1) / btpTotal) * 100)}%] — ` +
          `${formatDuration(performance.now() - t6)} elapsed`,
      );
    }
    appendLine(
      `${btpTotal} bootstrap Galois key(s) uploaded in ${formatDuration(performance.now() - t6)}`,
    );

    // Cleanup bootstrap handles
    bridge.deleteHandle(skN2HID);
    bridge.deleteHandle(kgN2HID);
    bridge.deleteHandle(btpParamsHID);

    appendLine(`Bootstrap keys complete in ${formatDuration(performance.now() - t5)}`);
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

  // Split into chunks of maxSlots, encode + encrypt each
  const maxSlots = params.maxSlots();
  setStatus("Encoding and encrypting...");
  const tEnc = performance.now();

  const defaultScale = params.defaultScale();
  const ctChunks: Uint8Array[] = [];
  for (let off = 0; off < values.length; off += maxSlots) {
    const chunk = values.slice(off, off + maxSlots);
    const padded = new Array<number>(maxSlots).fill(0);
    for (let i = 0; i < chunk.length; i++) padded[i] = chunk[i];
    const pt = encoder.encode(padded, inputLevel, defaultScale);
    const ct = encryptor.encryptNew(pt);
    ctChunks.push(ct.marshalBinary());
    ct.close();
    pt.close();
  }

  // Pack as length-prefixed: [u32 count][u64 len][bytes]...
  let totalSize = 4;
  for (const c of ctChunks) totalSize += 8 + c.length;
  const packed = new Uint8Array(totalSize);
  const dv = new DataView(packed.buffer);
  dv.setUint32(0, ctChunks.length, true);
  let pos = 4;
  for (const c of ctChunks) {
    dv.setUint32(pos, c.length, true);
    dv.setUint32(pos + 4, 0, true); // high 32 bits of u64
    pos += 8;
    packed.set(c, pos);
    pos += c.length;
  }

  const encTime = performance.now() - tEnc;

  // POST to server
  setStatus(
    `Sending ${ctChunks.length} ciphertext(s) (${formatBytes(totalSize)}) for FHE inference...`,
  );
  const tInfer = performance.now();

  let resultBytes: Uint8Array;
  let inferenceTimeHeader: string | null = null;
  try {
    const inferResp = await fetch(`/session/${sessionId}/infer`, {
      method: "POST",
      body: packed as unknown as ArrayBuffer,
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

  // Decrypt first result CT from length-prefixed response
  setStatus("Decrypting result...");
  const tDec = performance.now();

  const rdv = new DataView(resultBytes.buffer);
  // Skip count (4 bytes) + first length (8 bytes)
  const firstLen = rdv.getUint32(4, true);
  const firstCt = resultBytes.slice(12, 12 + firstLen);
  const resultCt = Ciphertext.unmarshalBinary(firstCt);
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
      `  Ciphertext : ${formatBytes(totalSize)} (${ctChunks.length} CT)\n` +
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
