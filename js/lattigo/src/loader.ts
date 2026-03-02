import type { WasmBridge } from "./types.js";
import { setBridge } from "./bridge.js";

/**
 * Minimal type for Go's WASM runtime class (from wasm_exec.js).
 * The real class is set on `globalThis.Go` after loading wasm_exec.js.
 */
interface GoRuntime {
  importObject: WebAssembly.Imports;
  run(instance: WebAssembly.Instance): Promise<void>;
}

interface GoRuntimeConstructor {
  new (): GoRuntime;
}

declare const Go: GoRuntimeConstructor | undefined;

/** Max time (ms) to wait for the Go bridge to set __ready. */
const READY_TIMEOUT_MS = 5000;
const READY_POLL_MS = 10;

/**
 * Load the Lattigo WASM bridge and return a typed interface to its functions.
 *
 * Works in both Node.js (reads .wasm from disk) and browser (fetches URL).
 *
 * @param wasmPath - Path or URL to lattigo.wasm. Defaults to `wasm/lattigo.wasm`
 *   relative to this module (Node.js) or the current page (browser).
 */
export async function loadLattigo(wasmPath?: string): Promise<WasmBridge> {
  if (typeof Go === "undefined") {
    throw new Error(
      "Go WASM runtime not found. Load wasm_exec.js before calling loadLattigo().",
    );
  }

  // Return existing bridge if already initialized — avoids spawning a second
  // Go WASM runtime that would run select{} indefinitely and leak resources.
  const existing = (globalThis as Record<string, unknown>).lattigo as
    | WasmBridge
    | undefined;
  if (existing?.__ready) {
    setBridge(existing);
    return existing;
  }

  const go = new Go();
  let instance: WebAssembly.Instance;

  if (
    typeof process !== "undefined" &&
    typeof process.versions?.node === "string"
  ) {
    // Node.js: read wasm file from disk
    const fs = await import("fs");
    const path = await import("path");
    const resolvedPath =
      wasmPath ?? path.join(path.dirname(import.meta.url.replace("file://", "")), "..", "wasm", "lattigo.wasm");
    const wasmBuffer = fs.readFileSync(resolvedPath);
    const result = await WebAssembly.instantiate(wasmBuffer, go.importObject);
    instance = result.instance;
  } else {
    // Browser: fetch wasm from URL
    const resolvedPath = wasmPath ?? "wasm/lattigo.wasm";
    if (typeof WebAssembly.instantiateStreaming === "function") {
      const result = await WebAssembly.instantiateStreaming(
        fetch(resolvedPath),
        go.importObject,
      );
      instance = result.instance;
    } else {
      const response = await fetch(resolvedPath);
      const bytes = await response.arrayBuffer();
      const result = await WebAssembly.instantiate(bytes, go.importObject);
      instance = result.instance;
    }
  }

  // Start the Go runtime (non-blocking — it runs `select{}` to stay alive)
  go.run(instance);

  // Poll for readiness: Go's main() sets globalThis.lattigo.__ready = true last
  const deadline = Date.now() + READY_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const lattigo = (globalThis as Record<string, unknown>).lattigo as
      | WasmBridge
      | undefined;
    if (lattigo?.__ready) {
      setBridge(lattigo);
      return lattigo;
    }
    await new Promise((resolve) => setTimeout(resolve, READY_POLL_MS));
  }

  throw new Error(
    `Lattigo WASM bridge did not become ready within ${READY_TIMEOUT_MS}ms`,
  );
}
