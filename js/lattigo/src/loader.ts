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
 * @param wasmPath - Path or URL to lattigo.wasm. In browsers this is fetched
 *   relative to the current page. In Node.js it is read from disk via fs.
 *   **Required** — there is no default.
 */
export async function loadLattigo(wasmPath: string): Promise<WasmBridge> {
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

  const isBrowser =
    typeof globalThis.window !== "undefined" ||
    typeof globalThis.self !== "undefined";

  if (isBrowser) {
    // Browser: fetch the WASM file over HTTP.
    if (typeof WebAssembly.instantiateStreaming === "function") {
      const result = await WebAssembly.instantiateStreaming(
        fetch(wasmPath),
        go.importObject,
      );
      instance = result.instance;
    } else {
      const response = await fetch(wasmPath);
      const bytes = await response.arrayBuffer();
      const result = await WebAssembly.instantiate(bytes, go.importObject);
      instance = result.instance;
    }
  } else {
    // Node.js: read the WASM file from disk.
    // Use a computed specifier so bundlers cannot statically resolve it.
    const modName = ["f", "s"].join("");
    const fs = await import(/* @vite-ignore */ modName);
    const wasmBuffer = fs.readFileSync(wasmPath);
    const result = await WebAssembly.instantiate(wasmBuffer, go.importObject);
    instance = result.instance;
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
