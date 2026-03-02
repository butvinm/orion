import { loadLattigo } from "../src/loader.js";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const WASM_PATH = join(__dirname, "..", "wasm", "lattigo.wasm");

let loaded = false;

/** Ensure WASM bridge is loaded exactly once across all tests. */
export async function ensureWasmLoaded(): Promise<void> {
  if (!loaded) {
    await loadLattigo(WASM_PATH);
    loaded = true;
  }
}

/** Small CKKS parameters for fast tests. */
export const TEST_PARAMS = {
  logN: 13,
  logQ: [29, 26, 26, 26, 26, 26],
  logP: [29, 29],
  logDefaultScale: 26,
  ringType: "ConjugateInvariant" as const,
};
