// Vitest setup: load Go WASM runtime (wasm_exec.js) before any tests.
// This sets globalThis.Go which is needed by loadLattigo().
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
require(join(__dirname, "..", "wasm", "wasm_exec.js"));
