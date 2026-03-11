import type { WasmBridge, Handle } from "./types.js";

let _bridge: WasmBridge | null = null;

/** Set the active bridge instance (called by loadLattigo). */
export function setBridge(bridge: WasmBridge): void {
  _bridge = bridge;
}

/** Get the active bridge instance. Throws if loadLattigo() hasn't been called. */
export function getBridge(): WasmBridge {
  if (_bridge === null) {
    throw new Error(
      "Lattigo WASM bridge not loaded. Call loadLattigo() first.",
    );
  }
  return _bridge;
}

/**
 * Shared FinalizationRegistry — catches forgotten .close() calls.
 * Each class registers its handle; .close() unregisters to avoid double-delete.
 */
export const registry = new FinalizationRegistry((handleId: Handle) => {
  if (_bridge !== null) {
    _bridge.deleteHandle(handleId);
  }
});
