// Node.js integration test: bootstrap parameter primitives via WASM bridge.
// Run: node js/lattigo/tests/test_bootstrap.js
//
// NOTE: Bootstrap key generation is heavy (5–60s in WASM). This test has a
// long timeout accordingly.

const fs = require("fs");
const path = require("path");

// Load Go WASM support
require(path.join(__dirname, "..", "wasm", "wasm_exec.js"));

const WASM_PATH = path.join(__dirname, "..", "wasm", "lattigo.wasm");

// Test parameters — same as experiments/lattigo-wasm-bootstrap/
const TEST_PARAMS = JSON.stringify({
  LogN: 13,
  LogQ: [29, 26, 26, 26, 26, 26],
  LogP: [29, 29],
  LogDefaultScale: 26,
  H: 8192,
  RingType: "ConjugateInvariant",
});

let passed = 0;
let failed = 0;

function assert(condition, msg) {
  if (!condition) {
    console.error(`  FAIL: ${msg}`);
    failed++;
    return false;
  }
  console.log(`  PASS: ${msg}`);
  passed++;
  return true;
}

function assertNoError(result, fnName) {
  if (result && result.error) {
    console.error(`  FAIL: ${fnName}: unexpected error: ${result.error}`);
    failed++;
    return false;
  }
  console.log(`  PASS: ${fnName}: no error`);
  passed++;
  return true;
}

function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

async function loadWasm() {
  const go = new Go();
  const wasmBuffer = fs.readFileSync(WASM_PATH);
  const result = await WebAssembly.instantiate(wasmBuffer, go.importObject);
  go.run(result.instance);

  // Poll for readiness
  for (let i = 0; i < 100; i++) {
    if (globalThis.lattigo && globalThis.lattigo.__ready) break;
    await new Promise((r) => setTimeout(r, 10));
  }
  if (!globalThis.lattigo || !globalThis.lattigo.__ready) {
    throw new Error("WASM bridge did not become ready");
  }
  return globalThis.lattigo;
}

async function testBootstrapParamsConstruction(bridge, paramsHID) {
  console.log("\n--- testBootstrapParamsConstruction ---");

  // For ConjugateInvariant ring, bootstrap needs LogN+1.
  // This adjustment is done in JS (not in the Go bridge — it's a thin binding).
  const btpLogN = 13 + 1; // LogN=13 + 1 for ConjugateInvariant
  const numSlots = 256;
  const logSlots = Math.log2(numSlots); // 8

  const btpLitJSON = JSON.stringify({
    LogN: btpLogN,
    LogP: [29, 29],
    H: 8192,
    LogSlots: logSlots,
  });

  const btpParamsResult = bridge.newBootstrapParametersFromLiteral(
    paramsHID,
    btpLitJSON,
  );
  assertNoError(btpParamsResult, "newBootstrapParametersFromLiteral");
  assert(
    typeof btpParamsResult.handle === "number" && btpParamsResult.handle > 0,
    `btp params handle is a positive number: ${btpParamsResult.handle}`,
  );

  // Clean up — this handle is just for testing construction
  bridge.deleteHandle(btpParamsResult.handle);
}

async function testBootstrapParamsErrors(bridge, paramsHID) {
  console.log("\n--- testBootstrapParamsErrors ---");

  // Invalid params handle
  const r1 = bridge.newBootstrapParametersFromLiteral(99999, "{}");
  assert(!!r1.error, `invalid params handle: ${r1.error}`);

  // Invalid JSON
  const r2 = bridge.newBootstrapParametersFromLiteral(
    paramsHID,
    "not valid json",
  );
  assert(!!r2.error, `invalid JSON: ${r2.error}`);

  // Missing arguments
  const r3 = bridge.newBootstrapParametersFromLiteral(paramsHID);
  assert(
    r3 === null || r3 === undefined || !!r3.error,
    `missing btpLitJSON argument handled`,
  );
}

async function testBootstrapKeyGenAndMarshalRoundtrip(bridge, paramsHID, skHID) {
  console.log("\n--- testBootstrapKeyGenAndMarshalRoundtrip ---");
  // Use LogSlots=1 (2 slots) to keep bootstrap keys small enough for
  // marshal+unmarshal within WASM's memory limit. With LogSlots=8 (256 slots),
  // the MemEvalKeySet grows to ~1.7 GB which OOMs during unmarshal.

  const btpLogN = 14; // 13+1 for ConjugateInvariant
  const btpLitJSON = JSON.stringify({
    LogN: btpLogN,
    LogP: [29, 29],
    H: 8192,
    LogSlots: 1,
  });

  const btpParamsResult = bridge.newBootstrapParametersFromLiteral(
    paramsHID,
    btpLitJSON,
  );
  assertNoError(btpParamsResult, "newBootstrapParametersFromLiteral (LogSlots=1)");

  console.log(
    "  Generating bootstrap keys with LogSlots=1 (this may take 10-60s)...",
  );
  const t0 = Date.now();

  // btpParamsGenEvaluationKeys returns a Promise
  const result = await bridge.btpParamsGenEvaluationKeys(
    btpParamsResult.handle,
    skHID,
  );
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`  Bootstrap key generation took ${elapsed}s`);

  assert(
    typeof result === "object" && result !== null,
    `btpParamsGenEvaluationKeys returns an object`,
  );
  assert(
    typeof result.evkHID === "number" && result.evkHID > 0,
    `evkHID is a positive number: ${result.evkHID}`,
  );
  assert(
    typeof result.btpEvkHID === "number" && result.btpEvkHID > 0,
    `btpEvkHID is a positive number: ${result.btpEvkHID}`,
  );
  assert(
    result.evkHID !== result.btpEvkHID,
    `evkHID and btpEvkHID are different handles`,
  );

  // Marshal the MemEvaluationKeySet from bootstrap keys
  const evkBytes = bridge.memEvalKeySetMarshal(result.evkHID);
  assert(
    evkBytes instanceof Uint8Array,
    `memEvalKeySetMarshal returns Uint8Array`,
  );
  assert(evkBytes.length > 0, `evk bytes length > 0: ${evkBytes.length}`);
  console.log(
    `  Bootstrap MemEvalKeySet size: ${evkBytes.length} bytes (${(evkBytes.length / 1048576).toFixed(2)} MB)`,
  );

  // Unmarshal
  const unmarshalResult = bridge.memEvalKeySetUnmarshal(evkBytes);
  assertNoError(unmarshalResult, "memEvalKeySetUnmarshal");
  assert(
    unmarshalResult.handle > 0,
    `unmarshaled evk handle is positive: ${unmarshalResult.handle}`,
  );

  // Re-marshal and compare bytes
  const evkBytes2 = bridge.memEvalKeySetMarshal(unmarshalResult.handle);
  assert(evkBytes2 instanceof Uint8Array, `re-marshaled evk is Uint8Array`);
  assert(
    arraysEqual(evkBytes, evkBytes2),
    `MemEvalKeySet marshal roundtrip: bytes match (${evkBytes.length} bytes)`,
  );

  // Clean up
  bridge.deleteHandle(unmarshalResult.handle);
  bridge.deleteHandle(btpParamsResult.handle);
  bridge.deleteHandle(result.evkHID);
  bridge.deleteHandle(result.btpEvkHID);
}

async function testBootstrapKeyGenErrors(bridge) {
  console.log("\n--- testBootstrapKeyGenErrors ---");

  // Invalid btp params handle — should reject the promise
  try {
    await bridge.btpParamsGenEvaluationKeys(99999, 99999);
    assert(false, "should have thrown for invalid btpParams handle");
  } catch (err) {
    assert(
      typeof err === "string" && err.includes("invalid"),
      `invalid btpParams handle rejects: ${err}`,
    );
  }
}

async function main() {
  console.log("Loading WASM bridge...");
  const bridge = await loadWasm();
  console.log("WASM bridge loaded.\n");

  // Setup: create params and keygen
  const paramsResult = bridge.newCKKSParams(TEST_PARAMS);
  assertNoError(paramsResult, "newCKKSParams");
  const paramsHID = paramsResult.handle;

  const kgResult = bridge.newKeyGenerator(paramsHID);
  assertNoError(kgResult, "newKeyGenerator");
  const skResult = bridge.keyGenGenSecretKey(kgResult.handle);
  assertNoError(skResult, "keyGenGenSecretKey");

  // Test 1: Bootstrap params construction (with LogSlots=8, verifies CI adjustment)
  await testBootstrapParamsConstruction(bridge, paramsHID);

  // Test 2: Error handling for params construction
  await testBootstrapParamsErrors(bridge, paramsHID);

  // Test 3+4: Bootstrap key generation + marshal/unmarshal roundtrip
  // Uses LogSlots=1 to keep key sizes manageable in WASM memory
  await testBootstrapKeyGenAndMarshalRoundtrip(
    bridge,
    paramsHID,
    skResult.handle,
  );

  // Test 5: Error handling for key generation
  await testBootstrapKeyGenErrors(bridge);

  // Clean up
  bridge.deleteHandle(paramsHID);
  bridge.deleteHandle(kgResult.handle);
  bridge.deleteHandle(skResult.handle);

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
