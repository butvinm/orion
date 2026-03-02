// Node.js integration test: CKKS parameters and key generation via WASM bridge.
// Run: node js/lattigo/tests/test_params_keygen.js

const fs = require("fs");
const path = require("path");

// Load Go WASM support
require(path.join(__dirname, "..", "wasm", "wasm_exec.js"));

const WASM_PATH = path.join(__dirname, "..", "wasm", "lattigo.wasm");

// Test parameters — small for fast tests
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

function assertType(val, type, msg) {
  return assert(typeof val === type, `${msg} (got ${typeof val}: ${val})`);
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

async function testNewCKKSParams(bridge) {
  console.log("\n--- testNewCKKSParams ---");

  const result = bridge.newCKKSParams(TEST_PARAMS);
  assert(!result.error, `no error (got: ${result.error || "none"})`);
  assertType(result.handle, "number", "handle is a number");
  assert(result.handle > 0, `handle is positive: ${result.handle}`);

  return result.handle;
}

async function testParamAccessors(bridge, paramsHID) {
  console.log("\n--- testParamAccessors ---");

  // MaxSlots: LogN=13, ConjugateInvariant → 2^13 = 8192
  const maxSlots = bridge.ckksMaxSlots(paramsHID);
  assertType(maxSlots, "number", "maxSlots is a number");
  assert(maxSlots === 8192, `maxSlots = ${maxSlots}, expected 8192`);

  // MaxLevel: len(LogQ) - 1 = 5
  const maxLevel = bridge.ckksMaxLevel(paramsHID);
  assertType(maxLevel, "number", "maxLevel is a number");
  assert(maxLevel === 5, `maxLevel = ${maxLevel}, expected 5`);

  // DefaultScale: 2^26
  const defaultScale = bridge.ckksDefaultScale(paramsHID);
  assertType(defaultScale, "number", "defaultScale is a number");
  assert(defaultScale === Math.pow(2, 26), `defaultScale = ${defaultScale}, expected ${Math.pow(2, 26)}`);

  // GaloisElement: for rotation 1
  const galEl = bridge.ckksGaloisElement(paramsHID, 1);
  assertType(galEl, "number", "galoisElement is a number");
  assert(galEl > 0, `galoisElement(1) = ${galEl}, expected > 0`);

  // ModuliChain: should have 6 elements (same as LogQ length)
  const moduliChain = bridge.ckksModuliChain(paramsHID);
  assert(moduliChain.length === 6, `moduliChain length = ${moduliChain.length}, expected 6`);
  assert(moduliChain[0] > 0, `moduliChain[0] > 0`);

  // AuxModuliChain: should have 2 elements (same as LogP length)
  const auxModuliChain = bridge.ckksAuxModuliChain(paramsHID);
  assert(auxModuliChain.length === 2, `auxModuliChain length = ${auxModuliChain.length}, expected 2`);
  assert(auxModuliChain[0] > 0, `auxModuliChain[0] > 0`);
}

async function testNewCKKSParamsErrors(bridge) {
  console.log("\n--- testNewCKKSParamsErrors ---");

  // Invalid JSON
  const r1 = bridge.newCKKSParams("{bad json");
  assert(!!r1.error, `invalid JSON returns error: ${r1.error}`);

  // Invalid ring type
  const r2 = bridge.newCKKSParams(JSON.stringify({
    LogN: 13,
    LogQ: [29, 26],
    LogP: [29],
    LogDefaultScale: 26,
    RingType: "BadType",
  }));
  assert(!!r2.error, `invalid ring type returns error: ${r2.error}`);
}

async function testKeyGeneration(bridge, paramsHID) {
  console.log("\n--- testKeyGeneration ---");

  // Create KeyGenerator
  const kgResult = bridge.newKeyGenerator(paramsHID);
  assert(!kgResult.error, `newKeyGenerator: no error (got: ${kgResult.error || "none"})`);
  const kgHID = kgResult.handle;
  assertType(kgHID, "number", "keyGenerator handle is a number");
  assert(kgHID > 0, `keyGenerator handle is positive: ${kgHID}`);

  // Generate SecretKey
  const skResult = bridge.keyGenGenSecretKey(kgHID);
  assert(!skResult.error, `genSecretKey: no error (got: ${skResult.error || "none"})`);
  const skHID = skResult.handle;
  assertType(skHID, "number", "secretKey handle is a number");
  assert(skHID > 0, `secretKey handle is positive: ${skHID}`);

  // Generate PublicKey
  const pkResult = bridge.keyGenGenPublicKey(kgHID, skHID);
  assert(!pkResult.error, `genPublicKey: no error (got: ${pkResult.error || "none"})`);
  const pkHID = pkResult.handle;
  assertType(pkHID, "number", "publicKey handle is a number");
  assert(pkHID > 0, `publicKey handle is positive: ${pkHID}`);

  // Generate RelinearizationKey
  const rlkResult = bridge.keyGenGenRelinKey(kgHID, skHID);
  assert(!rlkResult.error, `genRelinKey: no error (got: ${rlkResult.error || "none"})`);
  const rlkHID = rlkResult.handle;
  assertType(rlkHID, "number", "relinKey handle is a number");
  assert(rlkHID > 0, `relinKey handle is positive: ${rlkHID}`);

  // Generate GaloisKey
  const galEl = bridge.ckksGaloisElement(paramsHID, 1);
  const gkResult = bridge.keyGenGenGaloisKey(kgHID, skHID, galEl);
  assert(!gkResult.error, `genGaloisKey: no error (got: ${gkResult.error || "none"})`);
  const gkHID = gkResult.handle;
  assertType(gkHID, "number", "galoisKey handle is a number");
  assert(gkHID > 0, `galoisKey handle is positive: ${gkHID}`);

  // All handles should be distinct
  const handles = [kgHID, skHID, pkHID, rlkHID, gkHID];
  const unique = new Set(handles);
  assert(unique.size === handles.length, `all handles are unique: ${handles}`);

  // Clean up
  for (const h of handles) {
    bridge.deleteHandle(h);
  }
}

async function testKeyGenErrors(bridge) {
  console.log("\n--- testKeyGenErrors ---");

  // Invalid handle
  const r1 = bridge.newKeyGenerator(99999);
  assert(!!r1.error, `newKeyGenerator with invalid handle returns error: ${r1.error}`);

  const r2 = bridge.keyGenGenSecretKey(99999);
  assert(!!r2.error, `keyGenGenSecretKey with invalid handle returns error: ${r2.error}`);

  const r3 = bridge.keyGenGenPublicKey(99999, 99998);
  assert(!!r3.error, `keyGenGenPublicKey with invalid handles returns error: ${r3.error}`);
}

async function main() {
  console.log("Loading WASM bridge...");
  const bridge = await loadWasm();
  console.log("WASM bridge loaded.\n");

  const paramsHID = await testNewCKKSParams(bridge);
  await testParamAccessors(bridge, paramsHID);
  await testNewCKKSParamsErrors(bridge);
  await testKeyGeneration(bridge, paramsHID);
  await testKeyGenErrors(bridge);

  // Clean up params
  bridge.deleteHandle(paramsHID);

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
