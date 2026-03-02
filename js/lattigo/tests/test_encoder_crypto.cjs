// Node.js integration test: encoder, encryptor, decryptor roundtrip via WASM bridge.
// Run: node js/lattigo/tests/test_encoder_crypto.js

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

function assertNoError(result, fnName) {
  return assert(
    !result.error,
    `${fnName}: no error (got: ${result.error || "none"})`,
  );
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

async function testEncoderCreation(bridge, paramsHID) {
  console.log("\n--- testEncoderCreation ---");

  const result = bridge.newEncoder(paramsHID);
  assertNoError(result, "newEncoder");
  assert(result.handle > 0, `encoder handle is positive: ${result.handle}`);

  return result.handle;
}

async function testEncoderErrors(bridge) {
  console.log("\n--- testEncoderErrors ---");

  // Invalid handle
  const r1 = bridge.newEncoder(99999);
  assert(!!r1.error, `newEncoder with invalid handle returns error: ${r1.error}`);

  // Invalid encoder handle for encode
  const r2 = bridge.encoderEncode(99999, [1.0, 2.0], 5, Math.pow(2, 26));
  assert(!!r2.error, `encoderEncode with invalid handle returns error: ${r2.error}`);

  // Invalid encoder handle for decode
  const r3 = bridge.encoderDecode(99999, 99998, 10);
  assert(!!r3.error, `encoderDecode with invalid encoder handle returns error: ${r3.error}`);
}

async function testEncodeDecodeRoundtrip(bridge, paramsHID, encoderHID) {
  console.log("\n--- testEncodeDecodeRoundtrip ---");

  const maxSlots = bridge.ckksMaxSlots(paramsHID);
  const maxLevel = bridge.ckksMaxLevel(paramsHID);
  const defaultScale = bridge.ckksDefaultScale(paramsHID);

  // Create test values
  const numValues = 10;
  const inputValues = [];
  for (let i = 0; i < numValues; i++) {
    inputValues.push((i + 1) * 0.1); // [0.1, 0.2, ..., 1.0]
  }

  // Encode
  const ptResult = bridge.encoderEncode(encoderHID, inputValues, maxLevel, defaultScale);
  assertNoError(ptResult, "encoderEncode");
  assert(ptResult.handle > 0, `plaintext handle is positive: ${ptResult.handle}`);

  // Decode
  const decoded = bridge.encoderDecode(encoderHID, ptResult.handle, maxSlots);
  // encoderDecode returns a Float64Array directly (not an object with handle)
  assert(decoded instanceof Float64Array, `decoded is Float64Array`);
  assert(decoded.length === maxSlots, `decoded length = ${decoded.length}, expected ${maxSlots}`);

  // Check values match within tolerance
  const tolerance = 1e-4;
  let allMatch = true;
  for (let i = 0; i < numValues; i++) {
    const diff = Math.abs(decoded[i] - inputValues[i]);
    if (diff > tolerance) {
      console.error(`  Value mismatch at index ${i}: got ${decoded[i]}, expected ${inputValues[i]}, diff=${diff}`);
      allMatch = false;
    }
  }
  assert(allMatch, `all ${numValues} decoded values match within tolerance ${tolerance}`);

  // Clean up plaintext
  bridge.deleteHandle(ptResult.handle);
}

async function testEncryptorCreation(bridge, paramsHID) {
  console.log("\n--- testEncryptorCreation ---");

  // Create keygen, SK, PK
  const kgResult = bridge.newKeyGenerator(paramsHID);
  assertNoError(kgResult, "newKeyGenerator");
  const skResult = bridge.keyGenGenSecretKey(kgResult.handle);
  assertNoError(skResult, "keyGenGenSecretKey");
  const pkResult = bridge.keyGenGenPublicKey(kgResult.handle, skResult.handle);
  assertNoError(pkResult, "keyGenGenPublicKey");

  // Create encryptor
  const encResult = bridge.newEncryptor(paramsHID, pkResult.handle);
  assertNoError(encResult, "newEncryptor");
  assert(encResult.handle > 0, `encryptor handle is positive: ${encResult.handle}`);

  return {
    kgHID: kgResult.handle,
    skHID: skResult.handle,
    pkHID: pkResult.handle,
    encryptorHID: encResult.handle,
  };
}

async function testDecryptorCreation(bridge, paramsHID, skHID) {
  console.log("\n--- testDecryptorCreation ---");

  const decResult = bridge.newDecryptor(paramsHID, skHID);
  assertNoError(decResult, "newDecryptor");
  assert(decResult.handle > 0, `decryptor handle is positive: ${decResult.handle}`);

  return decResult.handle;
}

async function testCryptoErrors(bridge) {
  console.log("\n--- testCryptoErrors ---");

  // Invalid handles for encryptor
  const r1 = bridge.newEncryptor(99999, 99998);
  assert(!!r1.error, `newEncryptor with invalid params returns error: ${r1.error}`);

  // Invalid handles for decryptor
  const r2 = bridge.newDecryptor(99999, 99998);
  assert(!!r2.error, `newDecryptor with invalid params returns error: ${r2.error}`);

  // Invalid handles for encrypt
  const r3 = bridge.encryptorEncryptNew(99999, 99998);
  assert(!!r3.error, `encryptorEncryptNew with invalid handles returns error: ${r3.error}`);

  // Invalid handles for decrypt
  const r4 = bridge.decryptorDecryptNew(99999, 99998);
  assert(!!r4.error, `decryptorDecryptNew with invalid handles returns error: ${r4.error}`);
}

async function testFullRoundtrip(bridge, paramsHID, encoderHID, encryptorHID, decryptorHID) {
  console.log("\n--- testFullRoundtrip (encode → encrypt → decrypt → decode) ---");

  const maxSlots = bridge.ckksMaxSlots(paramsHID);
  const maxLevel = bridge.ckksMaxLevel(paramsHID);
  const defaultScale = bridge.ckksDefaultScale(paramsHID);

  // Create test values
  const numValues = 8;
  const inputValues = [];
  for (let i = 0; i < numValues; i++) {
    inputValues.push(Math.sin(i)); // some floating point values
  }
  console.log(`  Input values: [${inputValues.map(v => v.toFixed(4)).join(", ")}]`);

  // Encode
  const ptResult = bridge.encoderEncode(encoderHID, inputValues, maxLevel, defaultScale);
  assertNoError(ptResult, "encoderEncode");

  // Encrypt
  const ctResult = bridge.encryptorEncryptNew(encryptorHID, ptResult.handle);
  assertNoError(ctResult, "encryptorEncryptNew");
  assert(ctResult.handle > 0, `ciphertext handle is positive: ${ctResult.handle}`);

  // Decrypt
  const decPtResult = bridge.decryptorDecryptNew(decryptorHID, ctResult.handle);
  assertNoError(decPtResult, "decryptorDecryptNew");
  assert(decPtResult.handle > 0, `decrypted plaintext handle is positive: ${decPtResult.handle}`);

  // Decode
  const decoded = bridge.encoderDecode(encoderHID, decPtResult.handle, maxSlots);
  assert(decoded instanceof Float64Array, `decoded is Float64Array`);
  assert(decoded.length === maxSlots, `decoded length = ${decoded.length}, expected ${maxSlots}`);

  // Check values match within CKKS tolerance
  const tolerance = 1e-3;
  let allMatch = true;
  for (let i = 0; i < numValues; i++) {
    const diff = Math.abs(decoded[i] - inputValues[i]);
    if (diff > tolerance) {
      console.error(`  Value mismatch at index ${i}: got ${decoded[i]}, expected ${inputValues[i]}, diff=${diff}`);
      allMatch = false;
    }
  }
  assert(allMatch, `all ${numValues} roundtrip values match within tolerance ${tolerance}`);

  console.log(`  Decoded values:  [${Array.from(decoded).slice(0, numValues).map(v => v.toFixed(4)).join(", ")}]`);

  // Clean up
  bridge.deleteHandle(ptResult.handle);
  bridge.deleteHandle(ctResult.handle);
  bridge.deleteHandle(decPtResult.handle);
}

async function main() {
  console.log("Loading WASM bridge...");
  const bridge = await loadWasm();
  console.log("WASM bridge loaded.\n");

  // Setup: create params and encoder
  const paramsResult = bridge.newCKKSParams(TEST_PARAMS);
  assertNoError(paramsResult, "newCKKSParams");
  const paramsHID = paramsResult.handle;

  // Test encoder
  const encoderHID = await testEncoderCreation(bridge, paramsHID);
  await testEncoderErrors(bridge);
  await testEncodeDecodeRoundtrip(bridge, paramsHID, encoderHID);

  // Test encryptor/decryptor creation
  const keys = await testEncryptorCreation(bridge, paramsHID);
  const decryptorHID = await testDecryptorCreation(bridge, paramsHID, keys.skHID);
  await testCryptoErrors(bridge);

  // Full roundtrip: encode → encrypt → decrypt → decode
  await testFullRoundtrip(bridge, paramsHID, encoderHID, keys.encryptorHID, decryptorHID);

  // Clean up
  bridge.deleteHandle(paramsHID);
  bridge.deleteHandle(encoderHID);
  bridge.deleteHandle(keys.kgHID);
  bridge.deleteHandle(keys.skHID);
  bridge.deleteHandle(keys.pkHID);
  bridge.deleteHandle(keys.encryptorHID);
  bridge.deleteHandle(decryptorHID);

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
