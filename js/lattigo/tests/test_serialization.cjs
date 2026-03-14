// Node.js integration test: serialization (marshal/unmarshal) via WASM bridge.
// Run: node js/lattigo/tests/test_serialization.js

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
  if (result && result.error) {
    console.error(`  FAIL: ${fnName}: unexpected error: ${result.error}`);
    failed++;
    return false;
  }
  console.log(`  PASS: ${fnName}: no error`);
  passed++;
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

function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

async function testSecretKeyMarshalRoundtrip(bridge, skHID) {
  console.log("\n--- testSecretKeyMarshalRoundtrip ---");

  const bytes = bridge.secretKeyMarshal(skHID);
  assert(bytes instanceof Uint8Array, `secretKeyMarshal returns Uint8Array`);
  assert(bytes.length > 0, `secretKeyMarshal bytes length > 0: ${bytes.length}`);

  const unmarshalResult = bridge.secretKeyUnmarshal(bytes);
  assertNoError(unmarshalResult, "secretKeyUnmarshal");
  assert(unmarshalResult.handle > 0, `unmarshaled SK handle is positive: ${unmarshalResult.handle}`);

  // Re-marshal and compare bytes
  const bytes2 = bridge.secretKeyMarshal(unmarshalResult.handle);
  assert(bytes2 instanceof Uint8Array, `re-marshaled SK is Uint8Array`);
  assert(
    arraysEqual(bytes, bytes2),
    `SK marshal roundtrip: bytes match (${bytes.length} bytes)`,
  );

  bridge.deleteHandle(unmarshalResult.handle);
}

async function testPublicKeyMarshalRoundtrip(bridge, pkHID) {
  console.log("\n--- testPublicKeyMarshalRoundtrip ---");

  const bytes = bridge.publicKeyMarshal(pkHID);
  assert(bytes instanceof Uint8Array, `publicKeyMarshal returns Uint8Array`);
  assert(bytes.length > 0, `publicKeyMarshal bytes length > 0: ${bytes.length}`);

  const unmarshalResult = bridge.publicKeyUnmarshal(bytes);
  assertNoError(unmarshalResult, "publicKeyUnmarshal");

  const bytes2 = bridge.publicKeyMarshal(unmarshalResult.handle);
  assert(
    arraysEqual(bytes, bytes2),
    `PK marshal roundtrip: bytes match (${bytes.length} bytes)`,
  );

  bridge.deleteHandle(unmarshalResult.handle);
}

async function testRelinKeyMarshalRoundtrip(bridge, rlkHID) {
  console.log("\n--- testRelinKeyMarshalRoundtrip ---");

  const bytes = bridge.relinKeyMarshal(rlkHID);
  assert(bytes instanceof Uint8Array, `relinKeyMarshal returns Uint8Array`);
  assert(bytes.length > 0, `relinKeyMarshal bytes length > 0: ${bytes.length}`);

  const unmarshalResult = bridge.relinKeyUnmarshal(bytes);
  assertNoError(unmarshalResult, "relinKeyUnmarshal");

  const bytes2 = bridge.relinKeyMarshal(unmarshalResult.handle);
  assert(
    arraysEqual(bytes, bytes2),
    `RLK marshal roundtrip: bytes match (${bytes.length} bytes)`,
  );

  bridge.deleteHandle(unmarshalResult.handle);
}

async function testGaloisKeyMarshalRoundtrip(bridge, gkHID) {
  console.log("\n--- testGaloisKeyMarshalRoundtrip ---");

  const bytes = bridge.galoisKeyMarshal(gkHID);
  assert(bytes instanceof Uint8Array, `galoisKeyMarshal returns Uint8Array`);
  assert(bytes.length > 0, `galoisKeyMarshal bytes length > 0: ${bytes.length}`);

  const unmarshalResult = bridge.galoisKeyUnmarshal(bytes);
  assertNoError(unmarshalResult, "galoisKeyUnmarshal");

  const bytes2 = bridge.galoisKeyMarshal(unmarshalResult.handle);
  assert(
    arraysEqual(bytes, bytes2),
    `GaloisKey marshal roundtrip: bytes match (${bytes.length} bytes)`,
  );

  bridge.deleteHandle(unmarshalResult.handle);
}

async function testCiphertextMarshalRoundtrip(bridge, paramsHID) {
  console.log("\n--- testCiphertextMarshalRoundtrip ---");

  // Setup: keygen, encode, encrypt
  const kgResult = bridge.newKeyGenerator(paramsHID);
  assertNoError(kgResult, "newKeyGenerator");
  const skResult = bridge.keyGenGenSecretKey(kgResult.handle);
  assertNoError(skResult, "keyGenGenSecretKey");
  const pkResult = bridge.keyGenGenPublicKey(kgResult.handle, skResult.handle);
  assertNoError(pkResult, "keyGenGenPublicKey");

  const encoderResult = bridge.newEncoder(paramsHID);
  assertNoError(encoderResult, "newEncoder");
  const encryptorResult = bridge.newEncryptor(paramsHID, pkResult.handle);
  assertNoError(encryptorResult, "newEncryptor");
  const decryptorResult = bridge.newDecryptor(paramsHID, skResult.handle);
  assertNoError(decryptorResult, "newDecryptor");

  const maxLevel = bridge.ckksMaxLevel(paramsHID);
  const maxSlots = bridge.ckksMaxSlots(paramsHID);
  const defaultScale = bridge.ckksDefaultScale(paramsHID);

  // Encode and encrypt
  const inputValues = [1.5, -2.3, 0.7, 4.1, -0.5, 3.14, 2.71, -1.0];
  const ptResult = bridge.encoderEncode(
    encoderResult.handle,
    inputValues,
    maxLevel,
    defaultScale,
  );
  assertNoError(ptResult, "encoderEncode");

  const ctResult = bridge.encryptorEncryptNew(
    encryptorResult.handle,
    ptResult.handle,
  );
  assertNoError(ctResult, "encryptorEncryptNew");

  // Test ciphertext level accessor
  const ctLevel = bridge.ciphertextLevel(ctResult.handle);
  assert(ctLevel === maxLevel, `ciphertextLevel = ${ctLevel}, expected ${maxLevel}`);

  // Marshal ciphertext
  const ctBytes = bridge.ciphertextMarshal(ctResult.handle);
  assert(ctBytes instanceof Uint8Array, `ciphertextMarshal returns Uint8Array`);
  assert(ctBytes.length > 0, `ciphertext bytes length > 0: ${ctBytes.length}`);

  // Unmarshal ciphertext
  const ctUnmarshalResult = bridge.ciphertextUnmarshal(ctBytes);
  assertNoError(ctUnmarshalResult, "ciphertextUnmarshal");

  // Verify level is preserved
  const ctLevel2 = bridge.ciphertextLevel(ctUnmarshalResult.handle);
  assert(
    ctLevel2 === ctLevel,
    `unmarshaled ciphertext level preserved: ${ctLevel2} === ${ctLevel}`,
  );

  // Re-marshal and compare bytes
  const ctBytes2 = bridge.ciphertextMarshal(ctUnmarshalResult.handle);
  assert(
    arraysEqual(ctBytes, ctBytes2),
    `ciphertext marshal roundtrip: bytes match (${ctBytes.length} bytes)`,
  );

  // Decrypt the unmarshaled ciphertext and verify values
  const decPtResult = bridge.decryptorDecryptNew(
    decryptorResult.handle,
    ctUnmarshalResult.handle,
  );
  assertNoError(decPtResult, "decryptorDecryptNew on unmarshaled ct");

  const decoded = bridge.encoderDecode(
    encoderResult.handle,
    decPtResult.handle,
    maxSlots,
  );
  assert(decoded instanceof Float64Array, `decoded is Float64Array`);

  const tolerance = 1e-3;
  let allMatch = true;
  for (let i = 0; i < inputValues.length; i++) {
    const diff = Math.abs(decoded[i] - inputValues[i]);
    if (diff > tolerance) {
      console.error(
        `  Value mismatch at index ${i}: got ${decoded[i]}, expected ${inputValues[i]}, diff=${diff}`,
      );
      allMatch = false;
    }
  }
  assert(
    allMatch,
    `decrypt after unmarshal: values match within tolerance ${tolerance}`,
  );

  // Clean up
  bridge.deleteHandle(kgResult.handle);
  bridge.deleteHandle(skResult.handle);
  bridge.deleteHandle(pkResult.handle);
  bridge.deleteHandle(encoderResult.handle);
  bridge.deleteHandle(encryptorResult.handle);
  bridge.deleteHandle(decryptorResult.handle);
  bridge.deleteHandle(ptResult.handle);
  bridge.deleteHandle(ctResult.handle);
  bridge.deleteHandle(ctUnmarshalResult.handle);
  bridge.deleteHandle(decPtResult.handle);
}

async function testPlaintextMarshalRoundtrip(bridge, paramsHID) {
  console.log("\n--- testPlaintextMarshalRoundtrip ---");

  const encoderResult = bridge.newEncoder(paramsHID);
  assertNoError(encoderResult, "newEncoder");

  const maxLevel = bridge.ckksMaxLevel(paramsHID);
  const maxSlots = bridge.ckksMaxSlots(paramsHID);
  const defaultScale = bridge.ckksDefaultScale(paramsHID);

  const inputValues = [0.1, 0.2, 0.3, 0.4, 0.5];
  const ptResult = bridge.encoderEncode(
    encoderResult.handle,
    inputValues,
    maxLevel,
    defaultScale,
  );
  assertNoError(ptResult, "encoderEncode");

  // Test plaintext level accessor
  const ptLevel = bridge.plaintextLevel(ptResult.handle);
  assert(ptLevel === maxLevel, `plaintextLevel = ${ptLevel}, expected ${maxLevel}`);

  // Marshal
  const ptBytes = bridge.plaintextMarshal(ptResult.handle);
  assert(ptBytes instanceof Uint8Array, `plaintextMarshal returns Uint8Array`);
  assert(ptBytes.length > 0, `plaintext bytes length > 0: ${ptBytes.length}`);

  // Unmarshal
  const ptUnmarshalResult = bridge.plaintextUnmarshal(ptBytes);
  assertNoError(ptUnmarshalResult, "plaintextUnmarshal");

  // Re-marshal and compare
  const ptBytes2 = bridge.plaintextMarshal(ptUnmarshalResult.handle);
  assert(
    arraysEqual(ptBytes, ptBytes2),
    `plaintext marshal roundtrip: bytes match (${ptBytes.length} bytes)`,
  );

  // Verify level preserved
  const ptLevel2 = bridge.plaintextLevel(ptUnmarshalResult.handle);
  assert(
    ptLevel2 === ptLevel,
    `unmarshaled plaintext level preserved: ${ptLevel2} === ${ptLevel}`,
  );

  // Decode the unmarshaled plaintext and verify values
  const decoded = bridge.encoderDecode(
    encoderResult.handle,
    ptUnmarshalResult.handle,
    maxSlots,
  );
  assert(decoded instanceof Float64Array, `decoded unmarshaled pt is Float64Array`);

  const tolerance = 1e-4;
  let allMatch = true;
  for (let i = 0; i < inputValues.length; i++) {
    const diff = Math.abs(decoded[i] - inputValues[i]);
    if (diff > tolerance) {
      console.error(
        `  Value mismatch at index ${i}: got ${decoded[i]}, expected ${inputValues[i]}, diff=${diff}`,
      );
      allMatch = false;
    }
  }
  assert(
    allMatch,
    `decode after unmarshal: values match within tolerance ${tolerance}`,
  );

  bridge.deleteHandle(encoderResult.handle);
  bridge.deleteHandle(ptResult.handle);
  bridge.deleteHandle(ptUnmarshalResult.handle);
}

async function testMemEvalKeySet(bridge, paramsHID) {
  console.log("\n--- testMemEvalKeySet ---");

  const kgResult = bridge.newKeyGenerator(paramsHID);
  assertNoError(kgResult, "newKeyGenerator");
  const skResult = bridge.keyGenGenSecretKey(kgResult.handle);
  assertNoError(skResult, "keyGenGenSecretKey");

  // Generate RLK
  const rlkResult = bridge.keyGenGenRelinKey(kgResult.handle, skResult.handle);
  assertNoError(rlkResult, "keyGenGenRelinKey");

  // Generate a couple of Galois keys
  const galEl1 = bridge.ckksGaloisElement(paramsHID, 1);
  const galEl2 = bridge.ckksGaloisElement(paramsHID, 2);
  const gk1Result = bridge.keyGenGenGaloisKey(
    kgResult.handle,
    skResult.handle,
    galEl1,
  );
  assertNoError(gk1Result, "keyGenGenGaloisKey(1)");
  const gk2Result = bridge.keyGenGenGaloisKey(
    kgResult.handle,
    skResult.handle,
    galEl2,
  );
  assertNoError(gk2Result, "keyGenGenGaloisKey(2)");

  // Create MemEvaluationKeySet
  const evkResult = bridge.newMemEvalKeySet(rlkResult.handle, [
    gk1Result.handle,
    gk2Result.handle,
  ]);
  assertNoError(evkResult, "newMemEvalKeySet");
  assert(evkResult.handle > 0, `evk handle is positive: ${evkResult.handle}`);

  // Marshal
  const evkBytes = bridge.memEvalKeySetMarshal(evkResult.handle);
  assert(evkBytes instanceof Uint8Array, `memEvalKeySetMarshal returns Uint8Array`);
  assert(evkBytes.length > 0, `evk bytes length > 0: ${evkBytes.length}`);
  console.log(`  MemEvalKeySet size: ${evkBytes.length} bytes`);

  // Unmarshal
  const evkUnmarshalResult = bridge.memEvalKeySetUnmarshal(evkBytes);
  assertNoError(evkUnmarshalResult, "memEvalKeySetUnmarshal");

  // Re-marshal and compare
  const evkBytes2 = bridge.memEvalKeySetMarshal(evkUnmarshalResult.handle);
  assert(
    arraysEqual(evkBytes, evkBytes2),
    `MemEvalKeySet marshal roundtrip: bytes match (${evkBytes.length} bytes)`,
  );

  // Test with null RLK (no relinearization key)
  const evkNoRlk = bridge.newMemEvalKeySet(null, [gk1Result.handle]);
  assertNoError(evkNoRlk, "newMemEvalKeySet with null RLK");

  // Test with empty galois keys
  const evkNoGk = bridge.newMemEvalKeySet(rlkResult.handle, []);
  assertNoError(evkNoGk, "newMemEvalKeySet with empty galois keys");

  // Clean up
  bridge.deleteHandle(kgResult.handle);
  bridge.deleteHandle(skResult.handle);
  bridge.deleteHandle(rlkResult.handle);
  bridge.deleteHandle(gk1Result.handle);
  bridge.deleteHandle(gk2Result.handle);
  bridge.deleteHandle(evkResult.handle);
  bridge.deleteHandle(evkUnmarshalResult.handle);
  bridge.deleteHandle(evkNoRlk.handle);
  bridge.deleteHandle(evkNoGk.handle);
}

async function testSerializationErrors(bridge) {
  console.log("\n--- testSerializationErrors ---");

  // Invalid handles for marshal
  const r1 = bridge.secretKeyMarshal(99999);
  assert(!!r1.error, `secretKeyMarshal invalid handle: ${r1.error}`);

  const r2 = bridge.publicKeyMarshal(99999);
  assert(!!r2.error, `publicKeyMarshal invalid handle: ${r2.error}`);

  const r3 = bridge.relinKeyMarshal(99999);
  assert(!!r3.error, `relinKeyMarshal invalid handle: ${r3.error}`);

  const r4 = bridge.galoisKeyMarshal(99999);
  assert(!!r4.error, `galoisKeyMarshal invalid handle: ${r4.error}`);

  const r5 = bridge.ciphertextMarshal(99999);
  assert(!!r5.error, `ciphertextMarshal invalid handle: ${r5.error}`);

  const r6 = bridge.plaintextMarshal(99999);
  assert(!!r6.error, `plaintextMarshal invalid handle: ${r6.error}`);

  const r7 = bridge.memEvalKeySetMarshal(99999);
  assert(!!r7.error, `memEvalKeySetMarshal invalid handle: ${r7.error}`);

  // Invalid bytes for unmarshal
  const badBytes = new Uint8Array([1, 2, 3, 4]);

  const r8 = bridge.secretKeyUnmarshal(badBytes);
  assert(!!r8.error, `secretKeyUnmarshal bad bytes: ${r8.error}`);

  const r9 = bridge.publicKeyUnmarshal(badBytes);
  assert(!!r9.error, `publicKeyUnmarshal bad bytes: ${r9.error}`);

  const r10 = bridge.relinKeyUnmarshal(badBytes);
  assert(!!r10.error, `relinKeyUnmarshal bad bytes: ${r10.error}`);

  const r11 = bridge.galoisKeyUnmarshal(badBytes);
  assert(!!r11.error, `galoisKeyUnmarshal bad bytes: ${r11.error}`);

  const r12 = bridge.ciphertextUnmarshal(badBytes);
  assert(!!r12.error, `ciphertextUnmarshal bad bytes: ${r12.error}`);

  const r13 = bridge.plaintextUnmarshal(badBytes);
  assert(!!r13.error, `plaintextUnmarshal bad bytes: ${r13.error}`);

  const r14 = bridge.memEvalKeySetUnmarshal(badBytes);
  assert(!!r14.error, `memEvalKeySetUnmarshal bad bytes: ${r14.error}`);

  // Level accessors with invalid handles
  const r15 = bridge.ciphertextLevel(99999);
  assert(!!r15.error, `ciphertextLevel invalid handle: ${r15.error}`);

  const r16 = bridge.plaintextLevel(99999);
  assert(!!r16.error, `plaintextLevel invalid handle: ${r16.error}`);
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
  const pkResult = bridge.keyGenGenPublicKey(kgResult.handle, skResult.handle);
  assertNoError(pkResult, "keyGenGenPublicKey");
  const rlkResult = bridge.keyGenGenRelinKey(kgResult.handle, skResult.handle);
  assertNoError(rlkResult, "keyGenGenRelinKey");

  const galEl = bridge.ckksGaloisElement(paramsHID, 1);
  const gkResult = bridge.keyGenGenGaloisKey(
    kgResult.handle,
    skResult.handle,
    galEl,
  );
  assertNoError(gkResult, "keyGenGenGaloisKey");

  // Test key marshal roundtrips
  await testSecretKeyMarshalRoundtrip(bridge, skResult.handle);
  await testPublicKeyMarshalRoundtrip(bridge, pkResult.handle);
  await testRelinKeyMarshalRoundtrip(bridge, rlkResult.handle);
  await testGaloisKeyMarshalRoundtrip(bridge, gkResult.handle);

  // Test plaintext marshal roundtrip
  await testPlaintextMarshalRoundtrip(bridge, paramsHID);

  // Test ciphertext marshal roundtrip + decrypt after unmarshal
  await testCiphertextMarshalRoundtrip(bridge, paramsHID);

  // Test MemEvaluationKeySet
  await testMemEvalKeySet(bridge, paramsHID);

  // Test error handling
  await testSerializationErrors(bridge);

  // Clean up setup handles
  bridge.deleteHandle(paramsHID);
  bridge.deleteHandle(kgResult.handle);
  bridge.deleteHandle(skResult.handle);
  bridge.deleteHandle(pkResult.handle);
  bridge.deleteHandle(rlkResult.handle);
  bridge.deleteHandle(gkResult.handle);

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
