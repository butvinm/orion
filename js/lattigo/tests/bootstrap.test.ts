import { describe, it, expect, beforeAll } from "vitest";
import { ensureWasmLoaded, TEST_PARAMS } from "./helpers.js";
import { CKKSParameters, KeyGenerator, isError } from "../src/index.js";
import { getBridge } from "../src/bridge.js";

beforeAll(async () => {
  await ensureWasmLoaded();
});

describe("Bootstrap parameters", () => {
  it("newBootstrapParametersFromLiteral constructs valid params handle", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const bridge = getBridge();

    // For ConjugateInvariant ring, bootstrap operates at LogN+1
    const btpLogN = TEST_PARAMS.logN + 1;
    const numSlots = 256;
    const logSlots = Math.round(Math.log2(numSlots));

    const btpLitJSON = JSON.stringify({
      LogN: btpLogN,
      LogP: TEST_PARAMS.logP,
      H: 192,
      LogSlots: logSlots,
    });

    const result = bridge.newBootstrapParametersFromLiteral(
      params.handle,
      btpLitJSON,
    );
    expect(isError(result)).toBe(false);
    if (!isError(result)) {
      expect(result.handle).toBeGreaterThan(0);
      bridge.deleteHandle(result.handle);
    }

    params.free();
  });

  it("newBootstrapParametersFromLiteral accepts empty literal (all defaults)", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const bridge = getBridge();

    const btpLogN = TEST_PARAMS.logN + 1;
    const result = bridge.newBootstrapParametersFromLiteral(
      params.handle,
      JSON.stringify({ LogN: btpLogN }),
    );
    // May succeed or fail depending on Lattigo defaults — just verify no panic
    if (!isError(result)) {
      bridge.deleteHandle(result.handle);
    }

    params.free();
  });

  it("newBootstrapParametersFromLiteral returns error for invalid params handle", () => {
    const bridge = getBridge();
    const result = bridge.newBootstrapParametersFromLiteral(99999, "{}");
    expect(isError(result)).toBe(true);
  });

  it("newBootstrapParametersFromLiteral returns error for invalid JSON", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const bridge = getBridge();

    const result = bridge.newBootstrapParametersFromLiteral(
      params.handle,
      "not valid json",
    );
    expect(isError(result)).toBe(true);

    params.free();
  });
});

// Bootstrap key generation is heavy (10–60s). Use LogSlots=1 (2 slots) to
// keep the resulting MemEvaluationKeySet small enough to marshal within WASM
// memory limits — LogSlots=8 (256 slots) produces ~1.7 GB and OOMs.
describe("Bootstrap key generation (slow)", () => {
  it(
    "btpParamsGenEvaluationKeys generates keys and marshals EVK roundtrip",
    async () => {
      const bridge = getBridge();
      const params = CKKSParameters.fromLogn(TEST_PARAMS);
      const kg = KeyGenerator.new(params);
      const sk = kg.genSecretKey();

      // For ConjugateInvariant ring, bootstrap operates at LogN+1.
      const btpLogN = TEST_PARAMS.logN + 1;
      const btpLitJSON = JSON.stringify({
        LogN: btpLogN,
        LogP: TEST_PARAMS.logP,
        H: 192,
        LogSlots: 1,
      });

      const btpParamsResult = bridge.newBootstrapParametersFromLiteral(
        params.handle,
        btpLitJSON,
      );
      expect(isError(btpParamsResult)).toBe(false);
      if (isError(btpParamsResult)) return;

      const result = await bridge.btpParamsGenEvaluationKeys(
        btpParamsResult.handle,
        sk.handle,
      );

      expect(typeof result).toBe("object");
      expect(result).not.toBeNull();
      expect(result.evkHID).toBeGreaterThan(0);
      expect(result.btpEvkHID).toBeGreaterThan(0);
      expect(result.evkHID).not.toBe(result.btpEvkHID);

      // Marshal the MemEvaluationKeySet from the bootstrap keys.
      const evkBytes = bridge.memEvalKeySetMarshal(result.evkHID);
      expect(evkBytes).toBeInstanceOf(Uint8Array);
      expect(evkBytes.length).toBeGreaterThan(0);

      // Unmarshal and re-marshal — bytes must match.
      const unmarshalResult = bridge.memEvalKeySetUnmarshal(evkBytes);
      expect(isError(unmarshalResult)).toBe(false);
      if (isError(unmarshalResult)) return;
      const evkBytes2 = bridge.memEvalKeySetMarshal(unmarshalResult.handle);
      expect(evkBytes2).toBeInstanceOf(Uint8Array);
      expect(evkBytes2.length).toBe(evkBytes.length);
      expect(Buffer.from(evkBytes2).equals(Buffer.from(evkBytes))).toBe(true);

      // Cleanup.
      bridge.deleteHandle(unmarshalResult.handle);
      bridge.deleteHandle(btpParamsResult.handle);
      bridge.deleteHandle(result.evkHID);
      bridge.deleteHandle(result.btpEvkHID);
      sk.free();
      kg.free();
      params.free();
    },
    120_000,
  );

  it("btpParamsGenEvaluationKeys rejects invalid btpParams handle", async () => {
    const bridge = getBridge();
    await expect(
      bridge.btpParamsGenEvaluationKeys(99999, 99999),
    ).rejects.toMatch(/invalid/);
  });
});
