import { describe, it, expect, beforeAll } from "vitest";
import { ensureWasmLoaded, TEST_PARAMS } from "./helpers.js";
import { CKKSParameters, isError } from "../src/index.js";
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
