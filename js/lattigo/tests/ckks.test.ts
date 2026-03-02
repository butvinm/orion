import { describe, it, expect, beforeAll } from "vitest";
import { ensureWasmLoaded, TEST_PARAMS } from "./helpers.js";
import { CKKSParameters } from "../src/index.js";

beforeAll(async () => {
  await ensureWasmLoaded();
});

describe("CKKSParameters", () => {
  it("fromLogn creates valid parameters", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    expect(params.handle).toBeGreaterThan(0);
    params.free();
  });

  it("maxSlots returns positive power of 2", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const slots = params.maxSlots();
    expect(slots).toBeGreaterThan(0);
    expect(Math.log2(slots) % 1).toBe(0);
    params.free();
  });

  it("maxLevel returns len(logQ) - 1", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    expect(params.maxLevel()).toBe(TEST_PARAMS.logQ.length - 1);
    params.free();
  });

  it("defaultScale returns 2^logDefaultScale", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    expect(params.defaultScale()).toBe(
      Math.pow(2, TEST_PARAMS.logDefaultScale),
    );
    params.free();
  });

  it("galoisElement returns valid element for rotation", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const ge = params.galoisElement(1);
    expect(ge).toBeGreaterThan(0);
    params.free();
  });

  it("moduliChain returns array with len(logQ) elements", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const chain = params.moduliChain();
    expect(chain).toHaveLength(TEST_PARAMS.logQ.length);
    chain.forEach((q) => expect(q).toBeGreaterThan(0));
    params.free();
  });

  it("auxModuliChain returns array with len(logP) elements", () => {
    const params = CKKSParameters.fromLogn(TEST_PARAMS);
    const chain = params.auxModuliChain();
    expect(chain).toHaveLength(TEST_PARAMS.logP.length);
    chain.forEach((p) => expect(p).toBeGreaterThan(0));
    params.free();
  });

  it("fromJSON produces equivalent parameters", () => {
    const json = JSON.stringify({
      LogN: TEST_PARAMS.logN,
      LogQ: TEST_PARAMS.logQ,
      LogP: TEST_PARAMS.logP,
      LogDefaultScale: TEST_PARAMS.logDefaultScale,
      H: 192,
      RingType: TEST_PARAMS.ringType,
    });
    const params = CKKSParameters.fromJSON(json);
    expect(params.maxLevel()).toBe(TEST_PARAMS.logQ.length - 1);
    expect(params.defaultScale()).toBe(
      Math.pow(2, TEST_PARAMS.logDefaultScale),
    );
    params.free();
  });
});
