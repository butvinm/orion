import { describe, it, expect, beforeAll } from "vitest";
import { ensureWasmLoaded, TEST_PARAMS } from "./helpers.js";
import { CKKSParameters } from "../src/index.js";

beforeAll(async () => {
  await ensureWasmLoaded();
});

describe("CKKSParameters", () => {
  it("constructor creates valid parameters", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    expect(params.handle).toBeGreaterThan(0);
    params.close();
  });

  it("maxSlots returns positive power of 2", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    const slots = params.maxSlots();
    expect(slots).toBeGreaterThan(0);
    expect(Math.log2(slots) % 1).toBe(0);
    params.close();
  });

  it("maxLevel returns len(logQ) - 1", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    expect(params.maxLevel()).toBe(TEST_PARAMS.logQ.length - 1);
    params.close();
  });

  it("defaultScale returns 2^logDefaultScale", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    expect(params.defaultScale()).toBe(
      Math.pow(2, TEST_PARAMS.logDefaultScale),
    );
    params.close();
  });

  it("galoisElement returns valid element for rotation", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    const ge = params.galoisElement(1);
    expect(ge).toBeGreaterThan(0);
    params.close();
  });

  it("moduliChain returns array with len(logQ) elements", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    const chain = params.moduliChain();
    expect(chain).toHaveLength(TEST_PARAMS.logQ.length);
    chain.forEach((q) => expect(q).toBeGreaterThan(0));
    params.close();
  });

  it("auxModuliChain returns array with len(logP) elements", () => {
    const params = new CKKSParameters(TEST_PARAMS);
    const chain = params.auxModuliChain();
    expect(chain).toHaveLength(TEST_PARAMS.logP.length);
    chain.forEach((p) => expect(p).toBeGreaterThan(0));
    params.close();
  });

  it("constructor with explicit h produces equivalent parameters", () => {
    const params = new CKKSParameters({
      logN: TEST_PARAMS.logN,
      logQ: TEST_PARAMS.logQ,
      logP: TEST_PARAMS.logP,
      logDefaultScale: TEST_PARAMS.logDefaultScale,
      ringType: TEST_PARAMS.ringType,
      h: 192,
    });
    expect(params.maxLevel()).toBe(TEST_PARAMS.logQ.length - 1);
    expect(params.defaultScale()).toBe(
      Math.pow(2, TEST_PARAMS.logDefaultScale),
    );
    params.close();
  });
});
