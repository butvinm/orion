/**
 * OrionClient — JavaScript wrapper around the Go WASM Lattigo module.
 *
 * Handles WASM lifecycle, key generation, encryption/decryption, and
 * CipherText wire format serialization compatible with Python's CipherText.to_bytes().
 */
class OrionClient {
  constructor() {
    this._go = null;
    this._ready = false;
    this._onProgress = null;
  }

  /**
   * Set a progress callback: (stage: string, current: number, total: number) => void
   */
  set onProgress(fn) {
    this._onProgress = fn;
  }

  _progress(stage, current, total) {
    if (this._onProgress) {
      this._onProgress(stage, current, total);
    }
  }

  /**
   * Load the WASM binary and start the Go runtime.
   * @param {string} wasmUrl - URL to the orion.wasm file
   */
  async init(wasmUrl) {
    if (typeof Go === "undefined") {
      throw new Error(
        "Go WASM runtime not loaded. Include wasm_exec.js before orion-client.js",
      );
    }

    this._go = new Go();

    this._progress("wasm_load", 0, 1);
    const result = await WebAssembly.instantiateStreaming(
      fetch(wasmUrl),
      this._go.importObject,
    );

    // Start the Go runtime (non-blocking — it blocks on select{} internally)
    this._go.run(result.instance);
    this._progress("wasm_load", 1, 1);
  }

  /**
   * Initialize the CKKS scheme in the WASM module.
   * @param {Object} params - { logn, logq, logp, logscale, h, ring_type }
   */
  async setupScheme(params) {
    this._progress("scheme_init", 0, 1);
    try {
      await globalThis.orionInit(
        params.logn,
        params.logq,
        params.logp,
        params.logscale,
        params.h,
        params.ring_type,
      );
    } catch (err) {
      throw new Error(`WASM scheme init failed: ${err}`);
    }
    this._ready = true;
    this._progress("scheme_init", 1, 1);
  }

  /**
   * Returns the maximum number of plaintext slots.
   * @returns {number}
   */
  getMaxSlots() {
    this._checkReady();
    return globalThis.orionGetMaxSlots();
  }

  /**
   * Generate and serialize the relinearization key.
   * @returns {Promise<Uint8Array>}
   */
  async generateAndSerializeRlk() {
    this._checkReady();
    try {
      return await globalThis.orionSerializeRelinKey();
    } catch (err) {
      throw new Error(`RLK serialization failed: ${err}`);
    }
  }

  /**
   * Generate and serialize a single Galois key.
   * @param {number} galEl - Galois element
   * @returns {Promise<Uint8Array>}
   */
  async generateAndSerializeGaloisKey(galEl) {
    this._checkReady();
    try {
      return await globalThis.orionGenerateAndSerializeGaloisKey(galEl);
    } catch (err) {
      throw new Error(`Galois key generation failed for element ${galEl}: ${err}`);
    }
  }

  /**
   * Generate and serialize bootstrap evaluation keys.
   * @param {number} numSlots - Number of bootstrap slots
   * @param {number[]} logP - Auxiliary modulus chain
   * @returns {Promise<Uint8Array>}
   */
  async generateAndSerializeBootstrapKeys(numSlots, logP) {
    this._checkReady();
    try {
      return await globalThis.orionSerializeBootstrapKeys(numSlots, logP);
    } catch (err) {
      throw new Error(`Bootstrap key generation failed: ${err}`);
    }
  }

  /**
   * Encode, encrypt, and wrap input data in the CipherText wire format.
   *
   * Wire format (matches Python CipherText.to_bytes):
   *   [4 bytes] NUM_CIPHERTEXTS (uint32 LE)
   *   [4 bytes] SHAPE_LEN (uint32 LE)
   *   [N x 4 bytes] SHAPE_DIMS (int32 LE each)
   *   for each ciphertext:
   *     [8 bytes] CT_LEN (uint64 LE)
   *     [N bytes] CT_DATA (raw Lattigo MarshalBinary)
   *
   * @param {Float64Array} values - Input values (will be zero-padded to slot boundary)
   * @param {number[]} shape - Tensor shape (e.g. [1, 1, 28, 28])
   * @param {number} level - CKKS level for encoding
   * @returns {Promise<Uint8Array>} - Wire-format serialized ciphertext
   */
  async encryptInput(values, shape, level) {
    this._checkReady();

    const maxSlots = this.getMaxSlots();
    const scale = globalThis.orionGetDefaultScale();
    const numElements = values.length;

    // Pad to slot boundary
    const padLength = (maxSlots - (numElements % maxSlots)) % maxSlots;
    const totalLength = numElements + padLength;
    const numCts = totalLength / maxSlots;

    this._progress("encrypt", 0, numCts);

    const ctChunks = [];
    for (let i = 0; i < numCts; i++) {
      const chunk = new Float64Array(maxSlots);
      const srcStart = i * maxSlots;
      const srcEnd = Math.min(srcStart + maxSlots, numElements);
      for (let j = 0; j < srcEnd - srcStart; j++) {
        chunk[j] = values[srcStart + j];
      }
      // remaining slots stay 0

      const ptxtId = await globalThis.orionEncode(chunk, level, scale);
      const ctBytes = await globalThis.orionEncrypt(ptxtId);
      ctChunks.push(ctBytes);
      this._progress("encrypt", i + 1, numCts);
    }

    // Build wire format
    return this._buildWireFormat(ctChunks, shape);
  }

  /**
   * Strip wire format header, decrypt each ciphertext, and return trimmed results.
   * @param {Uint8Array} wireBytes - Wire-format serialized ciphertext
   * @param {number} numElements - Number of meaningful output elements to return
   * @returns {Promise<Float64Array>} - Decrypted values trimmed to numElements
   */
  async decryptOutput(wireBytes, numElements) {
    this._checkReady();

    const { ctDataList } = this._parseWireFormat(wireBytes);

    this._progress("decrypt", 0, ctDataList.length);

    const allValues = [];
    for (let i = 0; i < ctDataList.length; i++) {
      const result = await globalThis.orionDecrypt(ctDataList[i]);
      for (let j = 0; j < result.length; j++) {
        allValues.push(result[j]);
      }
      this._progress("decrypt", i + 1, ctDataList.length);
    }

    // Trim to numElements
    const output = new Float64Array(numElements);
    for (let i = 0; i < numElements; i++) {
      output[i] = allValues[i];
    }
    return output;
  }

  /**
   * Build the CipherText wire format from raw ciphertext byte arrays.
   * @param {Uint8Array[]} ctChunks - Array of raw Lattigo ciphertext bytes
   * @param {number[]} shape - Tensor shape
   * @returns {Uint8Array}
   */
  _buildWireFormat(ctChunks, shape) {
    // Calculate total size
    const headerSize = 4 + 4 + shape.length * 4; // num_cts + shape_len + shape_dims
    let dataSize = 0;
    for (const chunk of ctChunks) {
      dataSize += 8 + chunk.length; // ct_len + ct_data
    }

    const buf = new ArrayBuffer(headerSize + dataSize);
    const view = new DataView(buf);
    let offset = 0;

    // NUM_CIPHERTEXTS (uint32 LE)
    view.setUint32(offset, ctChunks.length, true);
    offset += 4;

    // SHAPE_LEN (uint32 LE)
    view.setUint32(offset, shape.length, true);
    offset += 4;

    // SHAPE_DIMS (int32 LE each)
    for (const dim of shape) {
      view.setInt32(offset, dim, true);
      offset += 4;
    }

    // For each ciphertext: CT_LEN (uint64 LE) + CT_DATA
    const uint8View = new Uint8Array(buf);
    for (const chunk of ctChunks) {
      // uint64 LE — use two uint32 writes (JS DataView has no setUint64)
      view.setUint32(offset, chunk.length, true); // low 32 bits
      view.setUint32(offset + 4, 0, true); // high 32 bits (ciphertext < 4GB)
      offset += 8;

      uint8View.set(chunk, offset);
      offset += chunk.length;
    }

    return new Uint8Array(buf);
  }

  /**
   * Parse the CipherText wire format into components.
   * @param {Uint8Array} wireBytes
   * @returns {{ numCts: number, shape: number[], ctDataList: Uint8Array[] }}
   */
  _parseWireFormat(wireBytes) {
    const view = new DataView(
      wireBytes.buffer,
      wireBytes.byteOffset,
      wireBytes.byteLength,
    );
    let offset = 0;

    const numCts = view.getUint32(offset, true);
    offset += 4;

    const shapeLen = view.getUint32(offset, true);
    offset += 4;

    const shape = [];
    for (let i = 0; i < shapeLen; i++) {
      shape.push(view.getInt32(offset, true));
      offset += 4;
    }

    const ctDataList = [];
    for (let i = 0; i < numCts; i++) {
      // Read uint64 LE as low 32 + high 32
      const ctLenLow = view.getUint32(offset, true);
      const ctLenHigh = view.getUint32(offset + 4, true);
      const ctLen = ctLenLow + ctLenHigh * 0x100000000;
      offset += 8;

      const ctData = wireBytes.slice(offset, offset + ctLen);
      ctDataList.push(ctData);
      offset += ctLen;
    }

    return { numCts, shape, ctDataList };
  }

  _checkReady() {
    if (!this._ready) {
      throw new Error(
        "OrionClient not initialized. Call init() and setupScheme() first.",
      );
    }
  }
}
