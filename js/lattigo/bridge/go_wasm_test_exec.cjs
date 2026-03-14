// Minimal Go WASM test runner for js/wasm test binaries.
//
// Called by go_wasm_test_exec.sh with:
//   node --stack-size=8192 <this-file> <goroot> <wasm-binary> [test-flags...]
//
// Why not use $GOROOT/lib/wasm/go_js_wasm_exec directly:
// That script forwards all of process.env to the Go WASM runtime. With a
// typical dev shell (~100+ env vars), the total serialized length exceeds
// wasm_exec.js's 12 KB hard limit, triggering "total length of command line
// and environment variables exceeds limit". This runner passes only TMPDIR.
"use strict";

const goroot = process.argv[2];
const wasmArgs = process.argv.slice(3); // [binary-path, test-flags...]

const fs = require("fs");
const os = require("os");

globalThis.require = require;
globalThis.fs = fs;
globalThis.path = require("path");
globalThis.TextEncoder = require("util").TextEncoder;
globalThis.TextDecoder = require("util").TextDecoder;
// performance and crypto are already built-in globals in Node.js 22;
// avoid assignment to their non-writable getters.

require(`${goroot}/lib/wasm/wasm_exec`);

const go = new Go();
go.argv = wasmArgs;
// Minimal env — only TMPDIR is needed for Go test infrastructure.
go.env = { TMPDIR: os.tmpdir() };
go.exit = process.exit;

WebAssembly.instantiate(fs.readFileSync(wasmArgs[0]), go.importObject)
  .then((result) => {
    process.on("exit", (code) => {
      if (code === 0 && !go.exited) {
        // Node.js exits if no event handler is pending; prompt Go to print
        // deadlock info.
        go._pendingEvent = { id: 0 };
        go._resume();
      }
    });
    return go.run(result.instance);
  })
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
