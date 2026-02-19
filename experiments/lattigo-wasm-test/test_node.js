require("./wasm_exec.js");

const fs = require("fs");

async function main() {
  const go = new Go();
  const wasmBuffer = fs.readFileSync("./lattigo-test.wasm");
  const result = await WebAssembly.instantiate(wasmBuffer, go.importObject);

  // Start Go runtime (don't await — it blocks until main exits)
  go.run(result.instance);

  // Give Go a moment to register globals
  await new Promise((r) => setTimeout(r, 100));

  // Call the registered function
  console.log("Calling lattigoTest()...");
  const startTime = Date.now();
  const output = globalThis.lattigoTest();
  const elapsed = Date.now() - startTime;
  console.log(`Result: ${output}`);
  console.log(`Time: ${elapsed}ms`);

  process.exit(0);
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
