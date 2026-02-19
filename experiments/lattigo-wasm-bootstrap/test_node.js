require("./wasm_exec.js");

const fs = require("fs");

async function main() {
  const go = new Go();
  const wasmBuffer = fs.readFileSync("./lattigo-bootstrap.wasm");
  const result = await WebAssembly.instantiate(wasmBuffer, go.importObject);

  go.run(result.instance);
  await new Promise((r) => setTimeout(r, 100));

  console.log("Starting bootstrap test (this may take a while)...\n");
  const startTime = Date.now();

  try {
    const output = await globalThis.lattigoBootstrapTest();
    const elapsed = Date.now() - startTime;
    console.log(`\nResult: ${output}`);
    console.log(`Wall time: ${elapsed}ms`);
  } catch (err) {
    console.error("Error:", err);
  }

  process.exit(0);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
