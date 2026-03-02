# WASM Browser Demo

End-to-end encrypted inference demo: the browser generates keys and encrypts input, a Go server runs Orion inference on ciphertexts, and the browser decrypts the result. The secret key never leaves the browser.

## Prerequisites

- Go 1.22+
- Python 3.9+ with a venv containing `orion-compiler` and its dependencies
- Node.js 18+

## Steps

### 1. Generate the demo model (once)

```bash
cd examples/wasm-demo
python generate_model.py
```

This compiles a small MLP and writes `model.orion`. Random weights are fine — the demo validates the pipeline, not accuracy.

### 2. Build the WASM binary (once or after Go bridge changes)

```bash
python tools/build_lattigo_wasm.py
```

This writes `js/lattigo/wasm/lattigo.wasm`. The demo client uses a pre-copied version at `client/wasm/lattigo.wasm`. Update it if the WASM binary changes:

```bash
cp js/lattigo/wasm/lattigo.wasm examples/wasm-demo/client/wasm/
cp js/lattigo/wasm/wasm_exec.js  examples/wasm-demo/client/wasm/
```

### 3. Build the browser client

```bash
cd examples/wasm-demo/client
npm install
npm run build
```

This bundles `client.ts` to `client.js` using esbuild.

### 4. Run the server

```bash
cd examples/wasm-demo/server
go run .
```

Optional arguments: `go run . <model_path> <client_dir> <addr>` (defaults: `../model.orion`, `../client`, `:8080`).

### 5. Open the browser

Navigate to http://localhost:8080. Click **Initialize Keys** (takes 1–5s for key generation), enter comma-separated float values, and click **Run Inference**.

## What the demo does

1. Browser fetches CKKS parameters and key manifest from the server
2. Lattigo WASM loads in the browser (~8 MB, ~3 MB gzipped)
3. Browser generates SK, PK, RLK, and Galois keys locally
4. Browser POSTs the evaluation key set to the server to create a session
5. Browser encodes and encrypts the input, POSTs the ciphertext
6. Server runs the Orion evaluator on the ciphertext and returns the result
7. Browser decrypts and decodes the result — the secret key never left the browser

## Server tests

```bash
cd examples/wasm-demo/server
go test ./...
```
