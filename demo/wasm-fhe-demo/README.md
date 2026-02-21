# WASM FHE Demo: Browser-Based Encrypted Inference

A complete demo of Fully Homomorphic Encryption inference from a web browser. A neural network classifies handwritten digits (MNIST) — the server never sees the input or output in plaintext.

- **FastAPI server** loads a pre-compiled MLP model and performs encrypted inference
- **Browser client** uses Lattigo compiled to WASM for client-side key generation, encryption, and decryption
- **The server is untrusted**: the secret key never leaves the browser

## Prerequisites

- Go 1.22+
- Python 3.9–3.12
- Orion installed in editable mode (`pip install -e .` from repo root)
- System dependencies: `libgmp-dev`, `libssl-dev`, C compiler (for Orion's Go backend)

## Quick Start

### 1. Build the WASM binary

```bash
cd demo/wasm-fhe-demo
bash tools/build_wasm.sh
```

This compiles the Go crypto module to `client/orion.wasm` and copies the Go WASM runtime (`wasm_exec.js`) into `client/`.

### 2. Compile the model

```bash
python tools/compile_model.py
```

This compiles the MLP (784 -> 128 -> 128 -> 10) with CKKS parameters and saves `model.bin`. Only needs to run once.

### 3. Start the server

```bash
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 4. Open the browser

Navigate to `http://localhost:8000`. The UI walks you through four steps:

1. **Initialize** — Fetches model manifest from server, loads WASM, sets up CKKS scheme
2. **Generate & Upload Keys** — Generates evaluation keys (RLK, Galois, bootstrap) in the browser and streams them to the server
3. **Encrypt & Infer** — Draw a digit on the canvas, encrypt it locally, send the ciphertext to the server for inference
4. **Result** — Decrypt the server's response and display classification scores

## Architecture

```
Browser (WASM)                          FastAPI Server
==============                          ==============

  Secret Key (never sent)
  |
  +-- Generate Eval Keys
  |     |
  |     +-- POST /api/keys/rlk -------> Store RLK
  |     +-- POST /api/keys/galois/N --> Store Galois key N
  |     +-- POST /api/keys/bootstrap -> Store bootstrap keys
  |     +-- POST /api/keys/finalize --> Build Evaluator
  |
  +-- Encrypt input
  |     |
  |     +-- POST /api/infer ----------> Run FHE inference
  |     |                               (on ciphertexts only)
  |     +-- <-- encrypted result ------
  |
  +-- Decrypt result (locally)
        |
        +-- Display classification
```

The server loads a pre-compiled model (`model.bin`) and evaluation keys uploaded by the client. It performs inference entirely on encrypted data using Orion's `Evaluator`. The secret key exists only in WASM memory.

## Security Model

| Data                               | Location                       | Confidentiality                   |
| ---------------------------------- | ------------------------------ | --------------------------------- |
| Secret Key                         | Browser WASM memory only       | Never transmitted                 |
| Eval Keys (RLK, Galois, Bootstrap) | Sent to server                 | Public — cannot derive secret key |
| Input Plaintext                    | Browser only                   | Encrypted before transmission     |
| Input/Output Ciphertext            | Transmitted                    | Encrypted — server cannot decrypt |
| Model Weights                      | Server only (in CompiledModel) | Not revealed to client            |

## Project Structure

```
demo/wasm-fhe-demo/
├── wasm/                        # Go source for WASM crypto module
│   ├── bindings_js.go           # syscall/js wrappers (wraps orionclient.Client)
│   ├── stub.go                  # Non-WASM stub for go test
│   ├── crypto_test.go           # Go tests (run with: go test ./...)
│   ├── go.mod / go.sum
│   └── Makefile
├── server/
│   ├── app.py                   # FastAPI endpoints
│   ├── test_app.py              # Server tests (run with: pytest)
│   └── requirements.txt
├── client/
│   ├── index.html               # Single-page demo UI
│   ├── orion-client.js          # JS wrapper around WASM
│   ├── wasm_exec.js             # Go WASM runtime (copied from toolchain)
│   └── orion.wasm               # Compiled WASM binary (generated)
├── tools/
│   ├── compile_model.py         # Compile MLP -> model.bin
│   └── build_wasm.sh            # Build WASM + copy runtime
└── model.bin                    # Pre-compiled MLP model (generated)
```

## Running Tests

Go crypto tests (no WASM target needed):

```bash
cd wasm
go test -v ./...
```

Server tests:

```bash
cd server
pytest test_app.py -v
```

## Known Limitations

- **WASM binary size**: The Lattigo WASM binary is ~15–30 MB uncompressed. The server serves it with gzip compression (~5–10 MB transfer).
- **Single-tenant**: The demo server uses a single session dictionary, so only one client session can be active at a time. The underlying Go backend (`orionclient`) supports multiple concurrent instances, but this demo is deliberately single-tenant for simplicity. Use the "Reset" button between sessions.
- **Key generation time**: Generating all evaluation keys in WASM takes 30–120 seconds depending on the browser and hardware.
- **Memory usage**: Lattigo is memory-intensive. The browser tab may use 500 MB–1 GB during key generation and inference.
- **Browser compatibility**: Tested with Chrome and Firefox. Safari support for Go WASM may vary.
