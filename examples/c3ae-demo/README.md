# C3AE Age Verification — Encrypted FHE Demo

Privacy-preserving age verification using CKKS homomorphic encryption. A browser client generates keys, encrypts a face image, and sends the ciphertext to a Go server that runs FHE inference. The secret key and face image never leave the browser.

Based on the C3AE architecture adapted for FHE: ReLU replaced with Quad (x²), BatchNorm fused, binary classification (18+ adult/minor).

## Prerequisites

- Go 1.22+
- Python 3.11+ with a venv containing `orion-compiler`, `orion-evaluator`, and `lattigo`
- Node.js 18+
- UTKFace dataset (for training)
- 128 GB RAM for FHE inference

## Quick Start

### 1. Train the model

```bash
cd examples/c3ae-demo

# Download UTKFace dataset
python -c "import kagglehub; kagglehub.dataset_download('jangedoo/utkface-new')"

python train.py --data-dir ./data/UTKFace --epochs 60 --output weights.pth
```

### 2. Compile to .orion format

```bash
python generate_model.py --weights weights.pth --output model.orion
```

### 3. Build the WASM binary (once or after Go bridge changes)

```bash
# From repo root
python tools/build_lattigo_wasm.py
```

### 4. Build the browser client

```bash
cd examples/c3ae-demo/client
npm install
npm run build
```

### 5. Run the server

```bash
cd examples/c3ae-demo/server
go run . ../model.orion ../client :8080
```

### 6. Open browser

Navigate to http://localhost:8080.

1. Click **Initialize Keys** — generates CKKS keys in browser, uploads to server
2. Upload a face image (JPEG/PNG)
3. Click **Encrypt & Infer** — encrypts image in browser, server runs FHE inference, browser decrypts result

## Standalone FHE Benchmark

```bash
python run_fhe.py --weights weights.pth --model model.orion --data-dir ./data/UTKFace --samples 3
```

## Architecture

```
Browser (WASM)                    Go Server
  |                                  |
  |-- GET /params ------------------>|  Load model.orion
  |<-- ckks_params, manifest --------|
  |                                  |
  |  [Generate SK, PK in browser]    |
  |                                  |
  |-- POST /session ---------------->|  Create session
  |<-- session_id -------------------|
  |                                  |
  |-- POST /keys/relin ------------->|  Upload RLK
  |-- POST /keys/galois/{el} ------->|  Stream Galois keys (one at a time)
  |-- POST /keys/finalize ---------->|  Validate + create evaluator
  |                                  |
  |  [User uploads face image]       |
  |  [Preprocess 64x64, normalize]   |
  |  [Encode + Encrypt in browser]   |
  |                                  |
  |-- POST /infer ------------------>|  FHE inference on ciphertext
  |<-- result ciphertext ------------|
  |                                  |
  |  [Decrypt + sigmoid → P(adult)]  |
  |  [Display: ADULT/MINOR]          |
```

## Model

| Property     | Value                                                |
| ------------ | ---------------------------------------------------- |
| Architecture | C3AE with stride-2 optimization                      |
| Input        | 64×64×3 RGB face image                               |
| Output       | Binary (adult/minor)                                 |
| Parameters   | 31,393                                               |
| Activations  | Quad (x²) instead of ReLU                            |
| Classifier   | Conv blocks → Flatten → FC(128,12) → Quad → FC(12,1) |

## CKKS Parameters

| Parameter | Value                                              |
| --------- | -------------------------------------------------- |
| LogN      | 15 (ring dim = 32,768)                             |
| LogQ      | [51, 40×15] (16 primes = 15 computation levels)    |
| LogP      | [50, 50, 50]                                       |
| LogScale  | 40                                                 |
| LogQP     | 801 (< 881 limit for 128-bit security at logN=15)  |
| Bootstrap | Not needed (15 levels sufficient for full network) |

Input (64×64×3 = 12,288 values) fits in a single ciphertext of 16,384 slots.

## Measurements

Run on immers.cloud VPS (cpu.16.128.240: 16 vCPUs, 128 GB RAM, Ubuntu 22.04).

| Metric             | Value               |
| ------------------ | ------------------- |
| Compilation time   | 2.4 min             |
| Model size         | 838 MB              |
| Compilation memory | 3.3 GB              |
| Key generation     | 83s (183 Galois)    |
| Eval keys size     | 10.24 GB            |
| **Inference time** | **139s per sample** |
| MAE vs cleartext   | 0.000000            |
| Peak server RSS    | 103 GB              |
| Cleartext accuracy | 97.9%               |
