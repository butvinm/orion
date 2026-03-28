# C3AE Age Verification — Encrypted FHE Demo

Privacy-preserving age verification using CKKS homomorphic encryption. A browser client generates keys, encrypts a face image, and sends the ciphertext to a Go server that runs FHE inference. The secret key and face image never leave the browser.

Based on the C3AE (Compact yet Comprehensive Age Estimation) architecture adapted for FHE: ReLU replaced with Quad (x^2), BatchNorm fused, binary classification (18+ adult/minor).

## Prerequisites

- Go 1.22+
- Python 3.11+ with a venv containing `orion-compiler`, `orion-evaluator`, and `lattigo`
- Node.js 18+
- UTKFace dataset (for training)
- ~20 GB RAM for the Go server during FHE inference (14 GB for compilation alone)

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
  |-- POST /keys/bootstrap --------->|  Upload bootstrap keys (if needed)
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

| Property     | Value                                                    |
| ------------ | -------------------------------------------------------- |
| Architecture | C3AE with stride-2 optimization                          |
| Input        | 64x64x3 RGB face image                                   |
| Output       | Binary (adult/minor)                                     |
| Parameters   | 31,393                                                   |
| Activations  | Quad (x^2) instead of ReLU                               |
| Classifier   | Conv blocks -> Flatten -> FC(128,12) -> Quad -> FC(12,1) |

## CKKS Parameters

| Parameter | Value                                              |
| --------- | -------------------------------------------------- |
| LogN      | 15 (ring dim = 32,768)                             |
| LogQ      | [51, 40×15] (16 primes = 15 computation levels)    |
| LogP      | [50, 50, 50]                                       |
| LogScale  | 40                                                 |
| Ring type | Standard                                           |
| LogQP     | 801 (< 881 limit for 128-bit security at logN=15)  |
| Bootstrap | Not needed (15 levels sufficient for full network) |

Input (64×64×3 = 12,288 values) fits in a single ciphertext of 16,384 slots.

## Measurements

Run on immers.cloud VPS (cpu.16.128.240: 16 vCPUs, 128 GB RAM, Ubuntu 22.04).

### Training

| Metric        | Value                                   |
| ------------- | --------------------------------------- |
| Dataset       | UTKFace, 23,708 images (70/15/15 split) |
| Epochs        | 60                                      |
| Training time | 27.5 min (27.4s/epoch)                  |
| Test accuracy | **94.2%**                               |
| Test FPR      | 15.9%                                   |
| Test FNR      | 3.7%                                    |
| Peak RSS      | 880 MB                                  |

### Compilation

| Metric              | Value |
| ------------------- | ----- |
| Fit time            | TBD   |
| Compile time        | TBD   |
| Model size (.orion) | TBD   |
| Peak RSS            | TBD   |

### Evaluator Setup

| Metric          | Value                          |
| --------------- | ------------------------------ |
| Model load time | TBD                            |
| Key generation  | TBD (183 Galois, no bootstrap) |
| Eval keys size  | TBD                            |
| Bootstrap keys  | None                           |
| Evaluator init  | TBD                            |
| RSS delta       | TBD                            |

### FHE Inference

| Metric               | Value |
| -------------------- | ----- |
| Encryption time      | TBD   |
| **Inference time**   | TBD   |
| Decryption time      | TBD   |
| **MAE vs cleartext** | TBD   |
| Peak server RSS      | TBD   |

### Cleartext Baseline

| Metric   | Value                                |
| -------- | ------------------------------------ |
| Accuracy | 97.9% (full test set, 3,557 samples) |
| FPR      | 6.1%                                 |
| FNR      | 1.3%                                 |

### Comparison with Previous Results

| Metric             | Old Orion (logN=14) | Orion v2 (logN=14, insecure) | Orion v2 (logN=15, secure) |
| ------------------ | ------------------- | ---------------------------- | -------------------------- |
| Security           | ~33 bits            | ~33 bits                     | **128 bits**               |
| Bootstrap          | Yes                 | Yes                          | **None**                   |
| Compilation time   | ~2 min              | 1.5 min                      | TBD                        |
| Compilation memory | ~10 GB              | 6.7 GB                       | TBD                        |
| Model artifact     | ~8 GB (HDF5)        | 427 MB (.orion)              | TBD                        |
| FHE inference      | 31.8s per sample    | 57s per sample               | TBD                        |
| Accuracy           | 93.9%               | 94.2%                        | TBD                        |
| Peak server RSS    | 10.19 GB            | 18.5 GB                      | TBD                        |
