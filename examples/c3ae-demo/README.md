# C3AE Age Verification — Encrypted FHE Demo

Privacy-preserving age verification using CKKS homomorphic encryption. A browser client generates keys, encrypts a face image, and sends the ciphertext to a Go server that runs FHE inference. The secret key and face image never leave the browser.

Based on the C3AE (Compact yet Comprehensive Age Estimation) architecture adapted for FHE: ReLU replaced with Quad (x^2), BatchNorm fused, binary classification (18+ adult/minor).

## Prerequisites

- Go 1.22+
- Python 3.11+ with a venv containing `orion-compiler`, `orion-evaluator`, and `lattigo`
- Node.js 18+
- UTKFace dataset (for training)
- ~33 GB RAM for full FHE inference (7 GB for compilation alone)

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

| Parameter | Value                                    |
| --------- | ---------------------------------------- |
| LogN      | 14 (ring dim = 16,384)                   |
| LogQ      | [55, 40, 40, 40, 40, 40, 40, 40, 40, 40] |
| LogP      | [61, 61, 61]                             |
| LogScale  | 40                                       |
| H         | 192                                      |
| Ring type | Standard (bootstrap-compatible)          |
| Bootstrap | LogP=[61]x8                              |

Input (64x64x3 = 12,288 values) is split across 2 ciphertexts of 8,192 slots each.

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

| Metric              | Value                          |
| ------------------- | ------------------------------ |
| Fit time            | 0.5s                           |
| Compile time        | **1.5 min**                    |
| Model size (.orion) | **427 MB** (447,638,484 bytes) |
| Peak RSS            | **6.7 GB**                     |

### Evaluator Setup

| Metric          | Value                        |
| --------------- | ---------------------------- |
| Model load time | 1.3s                         |
| Key generation  | 28s (334 Galois + bootstrap) |
| Eval keys size  | 2.26 GB                      |
| Bootstrap keys  | 1.03 GB                      |
| Evaluator init  | 6s                           |
| RSS delta       | 10.4 GB                      |

### FHE Inference

| Metric               | Value              |
| -------------------- | ------------------ |
| Encryption time      | 0.067s             |
| **Inference time**   | **57s per sample** |
| Decryption time      | 0.008s             |
| **MAE vs cleartext** | **0.000000**       |
| Peak RSS             | 33 GB              |

### Cleartext Baseline

| Metric   | Value                                |
| -------- | ------------------------------------ |
| Accuracy | 97.9% (full test set, 3,557 samples) |
| FPR      | 6.1%                                 |
| FNR      | 1.3%                                 |

### Comparison with Previous Results (Old Orion)

| Metric             | Old Orion        | Orion v2                 |
| ------------------ | ---------------- | ------------------------ |
| Compilation time   | ~2 min           | **1.5 min**              |
| Compilation memory | ~10 GB           | **6.7 GB**               |
| Model artifact     | ~8 GB (HDF5)     | **427 MB** (.orion)      |
| FHE inference      | 31.8s per sample | **57s per sample**       |
| Accuracy           | 93.9%            | **94.2%** (same weights) |
| FPR                | 18.9%            | **15.9%** (same weights) |
| Peak server memory | 10.19 GB         | **33 GB**                |
