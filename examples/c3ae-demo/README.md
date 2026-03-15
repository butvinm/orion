# C3AE Age Verification — Encrypted FHE Demo

Privacy-preserving age verification using CKKS homomorphic encryption. A browser client generates keys, encrypts a face image, and sends the ciphertext to a Go server that runs FHE inference. The secret key and face image never leave the browser.

Based on the C3AE (Compact yet Comprehensive Age Estimation) architecture adapted for FHE: ReLU replaced with Quad (x^2), BatchNorm fused, binary classification (18+ adult/minor).

## Prerequisites

- Go 1.22+
- Python 3.11+ with a venv containing `orion-compiler`, `orion-evaluator`, and `lattigo`
- Node.js 18+
- UTKFace dataset (for training)
- **128+ GB RAM for compilation, 256+ GB RAM for full FHE inference** (see [Memory Blocker](#memory-blocker))

## Quick Start

### 1. Train the model

```bash
cd examples/c3ae-demo

# Download UTKFace dataset
python -c "import kagglehub; kagglehub.dataset_download('jangedoo/utkface-new')"

python train.py --data-dir ./data/UTKFace --epochs 60 --output weights.pth
```

### 2. Compile to .orion format

**Requires 128+ GB RAM.** Compilation holds all weight diagonals in memory.

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

**Requires 256+ GB RAM.** Loading the 5.5 GB `.orion` model allocates 125 GB of Go heap.

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
| Bootstrap | LogP=[61]x8, slots=(4096, 8192)          |

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

| Metric              | Value                            |
| ------------------- | -------------------------------- |
| Fit time            | 0.05s                            |
| Compile time        | 27.7 min                         |
| Model size (.orion) | **5.5 GB** (5,828,835,332 bytes) |
| Total diagonals     | 46,304+                          |
| Peak RSS            | **103 GB**                       |
| Input level         | 5                                |
| Galois elements     | 334                              |
| Bootstrap slots     | (4096, 8192)                     |
| Graph               | 18 nodes, 17 edges               |

### Model Loading (Go Evaluator)

| Metric             | Value                                     |
| ------------------ | ----------------------------------------- |
| File read time     | 3.3s                                      |
| Model load time    | **8 min 48s**                             |
| Go heap after load | **125 GB** (from 5.5 GB file, 23x blowup) |

### FHE Inference

**Not completed** — model loading (125 GB) + keygen + eval keys exceeds 128 GB. Requires 256+ GB RAM. See [Memory Blocker](#memory-blocker).

### Cleartext Baseline (for reference)

| Metric   | Value                                |
| -------- | ------------------------------------ |
| Accuracy | 97.9% (full test set, 3,557 samples) |
| FPR      | 6.1%                                 |
| FNR      | 1.3%                                 |

### Comparison with Previous Results (Old Orion)

| Metric                | Old Orion                  | Orion v2                      |
| --------------------- | -------------------------- | ----------------------------- |
| Compilation memory    | ~10 GB (streams to HDF5)   | **103 GB** (all in-memory)    |
| Model artifact        | ~8 GB (diags.h5 + keys.h5) | **5.5 GB** (.orion binary)    |
| Evaluator load memory | ~10 GB (lazy from HDF5)    | **125 GB** (all deserialized) |
| FHE inference         | 31.8s per sample           | Not reached                   |
| Accuracy              | 93.9%                      | 94.2% (same weights)          |
| FPR                   | 18.9%                      | 15.9% (same weights)          |
| Total server memory   | 10.19 GB                   | 125+ GB                       |

## Memory Blocker

Orion v2 cannot run FHE inference on C3AE with less than ~256 GB RAM. The bottleneck is the Go evaluator's model loading:

1. **Compilation** (Python): The compiler stores all 46,304 weight diagonals in memory to produce the `.orion` binary. Peak RSS: 103 GB. Old Orion streamed to HDF5.

2. **Model loading** (Go): `evaluator.LoadModel()` deserializes all diagonals from raw float64 into Lattigo's NTT-encoded ring polynomials (CRT form across multiple moduli + BSGS expansion). This causes a **23x memory blowup**: 5.5 GB file → 125 GB Go heap. Old Orion loaded diagonals on-demand from HDF5.

3. **Keygen + inference**: Additional 3-10 GB for evaluation keys, bootstrap keys, and ciphertexts. Combined with model loading, this exceeds 128 GB.

The old Orion experiment ran the full pipeline on ~10 GB because it used lazy HDF5 I/O. Orion v2's `.orion` binary format requires all data in memory simultaneously.

See [ISSUES.md](./ISSUES.md) for the full list of library issues encountered.
