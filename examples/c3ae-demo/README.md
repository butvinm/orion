# C3AE Age Verification — Encrypted FHE Demo

Privacy-preserving age verification using CKKS homomorphic encryption. A browser client generates keys, encrypts a face image, and sends the ciphertext to a Go server that runs FHE inference. The secret key and face image never leave the browser.

Based on the C3AE architecture adapted for FHE: ReLU replaced with Quad (x²), BatchNorm fused, binary classification (18+ adult/minor).

## Prerequisites

- Go 1.24+ (the `bench/` module's `go.mod` declares 1.24)
- Python 3.11+
- **Python deps via `uv sync` from the repo root.** The workspace pulls in `orion-v2-lattigo`, `orion-v2-compiler`, `orion-v2-evaluator`, `kagglehub`, `torchvision`, etc. The demo lives inside the workspace — there is no separate pip-install file at this layer.
- Node.js 18+ (for the WASM browser client)
- UTKFace dataset (downloaded once via `kagglehub`, see Quick Start)
- For FHE benchmarks: at least **64 GB RAM** for `logn=15`, **128 GB** for `logn=16` (smaller boxes will OOM mid-inference).

### PyPI install (alternative — using Orion as a library)

```bash
pip install orion-v2-lattigo orion-v2-compiler orion-v2-evaluator
```

This gives you the libraries to use in your own code. It does **not** ship the c3ae-demo files (model defs, bench, scripts, server, browser client) — those live in this repo. Anyone running this demo must clone the repo and `uv sync`.

## Quick Start (browser demo)

```bash
# From the repo root
uv sync                                   # install all Python deps incl. kagglehub
python tools/build_lattigo.py             # build the CGO shared library

cd examples/c3ae-demo
source ../../.venv/bin/activate

# Download UTKFace dataset and symlink ./data/UTKFace to the
# kagglehub cache (kagglehub extracts to ~/.cache/kagglehub/...).
mkdir -p data
python -c "
import kagglehub, os
p = kagglehub.dataset_download('jangedoo/utkface-new')
for sub in ('UTKFace', 'utkface_aligned_cropped/UTKFace', 'utkface_aligned_cropped/crop_part1'):
    cand = os.path.join(p, sub)
    if os.path.isdir(cand) and any(f.endswith('.jpg') for f in os.listdir(cand)):
        target = 'data/UTKFace'
        if not os.path.islink(target) and not os.path.isdir(target):
            os.symlink(cand, target)
        print('symlinked:', cand, '->', target)
        break
else:
    raise SystemExit('UTKFace jpg directory not found inside ' + p)
"

# 1. Train the FHE (Quad) variant
python -m models.train --variant fhe --data-dir ./data/UTKFace --epochs 60

# 2. Compile to .orion
python -m models.compile --variant fhe --config logn15 \
    --weights out/weights_fhe.pth \
    --output out/logn15/model.orion

# 3. Build the WASM binary (from repo root)
cd ../..
python tools/build_lattigo_wasm.py

# 4. Build the browser client
cd examples/c3ae-demo/client
npm install
npm run build

# 5. Run the server
cd ../server
go run . ../out/logn15/model.orion ../client :8080
```

Open <http://localhost:8080> and:

1. Click **Initialize Keys** — generates CKKS keys in browser, uploads to server.
2. Upload a face image (JPEG/PNG).
3. Click **Encrypt & Infer** — encrypts the image in the browser, server runs FHE inference, browser decrypts the result.

## Benchmarking guide

Reproduce the cleartext quality + FHE inference cost measurements yourself.

### Prerequisites

- Project venv (`uv sync` from repo root) with `kagglehub` installed.
- Go 1.24+ (the `bench/go.mod` requires it).
- UTKFace dataset (downloaded via `kagglehub`, see Quick Start).
- For FHE: at least 64 GB RAM for `logn=15`, 128 GB for `logn=16` (smaller boxes will OOM mid-inference).

### Cleartext

```sh
bash scripts/run_cleartext.sh
```

Trains the ReLU variant if missing (~30 min on CPU, ~3 min on GPU), trains the Quad variant if missing (same), then evaluates both via `python -m models.eval` on the test split. Writes `results/cleartext.csv`.

### FHE inference

```sh
# Build the bench binary once
cd bench && go build && cd ..

# Run for a config
bash scripts/run_fhe.sh logn15  # or logn16
```

Compiles the model, generates keys, runs encrypt + infer + decrypt for the 3 boundary samples (idx 12, 35, 44). Streams metrics to `results/<cfg>/run.jsonl` (one line per sample) so a crashed/OOM-killed run still preserves what completed. Idempotent — re-runs skip already-completed samples.

### Verify FHE correctness

```sh
python verify_fhe.py --config logn15
```

Compares each FHE-decrypted probability against the cleartext PyTorch forward of the same input. Writes `results/<cfg>/cleartext_vs_fhe.csv` and exits non-zero if any sample exceeds `--tol` (default 0.05).

### Aggregate the report

```sh
python build_results.py
```

Reads `results/cleartext.csv` and `results/<cfg>/run.jsonl` files, emits `results/results.md` with two markdown tables.

### Provisioning a fresh VPS for benchmarking

We provisioned three VPSes on immers.cloud during the 2026-05-09 run. The provisioning script at `/home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-fhe.sh` captures the apt deps + Go 1.24 + uv + UTKFace download in one shot; total provisioning takes ~5 min on a fresh `cpu.16.128.240`. Use it as a reference rather than copy-pasting commands. The plan at `/home/butvinm/Dev/orion/docs/plans/completed/2026-05-09-c3ae-vps-runs.md` documents the full sequence including measured timings and cost.

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

The two no-bootstrap CKKS configurations live in [`models/params.py`](models/params.py). Both share `log_default_scale = 40`, `ring_type = standard`, and **15 multiplicative levels** so the comparison isolates the effect of doubling the ring degree.

| Config   | LogN | LogQ               | LogP       | LogQP | Notes                         |
| -------- | ---- | ------------------ | ---------- | ----- | ----------------------------- |
| `logn15` | 15   | `[51] + [40] * 15` | `[50] * 4` | 851   | ≤ 881 dense bound at logn=15  |
| `logn16` | 16   | `[55] + [40] * 15` | `[55] * 6` | 985   | ≤ 1770 dense bound at logn=16 |

Input (64×64×3 = 12,288 values) fits in a single ciphertext at both ring degrees.

## Measurements

### Cleartext quality (UTKFace test split, n=3557; boundary band 16–20, n=161)

| variant | scope    | n    | FPR    | FNR    | Accuracy |
| ------- | -------- | ---- | ------ | ------ | -------- |
| relu    | overall  | 3557 | 0.1675 | 0.0190 | 0.9556   |
| relu    | boundary | 161  | 0.6515 | 0.1474 | 0.6460   |
| fhe     | overall  | 3557 | 0.2085 | 0.0268 | 0.9421   |
| fhe     | boundary | 161  | 0.7121 | 0.1579 | 0.6149   |

The Quad-FHE variant trades ~1–3 percentage points of accuracy for FHE compatibility. The 16–20 boundary band is brutal for both variants — the asymmetric loss (`fpr_weight=40`) doesn't fully overcome the dataset's 18% minor / 82% adult class imbalance.

### FHE inference cost (cpu.16.128.240: 16 vCPUs, 128 GB RAM; Go-only `bench` binary; 3 boundary samples)

| config | compile_s | compile_peak_rss_GB | keygen_s | evk_GB | mean_forward_s | peak_rss_GB   |
| ------ | --------- | ------------------- | -------- | ------ | -------------- | ------------- |
| logn15 | 160.2     | 12.87               | 44.1     | 7.19   | 157.0 ± 2.9    | 54.19 ± 0.29  |
| logn16 | 386.7     | 25.77               | 68.1     | 12.70  | 543.7 ± 219.4  | 114.37 ± 0.07 |

**Headline: peak server RSS dropped 47% (54.19 GB vs 103 GB) at `logn=15`** compared to the pre-Go-bench Python-wrapped pipeline measured on the same VPS. Forward time is roughly comparable (~+13%, 157s vs 139s). The RSS reduction confirms the Python wrapper added ~50 GB of overhead at `logn=15`. `logn=16` fits in 128 GB by ~10 GB margin — going larger at this depth requires a 256+ GB box.

For the full audit trail (per-sample JSONL, VPS rental cost log, cold-cache notes), see [`results/results.md`](results/results.md) and [`/home/butvinm/Dev/orion/docs/plans/completed/2026-05-09-c3ae-vps-runs.md`](../../docs/plans/completed/2026-05-09-c3ae-vps-runs.md).
