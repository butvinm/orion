#!/usr/bin/env bash
set -euo pipefail

# Cleartext experiment runner.
# Trains both the ReLU and FHE-Quad variants (if missing) and runs cleartext
# eval producing results/cleartext.csv.

# Assert venv is active.
python -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)' || {
    echo "ERROR: activate the venv first" >&2
    exit 1
}

# Normalize working directory to the c3ae-demo root regardless of where
# the user invoked the script from.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEMO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DEMO_DIR}"

mkdir -p out results

# 1. Train ReLU variant if missing.
if [ ! -f out/weights_relu.pth ]; then
    echo "[run_cleartext] training ReLU variant (60 epochs)..."
    python -m models.train --variant relu --data-dir ./data/UTKFace --epochs 60
else
    echo "[run_cleartext] out/weights_relu.pth exists, skipping training"
fi

# 2. Train FHE (Quad) variant if missing. The pre-consolidation
#    `c3ae-demo/weights.pth` fallback is gone — training is the only way
#    to obtain Quad weights from a fresh checkout.
if [ ! -f out/weights_fhe.pth ]; then
    echo "[run_cleartext] training FHE (Quad) variant (60 epochs)..."
    python -m models.train --variant fhe --data-dir ./data/UTKFace --epochs 60
else
    echo "[run_cleartext] out/weights_fhe.pth exists, skipping training"
fi

# 3. Run cleartext eval (writes results/cleartext.csv).
echo "[run_cleartext] running cleartext eval..."
python -m models.eval

echo "[run_cleartext] DONE — see results/cleartext.csv"
