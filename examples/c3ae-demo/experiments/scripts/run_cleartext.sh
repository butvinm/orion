#!/usr/bin/env bash
set -euo pipefail

# Cleartext experiment runner.
# Trains the ReLU variant (if needed), copies the existing Quad weights as the
# FHE variant baseline (if needed), and runs cleartext eval producing
# results/cleartext.csv.

# Assert venv is active.
python -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)' || {
    echo "ERROR: activate the venv first" >&2
    exit 1
}

# Normalize working directory to experiments/ regardless of where the user
# invoked the script from.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EXPERIMENTS_DIR}"

mkdir -p out results

# 1. Train ReLU variant if missing.
if [ ! -f out/weights_relu.pth ]; then
    echo "[run_cleartext] training ReLU variant (60 epochs)..."
    python -m models.train --variant relu --data-dir ./data/UTKFace --epochs 60
else
    echo "[run_cleartext] out/weights_relu.pth exists, skipping training"
fi

# 2. Reuse existing Quad weights for the FHE variant.
if [ ! -f out/weights_fhe.pth ]; then
    echo "[run_cleartext] copying ../weights.pth -> out/weights_fhe.pth"
    cp ../weights.pth out/weights_fhe.pth
else
    echo "[run_cleartext] out/weights_fhe.pth exists, skipping copy"
fi

# 3. Run cleartext eval (writes results/cleartext.csv).
echo "[run_cleartext] running cleartext eval..."
python -m models.eval

echo "[run_cleartext] DONE — see results/cleartext.csv"
