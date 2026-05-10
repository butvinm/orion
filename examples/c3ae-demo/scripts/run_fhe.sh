#!/usr/bin/env bash
set -euo pipefail

# FHE experiment runner for a single CKKS configuration.
#
# Usage: bash scripts/run_fhe.sh <config>
# where <config> is one of: logn15, logn16
#
# Pipeline (each step idempotent — skipped if outputs already exist):
#   1. Compile model with the chosen CKKS params -> out/<cfg>/model.orion
#   2. Prep boundary-band inputs                  -> out/inputs/sample_*.bin
#   3. Build the bench Go binary                  -> bench/bench
#   4. Generate keys (with /usr/bin/time)         -> out/<cfg>/keys/{sk,evk}.bin
#   5. For each boundary sample: encrypt -> infer (measured) -> decrypt
#      Append per-sample metrics to results/<cfg>/run.jsonl

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <logn15|logn16>" >&2
    exit 1
fi

CFG="$1"
case "${CFG}" in
    logn15|logn16)
        ;;
    *)
        echo "ERROR: invalid config '${CFG}'. Must be one of: logn15, logn16" >&2
        echo "Usage: $0 <logn15|logn16>" >&2
        exit 1
        ;;
esac

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

mkdir -p "out/${CFG}/keys" "out/inputs" "results/${CFG}"

# Ensure FHE weights are in place (compile needs them).
# The pre-consolidation `c3ae-demo/weights.pth` fallback is gone — training
# is the only way to obtain Quad weights from a fresh checkout.
if [ ! -f out/weights_fhe.pth ]; then
    echo "[run_fhe:${CFG}] ERROR: out/weights_fhe.pth missing." >&2
    echo "  Run \`bash scripts/run_cleartext.sh\` first to train both" >&2
    echo "  variants (or \`python -m models.train --variant fhe" >&2
    echo "  --data-dir ./data/UTKFace --epochs 60\` to train just this one)." >&2
    exit 1
fi

# 1. Compile.
if [ ! -f "out/${CFG}/model.orion" ]; then
    echo "[run_fhe:${CFG}] compiling model..."
    python -m models.compile \
        --variant fhe \
        --config "${CFG}" \
        --weights out/weights_fhe.pth \
        --output "out/${CFG}/model.orion"
else
    echo "[run_fhe:${CFG}] out/${CFG}/model.orion exists, skipping compile"
fi

# 2. Prep boundary inputs (only if ground_truth.csv missing or has only header).
NEED_PREP=0
if [ ! -s out/inputs/ground_truth.csv ]; then
    NEED_PREP=1
elif [ "$(wc -l < out/inputs/ground_truth.csv)" -le 1 ]; then
    NEED_PREP=1
fi
if [ "${NEED_PREP}" -eq 1 ]; then
    echo "[run_fhe:${CFG}] preparing boundary-band inputs..."
    python -m models.prep_input --boundary-band --data-dir ./data/UTKFace
else
    echo "[run_fhe:${CFG}] out/inputs/ground_truth.csv populated, skipping prep_input"
fi

# 3. Build bench binary if missing.
if [ ! -x bench/bench ]; then
    echo "[run_fhe:${CFG}] building bench binary..."
    (cd bench && go build)
else
    echo "[run_fhe:${CFG}] bench/bench exists, skipping build"
fi

# 4. Keygen.
if [ ! -f "out/${CFG}/keys/sk.bin" ]; then
    echo "[run_fhe:${CFG}] keygen..."
    /usr/bin/time -v ./bench/bench keygen \
        --model "out/${CFG}/model.orion" \
        --out "out/${CFG}/keys/" \
        2> "results/${CFG}/keygen_time.log"
else
    echo "[run_fhe:${CFG}] out/${CFG}/keys/sk.bin exists, skipping keygen"
fi

# 5. Per-sample loop.
RUN_JSONL="results/${CFG}/run.jsonl"
INFER_LOG="results/${CFG}/infer_time.log"
touch "${RUN_JSONL}"

# Pre-flight: ground_truth.csv must exist before we feed it into the loop.
# Process substitution (`done < <(awk ...)`) silently swallows awk's
# nonzero exit code, so a missing CSV would produce an empty iteration —
# the script would "succeed" without doing anything. Fail loudly here.
if [ ! -f out/inputs/ground_truth.csv ]; then
    echo "[run_fhe:${CFG}] ERROR: out/inputs/ground_truth.csv missing — prep_input did not produce it." >&2
    exit 1
fi

# Iterate idx values from ground_truth.csv (skip header).
while IFS= read -r idx; do
    [ -z "${idx}" ] && continue

    # Idempotency: skip if this sample is already in run.jsonl.
    # Use a non-digit boundary class instead of GNU-specific '\b' so the
    # pattern is portable to other greps. The line shape we're matching is
    # `..."sample_idx":N,...` so '[^0-9]' after the digits is sufficient.
    if grep -qE "\"sample_idx\":${idx}([^0-9]|\$)" "${RUN_JSONL}" 2>/dev/null; then
        echo "[run_fhe:${CFG}] sample_idx=${idx} already in run.jsonl, skipping"
        continue
    fi

    INPUT_BIN="out/inputs/sample_${idx}.bin"
    CT_BIN="out/${CFG}/ct_${idx}.bin"
    RESULT_BIN="out/${CFG}/result_${idx}.bin"
    DECRYPT_JSON="results/${CFG}/decrypt_${idx}.json"

    echo "[run_fhe:${CFG}] sample_idx=${idx}: encrypt"
    ./bench/bench encrypt \
        --model "out/${CFG}/model.orion" \
        --sk "out/${CFG}/keys/sk.bin" \
        --input "${INPUT_BIN}" \
        --out "${CT_BIN}"

    echo "[run_fhe:${CFG}] sample_idx=${idx}: infer (measured)"
    /usr/bin/time -v ./bench/bench infer \
        --model "out/${CFG}/model.orion" \
        --evk "out/${CFG}/keys/evk.bin" \
        --ct "${CT_BIN}" \
        --out "${RESULT_BIN}" \
        --metrics "${RUN_JSONL}" \
        --sample-idx "${idx}" \
        2>> "${INFER_LOG}"

    echo "[run_fhe:${CFG}] sample_idx=${idx}: decrypt"
    ./bench/bench decrypt \
        --model "out/${CFG}/model.orion" \
        --sk "out/${CFG}/keys/sk.bin" \
        --ct "${RESULT_BIN}" \
        > "${DECRYPT_JSON}"
done < <(awk -F, 'NR>1 {print $1}' out/inputs/ground_truth.csv)

echo "[run_fhe:${CFG}] DONE — see ${RUN_JSONL}"
