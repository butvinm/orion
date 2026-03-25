#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../wasm"

mkdir -p "${OUT_DIR}"

echo "Building WASM..."
cd "${SCRIPT_DIR}"
GOOS=js GOARCH=wasm go build -ldflags="-s -w" -o "${OUT_DIR}/lattigo.wasm" .

SIZE=$(stat --printf="%s" "${OUT_DIR}/lattigo.wasm" 2>/dev/null || stat -f "%z" "${OUT_DIR}/lattigo.wasm")
SIZE_MB=$(awk "BEGIN {printf \"%.2f\", ${SIZE} / 1048576}")
echo "Built ${OUT_DIR}/lattigo.wasm (${SIZE_MB} MB)"

if [ "${SIZE}" -gt 15728640 ]; then
    echo "WARNING: WASM binary exceeds 15MB target (${SIZE_MB} MB)"
    exit 1
fi
