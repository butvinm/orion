#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WASM_DIR="$SCRIPT_DIR/../wasm"
CLIENT_DIR="$SCRIPT_DIR/../client"

echo "Building WASM binary..."
cd "$WASM_DIR"
make build
echo "  -> $(ls -lh "$CLIENT_DIR/orion.wasm" | awk '{print $5}') $CLIENT_DIR/orion.wasm"

echo "Copying wasm_exec.js from Go toolchain..."
GOROOT="$(go env GOROOT)"
WASM_EXEC=""
for candidate in "$GOROOT/misc/wasm/wasm_exec.js" "$GOROOT/lib/wasm/wasm_exec.js"; do
    if [ -f "$candidate" ]; then
        WASM_EXEC="$candidate"
        break
    fi
done
if [ -z "$WASM_EXEC" ]; then
    echo "ERROR: wasm_exec.js not found in $GOROOT/{misc,lib}/wasm/"
    exit 1
fi
cp "$WASM_EXEC" "$CLIENT_DIR/wasm_exec.js"
echo "  -> $CLIENT_DIR/wasm_exec.js"

echo "Done."
