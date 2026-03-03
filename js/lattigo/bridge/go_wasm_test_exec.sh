#!/bin/sh
# Wrapper called by: go test -exec ./go_wasm_test_exec.sh <wasm-binary> [args...]
# Resolves GOROOT at runtime so the JS runner can load wasm_exec.js.
GOROOT=$(go env GOROOT)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec node --stack-size=8192 "${SCRIPT_DIR}/go_wasm_test_exec.cjs" "${GOROOT}" "$@"
