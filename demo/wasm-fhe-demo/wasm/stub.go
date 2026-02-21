//go:build !(js && wasm)

// Package main provides a WASM binary wrapping orionclient for browser-based
// FHE key generation, encryption, and decryption. On non-WASM platforms this
// stub exists solely so that `go test` can compile the package.
package main

func main() {}
