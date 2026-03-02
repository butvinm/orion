//go:build js && wasm

package main

import "syscall/js"

func main() {
	ns := js.Global().Get("Object").New()

	// deleteHandle(handleID) — frees a Go-side handle. Idempotent.
	ns.Set("deleteHandle", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) < 1 {
			return nil
		}
		id := uint32(args[0].Int())
		Delete(id)
		return nil
	}))

	// CKKS Parameters
	ns.Set("newCKKSParams", js.FuncOf(newCKKSParams))
	ns.Set("ckksMaxSlots", js.FuncOf(ckksMaxSlots))
	ns.Set("ckksMaxLevel", js.FuncOf(ckksMaxLevel))
	ns.Set("ckksDefaultScale", js.FuncOf(ckksDefaultScale))
	ns.Set("ckksGaloisElement", js.FuncOf(ckksGaloisElement))
	ns.Set("ckksModuliChain", js.FuncOf(ckksModuliChain))
	ns.Set("ckksAuxModuliChain", js.FuncOf(ckksAuxModuliChain))

	// Key Generation
	ns.Set("newKeyGenerator", js.FuncOf(newKeyGenerator))
	ns.Set("keyGenGenSecretKey", js.FuncOf(keyGenGenSecretKey))
	ns.Set("keyGenGenPublicKey", js.FuncOf(keyGenGenPublicKey))
	ns.Set("keyGenGenRelinKey", js.FuncOf(keyGenGenRelinKey))
	ns.Set("keyGenGenGaloisKey", js.FuncOf(keyGenGenGaloisKey))

	// Encoder
	ns.Set("newEncoder", js.FuncOf(newEncoder))
	ns.Set("encoderEncode", js.FuncOf(encoderEncode))
	ns.Set("encoderDecode", js.FuncOf(encoderDecode))

	// Encryptor
	ns.Set("newEncryptor", js.FuncOf(newEncryptor))
	ns.Set("encryptorEncryptNew", js.FuncOf(encryptorEncryptNew))

	// Decryptor
	ns.Set("newDecryptor", js.FuncOf(newDecryptor))
	ns.Set("decryptorDecryptNew", js.FuncOf(decryptorDecryptNew))

	// Readiness signal — MUST be last registration.
	ns.Set("__ready", true)

	js.Global().Set("lattigo", ns)

	// Keep Go runtime alive.
	select {}
}
