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

	// Serialization — SecretKey
	ns.Set("secretKeyMarshal", js.FuncOf(secretKeyMarshal))
	ns.Set("secretKeyUnmarshal", js.FuncOf(secretKeyUnmarshal))

	// Serialization — PublicKey
	ns.Set("publicKeyMarshal", js.FuncOf(publicKeyMarshal))
	ns.Set("publicKeyUnmarshal", js.FuncOf(publicKeyUnmarshal))

	// Serialization — RelinearizationKey
	ns.Set("relinKeyMarshal", js.FuncOf(relinKeyMarshal))
	ns.Set("relinKeyUnmarshal", js.FuncOf(relinKeyUnmarshal))

	// Serialization — GaloisKey
	ns.Set("galoisKeyMarshal", js.FuncOf(galoisKeyMarshal))
	ns.Set("galoisKeyUnmarshal", js.FuncOf(galoisKeyUnmarshal))

	// Serialization — Ciphertext
	ns.Set("ciphertextMarshal", js.FuncOf(ciphertextMarshal))
	ns.Set("ciphertextUnmarshal", js.FuncOf(ciphertextUnmarshal))
	ns.Set("ciphertextLevel", js.FuncOf(ciphertextLevel))

	// Serialization — Plaintext
	ns.Set("plaintextMarshal", js.FuncOf(plaintextMarshal))
	ns.Set("plaintextUnmarshal", js.FuncOf(plaintextUnmarshal))
	ns.Set("plaintextLevel", js.FuncOf(plaintextLevel))

	// MemEvaluationKeySet
	ns.Set("newMemEvalKeySet", js.FuncOf(newMemEvalKeySet))
	ns.Set("memEvalKeySetMarshal", js.FuncOf(memEvalKeySetMarshal))
	ns.Set("memEvalKeySetUnmarshal", js.FuncOf(memEvalKeySetUnmarshal))

	// Bootstrap
	ns.Set("newBootstrapParametersFromLiteral", js.FuncOf(newBootstrapParametersFromLiteral))
	ns.Set("btpParamsGenEvaluationKeys", js.FuncOf(btpParamsGenEvaluationKeys))

	// Readiness signal — MUST be last registration.
	ns.Set("__ready", true)

	js.Global().Set("lattigo", ns)

	// Keep Go runtime alive.
	select {}
}
