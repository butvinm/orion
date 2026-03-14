//go:build js && wasm

package main

import (
	"syscall/js"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

// decryptorHandle bundles decryptor with its parameters (needed to create plaintexts).
type decryptorHandle struct {
	decryptor *rlwe.Decryptor
	params    *ckks.Parameters
}

// newDecryptor(paramsHID: number, skHID: number) → {handle: number} | {error: string}
func newDecryptor(_ js.Value, args []js.Value) any {
	if len(args) < 2 {
		return errorResult("newDecryptor: missing arguments (paramsHID, skHID)")
	}

	pObj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("newDecryptor: invalid params handle")
	}
	params := pObj.(*ckks.Parameters)

	skObj, ok := Load(uint32(args[1].Int()))
	if !ok {
		return errorResult("newDecryptor: invalid secret key handle")
	}
	sk := skObj.(*rlwe.SecretKey)

	decryptor := ckks.NewDecryptor(*params, sk)
	return handleResult(Store(&decryptorHandle{decryptor: decryptor, params: params}))
}

// decryptorDecryptNew(decHID: number, ctHID: number) → {handle: number} | {error: string}
func decryptorDecryptNew(_ js.Value, args []js.Value) any {
	if len(args) < 2 {
		return errorResult("decryptorDecryptNew: missing arguments (decHID, ctHID)")
	}

	decObj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("decryptorDecryptNew: invalid decryptor handle")
	}
	dh := decObj.(*decryptorHandle)

	ctObj, ok := Load(uint32(args[1].Int()))
	if !ok {
		return errorResult("decryptorDecryptNew: invalid ciphertext handle")
	}
	ct := ctObj.(*rlwe.Ciphertext)

	pt := ckks.NewPlaintext(*dh.params, ct.Level())
	dh.decryptor.Decrypt(ct, pt)
	return handleResult(Store(pt))
}
