//go:build js && wasm

package main

import (
	"fmt"
	"syscall/js"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// encryptorHandle bundles encryptor with its parameters (needed to create ciphertexts).
type encryptorHandle struct {
	encryptor *rlwe.Encryptor
	params    *ckks.Parameters
}

// newEncryptor(paramsHID: number, pkHID: number) → {handle: number} | {error: string}
func newEncryptor(_ js.Value, args []js.Value) any {
	if len(args) < 2 {
		return errorResult("newEncryptor: missing arguments (paramsHID, pkHID)")
	}

	pObj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("newEncryptor: invalid params handle")
	}
	params := pObj.(*ckks.Parameters)

	pkObj, ok := Load(uint32(args[1].Int()))
	if !ok {
		return errorResult("newEncryptor: invalid public key handle")
	}
	pk := pkObj.(*rlwe.PublicKey)

	encryptor := ckks.NewEncryptor(*params, pk)
	return handleResult(Store(&encryptorHandle{encryptor: encryptor, params: params}))
}

// encryptorEncryptNew(encHID: number, ptHID: number) → {handle: number} | {error: string}
func encryptorEncryptNew(_ js.Value, args []js.Value) any {
	if len(args) < 2 {
		return errorResult("encryptorEncryptNew: missing arguments (encHID, ptHID)")
	}

	encObj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("encryptorEncryptNew: invalid encryptor handle")
	}
	eh := encObj.(*encryptorHandle)

	ptObj, ok := Load(uint32(args[1].Int()))
	if !ok {
		return errorResult("encryptorEncryptNew: invalid plaintext handle")
	}
	pt := ptObj.(*rlwe.Plaintext)

	ct := ckks.NewCiphertext(*eh.params, 1, pt.Level())
	if err := eh.encryptor.Encrypt(pt, ct); err != nil {
		return errorResult(fmt.Sprintf("encryptorEncryptNew: %v", err))
	}
	return handleResult(Store(ct))
}
