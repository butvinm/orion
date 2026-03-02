//go:build js && wasm

package main

import (
	"syscall/js"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

// newKeyGenerator(paramsHID: number) → {handle: number} | {error: string}
func newKeyGenerator(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("newKeyGenerator: missing paramsHID argument")
	}
	p, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("newKeyGenerator: invalid params handle")
	}
	params := p.(*ckks.Parameters)
	kg := rlwe.NewKeyGenerator(*params)
	return handleResult(Store(kg))
}

// keyGenGenSecretKey(kgHID: number) → {handle: number} | {error: string}
func keyGenGenSecretKey(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("keyGenGenSecretKey: missing kgHID argument")
	}
	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("keyGenGenSecretKey: invalid keygen handle")
	}
	kg := obj.(*rlwe.KeyGenerator)
	sk := kg.GenSecretKeyNew()
	return handleResult(Store(sk))
}

// keyGenGenPublicKey(kgHID: number, skHID: number) → {handle: number} | {error: string}
func keyGenGenPublicKey(_ js.Value, args []js.Value) any {
	if len(args) < 2 {
		return errorResult("keyGenGenPublicKey: missing arguments (kgHID, skHID)")
	}
	kgObj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("keyGenGenPublicKey: invalid keygen handle")
	}
	skObj, ok := Load(uint32(args[1].Int()))
	if !ok {
		return errorResult("keyGenGenPublicKey: invalid secret key handle")
	}
	kg := kgObj.(*rlwe.KeyGenerator)
	sk := skObj.(*rlwe.SecretKey)
	pk := kg.GenPublicKeyNew(sk)
	return handleResult(Store(pk))
}

// keyGenGenRelinKey(kgHID: number, skHID: number) → {handle: number} | {error: string}
func keyGenGenRelinKey(_ js.Value, args []js.Value) any {
	if len(args) < 2 {
		return errorResult("keyGenGenRelinKey: missing arguments (kgHID, skHID)")
	}
	kgObj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("keyGenGenRelinKey: invalid keygen handle")
	}
	skObj, ok := Load(uint32(args[1].Int()))
	if !ok {
		return errorResult("keyGenGenRelinKey: invalid secret key handle")
	}
	kg := kgObj.(*rlwe.KeyGenerator)
	sk := skObj.(*rlwe.SecretKey)
	rlk := kg.GenRelinearizationKeyNew(sk)
	return handleResult(Store(rlk))
}

// keyGenGenGaloisKey(kgHID: number, skHID: number, galoisElement: number) → {handle: number} | {error: string}
func keyGenGenGaloisKey(_ js.Value, args []js.Value) any {
	if len(args) < 3 {
		return errorResult("keyGenGenGaloisKey: missing arguments (kgHID, skHID, galoisElement)")
	}
	kgObj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("keyGenGenGaloisKey: invalid keygen handle")
	}
	skObj, ok := Load(uint32(args[1].Int()))
	if !ok {
		return errorResult("keyGenGenGaloisKey: invalid secret key handle")
	}
	kg := kgObj.(*rlwe.KeyGenerator)
	sk := skObj.(*rlwe.SecretKey)
	galEl := uint64(args[2].Float())
	gk := kg.GenGaloisKeyNew(galEl, sk)
	return handleResult(Store(gk))
}

