//go:build js && wasm

package main

import (
	"fmt"
	"syscall/js"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
)

// --- SecretKey ---

// secretKeyMarshal(skHID: number) → Uint8Array | {error: string}
func secretKeyMarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("secretKeyMarshal: missing skHID argument")
	}
	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("secretKeyMarshal: invalid secret key handle")
	}
	sk := obj.(*rlwe.SecretKey)
	data, err := sk.MarshalBinary()
	if err != nil {
		return errorResult(fmt.Sprintf("secretKeyMarshal: %v", err))
	}
	return bytesToJS(data)
}

// secretKeyUnmarshal(bytes: Uint8Array) → {handle: number} | {error: string}
func secretKeyUnmarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("secretKeyUnmarshal: missing bytes argument")
	}
	data := jsToBytes(args[0])
	sk := new(rlwe.SecretKey)
	if err := sk.UnmarshalBinary(data); err != nil {
		return errorResult(fmt.Sprintf("secretKeyUnmarshal: %v", err))
	}
	return handleResult(Store(sk))
}

// --- PublicKey ---

// publicKeyMarshal(pkHID: number) → Uint8Array | {error: string}
func publicKeyMarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("publicKeyMarshal: missing pkHID argument")
	}
	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("publicKeyMarshal: invalid public key handle")
	}
	pk := obj.(*rlwe.PublicKey)
	data, err := pk.MarshalBinary()
	if err != nil {
		return errorResult(fmt.Sprintf("publicKeyMarshal: %v", err))
	}
	return bytesToJS(data)
}

// publicKeyUnmarshal(bytes: Uint8Array) → {handle: number} | {error: string}
func publicKeyUnmarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("publicKeyUnmarshal: missing bytes argument")
	}
	data := jsToBytes(args[0])
	pk := new(rlwe.PublicKey)
	if err := pk.UnmarshalBinary(data); err != nil {
		return errorResult(fmt.Sprintf("publicKeyUnmarshal: %v", err))
	}
	return handleResult(Store(pk))
}

// --- RelinearizationKey ---

// relinKeyMarshal(rlkHID: number) → Uint8Array | {error: string}
func relinKeyMarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("relinKeyMarshal: missing rlkHID argument")
	}
	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("relinKeyMarshal: invalid relinearization key handle")
	}
	rlk := obj.(*rlwe.RelinearizationKey)
	data, err := rlk.MarshalBinary()
	if err != nil {
		return errorResult(fmt.Sprintf("relinKeyMarshal: %v", err))
	}
	return bytesToJS(data)
}

// relinKeyUnmarshal(bytes: Uint8Array) → {handle: number} | {error: string}
func relinKeyUnmarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("relinKeyUnmarshal: missing bytes argument")
	}
	data := jsToBytes(args[0])
	rlk := new(rlwe.RelinearizationKey)
	if err := rlk.UnmarshalBinary(data); err != nil {
		return errorResult(fmt.Sprintf("relinKeyUnmarshal: %v", err))
	}
	return handleResult(Store(rlk))
}

// --- GaloisKey ---

// galoisKeyMarshal(gkHID: number) → Uint8Array | {error: string}
func galoisKeyMarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("galoisKeyMarshal: missing gkHID argument")
	}
	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("galoisKeyMarshal: invalid galois key handle")
	}
	gk := obj.(*rlwe.GaloisKey)
	data, err := gk.MarshalBinary()
	if err != nil {
		return errorResult(fmt.Sprintf("galoisKeyMarshal: %v", err))
	}
	return bytesToJS(data)
}

// galoisKeyUnmarshal(bytes: Uint8Array) → {handle: number} | {error: string}
func galoisKeyUnmarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("galoisKeyUnmarshal: missing bytes argument")
	}
	data := jsToBytes(args[0])
	gk := new(rlwe.GaloisKey)
	if err := gk.UnmarshalBinary(data); err != nil {
		return errorResult(fmt.Sprintf("galoisKeyUnmarshal: %v", err))
	}
	return handleResult(Store(gk))
}

// --- Ciphertext ---

// ciphertextMarshal(ctHID: number) → Uint8Array | {error: string}
func ciphertextMarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("ciphertextMarshal: missing ctHID argument")
	}
	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("ciphertextMarshal: invalid ciphertext handle")
	}
	ct := obj.(*rlwe.Ciphertext)
	data, err := ct.MarshalBinary()
	if err != nil {
		return errorResult(fmt.Sprintf("ciphertextMarshal: %v", err))
	}
	return bytesToJS(data)
}

// ciphertextUnmarshal(bytes: Uint8Array) → {handle: number} | {error: string}
func ciphertextUnmarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("ciphertextUnmarshal: missing bytes argument")
	}
	data := jsToBytes(args[0])
	ct := new(rlwe.Ciphertext)
	if err := ct.UnmarshalBinary(data); err != nil {
		return errorResult(fmt.Sprintf("ciphertextUnmarshal: %v", err))
	}
	return handleResult(Store(ct))
}

// ciphertextLevel(ctHID: number) → number | {error: string}
func ciphertextLevel(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("ciphertextLevel: missing ctHID argument")
	}
	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("ciphertextLevel: invalid ciphertext handle")
	}
	ct := obj.(*rlwe.Ciphertext)
	return ct.Level()
}

// --- Plaintext ---

// plaintextMarshal(ptHID: number) → Uint8Array | {error: string}
func plaintextMarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("plaintextMarshal: missing ptHID argument")
	}
	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("plaintextMarshal: invalid plaintext handle")
	}
	pt := obj.(*rlwe.Plaintext)
	data, err := pt.MarshalBinary()
	if err != nil {
		return errorResult(fmt.Sprintf("plaintextMarshal: %v", err))
	}
	return bytesToJS(data)
}

// plaintextUnmarshal(bytes: Uint8Array) → {handle: number} | {error: string}
func plaintextUnmarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("plaintextUnmarshal: missing bytes argument")
	}
	data := jsToBytes(args[0])
	pt := new(rlwe.Plaintext)
	if err := pt.UnmarshalBinary(data); err != nil {
		return errorResult(fmt.Sprintf("plaintextUnmarshal: %v", err))
	}
	return handleResult(Store(pt))
}

// plaintextLevel(ptHID: number) → number | {error: string}
func plaintextLevel(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("plaintextLevel: missing ptHID argument")
	}
	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("plaintextLevel: invalid plaintext handle")
	}
	pt := obj.(*rlwe.Plaintext)
	return pt.Level()
}

// --- MemEvaluationKeySet ---

// newMemEvalKeySet(rlkHID: number|null, galoisKeyHIDs: Array<number>) → {handle: number} | {error: string}
// rlkHID can be 0 or null to indicate no relinearization key.
func newMemEvalKeySet(_ js.Value, args []js.Value) any {
	if len(args) < 2 {
		return errorResult("newMemEvalKeySet: missing arguments (rlkHID, galoisKeyHIDs)")
	}

	// rlkHID — 0 or null means no RLK
	var rlk *rlwe.RelinearizationKey
	if !args[0].IsNull() && !args[0].IsUndefined() && args[0].Int() != 0 {
		obj, ok := Load(uint32(args[0].Int()))
		if !ok {
			return errorResult("newMemEvalKeySet: invalid relinearization key handle")
		}
		rlk = obj.(*rlwe.RelinearizationKey)
	}

	// galoisKeyHIDs — JS array of handle IDs
	gkArr := args[1]
	n := gkArr.Length()
	galKeys := make([]*rlwe.GaloisKey, n)
	for i := 0; i < n; i++ {
		gkHID := uint32(gkArr.Index(i).Int())
		obj, ok := Load(gkHID)
		if !ok {
			return errorResult(fmt.Sprintf("newMemEvalKeySet: invalid galois key handle at index %d", i))
		}
		galKeys[i] = obj.(*rlwe.GaloisKey)
	}

	evk := rlwe.NewMemEvaluationKeySet(rlk, galKeys...)
	return handleResult(Store(evk))
}

// memEvalKeySetMarshal(evkHID: number) → Uint8Array | {error: string}
func memEvalKeySetMarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("memEvalKeySetMarshal: missing evkHID argument")
	}
	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("memEvalKeySetMarshal: invalid eval key set handle")
	}
	evk := obj.(*rlwe.MemEvaluationKeySet)
	data, err := evk.MarshalBinary()
	if err != nil {
		return errorResult(fmt.Sprintf("memEvalKeySetMarshal: %v", err))
	}
	return bytesToJS(data)
}

// memEvalKeySetUnmarshal(bytes: Uint8Array) → {handle: number} | {error: string}
func memEvalKeySetUnmarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("memEvalKeySetUnmarshal: missing bytes argument")
	}
	data := jsToBytes(args[0])
	evk := new(rlwe.MemEvaluationKeySet)
	if err := evk.UnmarshalBinary(data); err != nil {
		return errorResult(fmt.Sprintf("memEvalKeySetUnmarshal: %v", err))
	}
	return handleResult(Store(evk))
}
