//go:build js && wasm

package main

import (
	"fmt"
	"syscall/js"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

// bootstrapExtendSK(btpParamsHID, skHID) → {skN2HID, kgN2HID} | {error}
//
// Extends the client SK to the bootstrap ring dimension / modulus chain.
// Returns handles for the extended SK and a KeyGenerator at bootstrap params.
// These are needed to stream-generate bootstrap Galois keys one at a time.
func bootstrapExtendSK(_ js.Value, args []js.Value) any {
	if len(args) < 2 {
		return errorResult("bootstrapExtendSK: need btpParamsHID and skHID")
	}

	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("bootstrapExtendSK: invalid btp params handle")
	}
	btpParams := obj.(*bootstrapping.Parameters)

	skObj, ok := Load(uint32(args[1].Int()))
	if !ok {
		return errorResult("bootstrapExtendSK: invalid sk handle")
	}
	skN1 := skObj.(*rlwe.SecretKey)

	paramsN2 := btpParams.BootstrappingParameters
	kgen := rlwe.NewKeyGenerator(paramsN2)

	var skN2 *rlwe.SecretKey

	if btpParams.ResidualParameters.N() != paramsN2.N() {
		skN2 = kgen.GenSecretKeyNew()
	} else {
		ringQ := paramsN2.RingQ()
		ringP := paramsN2.RingP()
		skN2 = rlwe.NewSecretKey(paramsN2)
		buff := ringQ.NewPoly()
		rlwe.ExtendBasisSmallNormAndCenterNTTMontgomery(ringQ, ringQ, skN1.Value.Q, buff, skN2.Value.Q)
		rlwe.ExtendBasisSmallNormAndCenterNTTMontgomery(ringQ, ringP, skN1.Value.Q, buff, skN2.Value.P)
	}

	result := js.Global().Get("Object").New()
	result.Set("skN2HID", int(Store(skN2)))
	result.Set("kgN2HID", int(Store(kgen)))
	// Store whether ring degrees differ (needed for switching keys)
	result.Set("needsRingSwitch", btpParams.ResidualParameters.N() != paramsN2.N())
	result.Set("isConjugateInvariant", btpParams.ResidualParameters.RingType() == ring.ConjugateInvariant)
	return result
}

// bootstrapGaloisElements(btpParamsHID) → number[]
//
// Returns the Galois elements required by the bootstrap circuit.
func bootstrapGaloisElements(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("bootstrapGaloisElements: need btpParamsHID")
	}

	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("bootstrapGaloisElements: invalid btp params handle")
	}
	btpParams := obj.(*bootstrapping.Parameters)

	paramsN2 := btpParams.BootstrappingParameters
	elements := btpParams.GaloisElements(paramsN2)
	elements = append(elements, paramsN2.GaloisElementForComplexConjugation())

	arr := js.Global().Get("Array").New(len(elements))
	for i, e := range elements {
		arr.SetIndex(i, int(e))
	}
	return arr
}

// bootstrapGenSwitchingKeys(btpParamsHID, skN1HID, skN2HID) → {keys: [{hid, name}...]}
//
// Generates the ring-switching and encapsulation evaluation keys.
// These are small (2-6 keys total) and can be uploaded individually.
// Each key is stored as a handle; caller marshals + uploads + frees each.
func bootstrapGenSwitchingKeys(_ js.Value, args []js.Value) any {
	if len(args) < 3 {
		return errorResult("bootstrapGenSwitchingKeys: need btpParamsHID, skN1HID, skN2HID")
	}

	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("bootstrapGenSwitchingKeys: invalid btp params handle")
	}
	btpParams := obj.(*bootstrapping.Parameters)

	skN1Obj, ok := Load(uint32(args[1].Int()))
	if !ok {
		return errorResult("bootstrapGenSwitchingKeys: invalid skN1 handle")
	}
	skN1 := skN1Obj.(*rlwe.SecretKey)

	skN2Obj, ok := Load(uint32(args[2].Int()))
	if !ok {
		return errorResult("bootstrapGenSwitchingKeys: invalid skN2 handle")
	}
	skN2 := skN2Obj.(*rlwe.SecretKey)

	paramsN2 := btpParams.BootstrappingParameters
	kgen := rlwe.NewKeyGenerator(paramsN2)

	type keyEntry struct {
		key  *rlwe.EvaluationKey
		name string
	}
	var keys []keyEntry

	if btpParams.ResidualParameters.N() != paramsN2.N() {
		if btpParams.ResidualParameters.RingType() == ring.ConjugateInvariant {
			evkC2R, evkR2C := kgen.GenEvaluationKeysForRingSwapNew(skN2, skN1)
			keys = append(keys,
				keyEntry{evkC2R, "EvkCmplxToReal"},
				keyEntry{evkR2C, "EvkRealToCmplx"},
			)
		} else {
			keys = append(keys,
				keyEntry{kgen.GenEvaluationKeyNew(skN1, skN2), "EvkN1ToN2"},
				keyEntry{kgen.GenEvaluationKeyNew(skN2, skN1), "EvkN2ToN1"},
			)
		}
	}

	// Encapsulation keys
	if btpParams.EphemeralSecretWeight > 0 {
		paramsSparse, _ := rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{
			LogN: paramsN2.LogN(),
			Q:    paramsN2.Q()[:1],
			P:    paramsN2.P()[:1],
		})
		kgenSparse := rlwe.NewKeyGenerator(paramsSparse)
		kgenDense := rlwe.NewKeyGenerator(paramsN2)
		skSparse := kgenSparse.GenSecretKeyWithHammingWeightNew(btpParams.EphemeralSecretWeight)
		keys = append(keys,
			keyEntry{kgenDense.GenEvaluationKeyNew(skN2, skSparse), "EvkDenseToSparse"},
			keyEntry{kgenDense.GenEvaluationKeyNew(skSparse, skN2), "EvkSparseToDense"},
		)
	}

	arr := js.Global().Get("Array").New(len(keys))
	for i, k := range keys {
		entry := js.Global().Get("Object").New()
		entry.Set("hid", int(Store(k.key)))
		entry.Set("name", k.name)
		arr.SetIndex(i, entry)
	}

	result := js.Global().Get("Object").New()
	result.Set("keys", arr)
	return result
}

// evalKeyMarshal(hid) → Uint8Array | {error}
// Marshal an individual rlwe.EvaluationKey by handle.
func evalKeyMarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("evalKeyMarshal: missing handle")
	}
	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("evalKeyMarshal: invalid handle")
	}

	var data []byte
	var err error

	switch key := obj.(type) {
	case *rlwe.EvaluationKey:
		data, err = key.MarshalBinary()
	case *rlwe.RelinearizationKey:
		data, err = key.MarshalBinary()
	case *rlwe.GaloisKey:
		data, err = key.MarshalBinary()
	default:
		return errorResult(fmt.Sprintf("evalKeyMarshal: unsupported type %T", obj))
	}

	if err != nil {
		return errorResult(fmt.Sprintf("evalKeyMarshal: %v", err))
	}
	return bytesToJS(data)
}

func init() {
	// These get registered from registerBootstrapStreaming, called in main.go
}

func registerBootstrapStreaming(ns js.Value) {
	ns.Set("bootstrapExtendSK", js.FuncOf(bootstrapExtendSK))
	ns.Set("bootstrapGaloisElements", js.FuncOf(bootstrapGaloisElements))
	ns.Set("bootstrapGenSwitchingKeys", js.FuncOf(bootstrapGenSwitchingKeys))
	ns.Set("evalKeyMarshal", js.FuncOf(evalKeyMarshal))
}

// newKeyGeneratorFromParams creates a KeyGenerator at the bootstrap parameters' ring.
// This lets the client use keyGenGenRelinKey/keyGenGenGaloisKey with skN2.
func newKeyGeneratorFromBootstrapParams(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("newKeyGeneratorFromBootstrapParams: need btpParamsHID")
	}
	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("newKeyGeneratorFromBootstrapParams: invalid handle")
	}
	btpParams := obj.(*bootstrapping.Parameters)

	// Cast ckks.Parameters to rlwe.Parameters for KeyGenerator
	params := ckks.Parameters(btpParams.BootstrappingParameters)
	kgen := rlwe.NewKeyGenerator(params)
	return handleResult(Store(kgen))
}
