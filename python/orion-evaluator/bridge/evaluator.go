package main

// Bridge exports for the orion-evaluator Python package.
// Accepts raw Lattigo binary data (MarshalBinary bytes) — no Orion-specific
// key or ciphertext types cross the FFI boundary.

//#include <stdlib.h>
//#include <stdint.h>
import "C"

import (
	"encoding/json"
	"runtime/cgo"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"

	orion "github.com/baahl-nyu/orion"
	"github.com/baahl-nyu/orion/evaluator"
)

// =====================================================================
// Model
// =====================================================================

// EvalLoadModel loads a .orion v2 file and returns a Model handle.
//
//export EvalLoadModel
func EvalLoadModel(data *C.char, dataLen C.ulong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goData := cBytesToGoSlice(data, dataLen)
	model, err := evaluator.LoadModel(goData)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(model))
}

// EvalModelClientParams returns the CKKS params JSON, manifest JSON, and input level.
//
//export EvalModelClientParams
func EvalModelClientParams(modelH C.uintptr_t, paramsOut **C.char, paramsOutLen *C.ulong, manifestOut **C.char, manifestOutLen *C.ulong, inputLevelOut *C.int, errOut **C.char) {
	defer catchPanic(errOut)
	model := cgo.Handle(modelH).Value().(*evaluator.Model)
	params, manifest, inputLevel := model.ClientParams()

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		setErr(errOut, "marshal params: "+err.Error())
		return
	}
	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		setErr(errOut, "marshal manifest: "+err.Error())
		return
	}

	pPtr, pLen := goSliceToCBytes(paramsJSON)
	*paramsOut = pPtr
	*paramsOutLen = pLen

	mPtr, mLen := goSliceToCBytes(manifestJSON)
	*manifestOut = mPtr
	*manifestOutLen = mLen

	*inputLevelOut = C.int(inputLevel)
}

// EvalModelClose releases model resources (handle freed by DeleteHandle).
//
//export EvalModelClose
func EvalModelClose(modelH C.uintptr_t) {
	defer logPanic()
	// Model has no Close method — it's immutable. Handle deletion frees it.
	// This is a no-op placeholder for API symmetry.
}

// =====================================================================
// Evaluator
// =====================================================================

// EvalNewEvaluator creates a new Evaluator from CKKS params JSON and
// MemEvaluationKeySet binary bytes. btpKeysData/btpKeysDataLen are optional
// bootstrap evaluation keys (pass NULL/0 when not needed).
//
//export EvalNewEvaluator
func EvalNewEvaluator(paramsJSON *C.char, keysData *C.char, keysDataLen C.ulong, btpKeysData *C.char, btpKeysDataLen C.ulong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)

	// Parse CKKS params.
	var p orion.Params
	if err := json.Unmarshal([]byte(C.GoString(paramsJSON)), &p); err != nil {
		setErr(errOut, "parsing params: "+err.Error())
		return 0
	}
	ckksParams, err := p.NewCKKSParameters()
	if err != nil {
		setErr(errOut, "creating CKKS parameters: "+err.Error())
		return 0
	}

	// Unmarshal MemEvaluationKeySet.
	goKeysData := cBytesToGoSlice(keysData, keysDataLen)
	evk := new(rlwe.MemEvaluationKeySet)
	if err := evk.UnmarshalBinary(goKeysData); err != nil {
		setErr(errOut, "unmarshalling evaluation key set: "+err.Error())
		return 0
	}

	// Unmarshal bootstrap keys if provided.
	var btpKeys *bootstrapping.EvaluationKeys
	if btpKeysData != nil && btpKeysDataLen > 0 {
		goBtpData := cBytesToGoSlice(btpKeysData, btpKeysDataLen)
		btpKeys = new(bootstrapping.EvaluationKeys)
		if err := btpKeys.UnmarshalBinary(goBtpData); err != nil {
			setErr(errOut, "unmarshalling bootstrap keys: "+err.Error())
			return 0
		}
	}

	eval, err := evaluator.NewEvaluatorFromKeySet(ckksParams, evk, btpKeys)
	if err != nil {
		setErr(errOut, "creating evaluator: "+err.Error())
		return 0
	}

	return C.uintptr_t(cgo.NewHandle(eval))
}

// EvalForward runs the Forward pass. Accepts rlwe.Ciphertext.MarshalBinary bytes,
// returns rlwe.Ciphertext.MarshalBinary bytes.
//
//export EvalForward
func EvalForward(evalH C.uintptr_t, modelH C.uintptr_t, ctData *C.char, ctDataLen C.ulong, outLen *C.ulong, errOut **C.char) *C.char {
	defer catchPanic(errOut)

	eval := cgo.Handle(evalH).Value().(*evaluator.Evaluator)
	model := cgo.Handle(modelH).Value().(*evaluator.Model)

	// Unmarshal input ciphertext.
	goCtData := cBytesToGoSlice(ctData, ctDataLen)
	ct := new(rlwe.Ciphertext)
	if err := ct.UnmarshalBinary(goCtData); err != nil {
		setErr(errOut, "unmarshalling input ciphertext: "+err.Error())
		return nil
	}

	// Run Forward.
	result, err := eval.Forward(model, ct)
	if err != nil {
		setErr(errOut, "forward: "+err.Error())
		return nil
	}

	// Marshal output ciphertext.
	outData, err := result.MarshalBinary()
	if err != nil {
		setErr(errOut, "marshalling output ciphertext: "+err.Error())
		return nil
	}

	ptr, length := goSliceToCBytes(outData)
	*outLen = length
	return ptr
}

// EvalClose releases evaluator resources (handle freed by DeleteHandle).
//
//export EvalClose
func EvalClose(evalH C.uintptr_t) {
	defer logPanic()
	eval := cgo.Handle(evalH).Value().(*evaluator.Evaluator)
	eval.Close()
}

