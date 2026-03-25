package main

// Bridge exports for the orion-evaluator Python package.
// Accepts raw Lattigo binary data (MarshalBinary bytes) — no Orion-specific
// key or ciphertext types cross the FFI boundary.

//#include <stdlib.h>
//#include <stdint.h>
import "C"

import (
	"encoding/binary"
	"encoding/json"
	"runtime/cgo"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"

	orion "github.com/butvinm/orion"
	"github.com/butvinm/orion/evaluator"
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

// EvalForward runs the Forward pass with multiple input/output CTs.
// ctData is a length-prefixed concatenation of marshalled CTs:
//
//	[uint64 len1][ct1 bytes][uint64 len2][ct2 bytes]...
//
// numCTs specifies how many CTs are in the buffer.
// Returns the same format for output CTs.
//
//export EvalForward
func EvalForward(evalH C.uintptr_t, modelH C.uintptr_t, ctData *C.char, ctDataLen C.ulong, numCTs C.int, outLen *C.ulong, errOut **C.char) *C.char {
	defer catchPanic(errOut)

	eval := cgo.Handle(evalH).Value().(*evaluator.Evaluator)
	model := cgo.Handle(modelH).Value().(*evaluator.Model)

	// Parse length-prefixed CT blobs.
	goCtData := cBytesToGoSlice(ctData, ctDataLen)
	n := int(numCTs)
	inputs := make([]*rlwe.Ciphertext, n)
	offset := 0
	for i := 0; i < n; i++ {
		if offset+8 > len(goCtData) {
			setErr(errOut, "CT data too short for length prefix")
			return nil
		}
		ctLen := int(binary.LittleEndian.Uint64(goCtData[offset : offset+8]))
		offset += 8
		if offset+ctLen > len(goCtData) {
			setErr(errOut, "CT data too short for CT body")
			return nil
		}
		ct := new(rlwe.Ciphertext)
		if err := ct.UnmarshalBinary(goCtData[offset : offset+ctLen]); err != nil {
			setErr(errOut, "unmarshalling input ciphertext: "+err.Error())
			return nil
		}
		inputs[i] = ct
		offset += ctLen
	}

	// Run Forward.
	results, err := eval.Forward(model, inputs)
	if err != nil {
		setErr(errOut, "forward: "+err.Error())
		return nil
	}

	// Marshal output CTs in length-prefixed format.
	var outBuf []byte
	for _, ct := range results {
		data, err := ct.MarshalBinary()
		if err != nil {
			setErr(errOut, "marshalling output ciphertext: "+err.Error())
			return nil
		}
		lenBuf := make([]byte, 8)
		binary.LittleEndian.PutUint64(lenBuf, uint64(len(data)))
		outBuf = append(outBuf, lenBuf...)
		outBuf = append(outBuf, data...)
	}

	ptr, length := goSliceToCBytes(outBuf)
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
