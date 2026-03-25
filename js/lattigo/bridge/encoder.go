//go:build js && wasm

package main

import (
	"fmt"
	"syscall/js"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// encoderHandle bundles encoder with its parameters (needed to create plaintexts).
type encoderHandle struct {
	encoder *ckks.Encoder
	params  *ckks.Parameters
}

// newEncoder(paramsHID: number) → {handle: number} | {error: string}
func newEncoder(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("newEncoder: missing paramsHID argument")
	}
	p, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("newEncoder: invalid params handle")
	}
	params := p.(*ckks.Parameters)
	enc := ckks.NewEncoder(*params)
	return handleResult(Store(&encoderHandle{encoder: enc, params: params}))
}

// encoderEncode(encHID: number, values: Float64Array|Array, level: number, scale: number) → {handle: number} | {error: string}
func encoderEncode(_ js.Value, args []js.Value) any {
	if len(args) < 4 {
		return errorResult("encoderEncode: missing arguments (encHID, values, level, scale)")
	}

	encObj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("encoderEncode: invalid encoder handle")
	}
	eh := encObj.(*encoderHandle)

	values := jsToFloat64Slice(args[1])
	level := args[2].Int()
	scale := args[3].Float()

	pt := ckks.NewPlaintext(*eh.params, level)
	pt.Scale = rlwe.NewScale(scale)

	if err := eh.encoder.Encode(values, pt); err != nil {
		return errorResult(fmt.Sprintf("encoderEncode: %v", err))
	}
	return handleResult(Store(pt))
}

// encoderDecode(encHID: number, ptHID: number, numSlots: number) → Float64Array | {error: string}
func encoderDecode(_ js.Value, args []js.Value) any {
	if len(args) < 3 {
		return errorResult("encoderDecode: missing arguments (encHID, ptHID, numSlots)")
	}

	encObj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("encoderDecode: invalid encoder handle")
	}
	eh := encObj.(*encoderHandle)

	ptObj, ok := Load(uint32(args[1].Int()))
	if !ok {
		return errorResult("encoderDecode: invalid plaintext handle")
	}
	pt := ptObj.(*rlwe.Plaintext)

	numSlots := args[2].Int()
	result := make([]float64, numSlots)
	if err := eh.encoder.Decode(pt, result); err != nil {
		return errorResult(fmt.Sprintf("encoderDecode: %v", err))
	}
	return float64SliceToJS(result)
}
