//go:build js && wasm

package main

import (
	"fmt"
	"syscall/js"

	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

// newCKKSParams(logN, logQ, logP, logDefaultScale, ringType, h?, logNthRoot?)
// → {handle: number} | {error: string}
func newCKKSParams(_ js.Value, args []js.Value) any {
	if len(args) < 5 {
		return errorResult("newCKKSParams: need logN, logQ, logP, logDefaultScale, ringType")
	}

	logN := args[0].Int()
	logQ := jsToIntSlice(args[1])
	logP := jsToIntSlice(args[2])
	logDefaultScale := args[3].Int()
	ringTypeStr := args[4].String()

	h := 0
	if len(args) > 5 && !args[5].IsUndefined() && !args[5].IsNull() {
		h = args[5].Int()
	}
	logNthRoot := 0
	if len(args) > 6 && !args[6].IsUndefined() && !args[6].IsNull() {
		logNthRoot = args[6].Int()
	}

	rt := ring.ConjugateInvariant
	switch ringTypeStr {
	case "Standard", "standard":
		rt = ring.Standard
	case "ConjugateInvariant", "conjugate_invariant", "conjugateinvariant", "":
		rt = ring.ConjugateInvariant
	default:
		return errorResult(fmt.Sprintf("newCKKSParams: unknown ring type: %q", ringTypeStr))
	}

	lit := ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            logQ,
		LogP:            logP,
		LogDefaultScale: logDefaultScale,
		RingType:        rt,
	}
	if h > 0 {
		lit.Xs = ring.Ternary{H: h}
	}
	if logNthRoot > 0 {
		lit.LogNthRoot = logNthRoot
	}

	params, err := ckks.NewParametersFromLiteral(lit)
	if err != nil {
		return errorResult(fmt.Sprintf("newCKKSParams: %v", err))
	}

	hID := Store(&params)
	return handleResult(hID)
}

// ckksMaxSlots(paramsHID: number) → number
func ckksMaxSlots(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return nil
	}
	p, ok := Load(uint32(args[0].Int()))
	if !ok {
		return nil
	}
	return p.(*ckks.Parameters).MaxSlots()
}

// ckksMaxLevel(paramsHID: number) → number
func ckksMaxLevel(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return nil
	}
	p, ok := Load(uint32(args[0].Int()))
	if !ok {
		return nil
	}
	return p.(*ckks.Parameters).MaxLevel()
}

// ckksDefaultScale(paramsHID: number) → number (float64 representation of scale)
func ckksDefaultScale(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return nil
	}
	p, ok := Load(uint32(args[0].Int()))
	if !ok {
		return nil
	}
	scale := p.(*ckks.Parameters).DefaultScale()
	f, _ := scale.Value.Float64()
	return f
}

// ckksGaloisElement(paramsHID: number, rotation: number) → number
func ckksGaloisElement(_ js.Value, args []js.Value) any {
	if len(args) < 2 {
		return nil
	}
	p, ok := Load(uint32(args[0].Int()))
	if !ok {
		return nil
	}
	rotation := args[1].Int()
	return float64(p.(*ckks.Parameters).GaloisElement(rotation))
}

// ckksModuliChain(paramsHID: number) → Array<number>
func ckksModuliChain(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return nil
	}
	p, ok := Load(uint32(args[0].Int()))
	if !ok {
		return nil
	}
	qi := p.(*ckks.Parameters).Q()
	arr := js.Global().Get("Array").New(len(qi))
	for i, v := range qi {
		arr.SetIndex(i, float64(v))
	}
	return arr
}

// ckksAuxModuliChain(paramsHID: number) → Array<number>
func ckksAuxModuliChain(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return nil
	}
	p, ok := Load(uint32(args[0].Int()))
	if !ok {
		return nil
	}
	pi := p.(*ckks.Parameters).P()
	arr := js.Global().Get("Array").New(len(pi))
	for i, v := range pi {
		arr.SetIndex(i, float64(v))
	}
	return arr
}

// errorResult returns a JS object {error: msg}.
func errorResult(msg string) any {
	obj := js.Global().Get("Object").New()
	obj.Set("error", msg)
	return obj
}

// handleResult returns a JS object {handle: id}.
func handleResult(id uint32) any {
	obj := js.Global().Get("Object").New()
	obj.Set("handle", int(id))
	return obj
}
