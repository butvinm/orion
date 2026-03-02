//go:build js && wasm

package main

import (
	"encoding/json"
	"fmt"
	"syscall/js"

	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

// paramsJSON mirrors the JSON format used by Python bridge (orion.Params).
type paramsJSON struct {
	LogN     int    `json:"LogN"`
	LogQ     []int  `json:"LogQ"`
	LogP     []int  `json:"LogP"`
	LogScale int    `json:"LogDefaultScale"`
	H        int    `json:"H"`
	RingType string `json:"RingType"`
}

// newCKKSParams(paramsJSON: string) → {handle: number} | {error: string}
func newCKKSParams(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("newCKKSParams: missing paramsJSON argument")
	}
	jsonStr := args[0].String()

	var p paramsJSON
	if err := json.Unmarshal([]byte(jsonStr), &p); err != nil {
		return errorResult(fmt.Sprintf("newCKKSParams: parsing JSON: %v", err))
	}

	rt := ring.ConjugateInvariant
	switch p.RingType {
	case "Standard", "standard":
		rt = ring.Standard
	case "ConjugateInvariant", "conjugate_invariant", "conjugateinvariant", "":
		rt = ring.ConjugateInvariant
	default:
		return errorResult(fmt.Sprintf("newCKKSParams: unknown ring type: %q", p.RingType))
	}

	lit := ckks.ParametersLiteral{
		LogN:            p.LogN,
		LogQ:            p.LogQ,
		LogP:            p.LogP,
		LogDefaultScale: p.LogScale,
		RingType:        rt,
	}
	if p.H > 0 {
		lit.Xs = ring.Ternary{H: p.H}
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
	return int64(p.(*ckks.Parameters).GaloisElement(rotation))
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
		// JS numbers are float64 — safe for primes up to 2^53
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
