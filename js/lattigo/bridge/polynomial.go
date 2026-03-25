//go:build js && wasm

package main

import (
	"fmt"
	"syscall/js"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/minimax"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// newPolynomialMonomial(coeffs: Float64Array | number[]) → {handle: number} | {error: string}
func newPolynomialMonomial(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("newPolynomialMonomial: missing coeffs argument")
	}
	coeffs := jsToFloat64Slice(args[0])
	poly := bignum.NewPolynomial(bignum.Monomial, coeffs, nil)
	return handleResult(Store(&poly))
}

// newPolynomialChebyshev(coeffs, intervalA, intervalB) → {handle: number} | {error: string}
func newPolynomialChebyshev(_ js.Value, args []js.Value) any {
	if len(args) < 3 {
		return errorResult("newPolynomialChebyshev: need coeffs, intervalA, intervalB")
	}
	coeffs := jsToFloat64Slice(args[0])
	intervalA := args[1].Float()
	intervalB := args[2].Float()
	poly := bignum.NewPolynomial(
		bignum.Chebyshev, coeffs, [2]float64{intervalA, intervalB},
	)
	return handleResult(Store(&poly))
}

// genMinimaxCompositePolynomial(prec, logAlpha, logErr, degrees, debug)
// → {coeffs: Float64Array, seps: number[]} | {error: string}
// Returns raw Lattigo output — no caching, no sign→[0,1] rescaling.
func genMinimaxCompositePolynomial(_ js.Value, args []js.Value) any {
	if len(args) < 5 {
		return errorResult("genMinimaxCompositePolynomial: need prec, logAlpha, logErr, degrees, debug")
	}
	prec := uint(args[0].Int())
	logAlpha := args[1].Int()
	logErr := args[2].Int()
	degrees := jsToIntSlice(args[3])
	debug := args[4].Int() != 0

	defer func() {
		if r := recover(); r != nil {
			// Can't return error from deferred panic in WASM easily;
			// the function already returned errorResult below when possible.
			fmt.Printf("genMinimaxCompositePolynomial panic: %v\n", r)
		}
	}()

	coeffsBig := minimax.GenMinimaxCompositePolynomial(
		prec, logAlpha, logErr, degrees, bignum.Sign, debug,
	)

	// Count total and build separators
	total := 0
	nPolys := len(coeffsBig)
	seps := make([]int, nPolys)
	for i, poly := range coeffsBig {
		seps[i] = total
		total += len(poly)
	}

	// Convert to float64
	flat := make([]float64, total)
	idx := 0
	for _, poly := range coeffsBig {
		for _, c := range poly {
			f, _ := c.Float64()
			flat[idx] = f
			idx++
		}
	}

	result := js.Global().Get("Object").New()
	result.Set("coeffs", float64SliceToJS(flat))
	result.Set("seps", intSliceToJS(seps))
	return result
}
