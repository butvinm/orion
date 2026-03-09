package main

//#include <stdlib.h>
//#include <stdint.h>
import "C"

import (
	"runtime/cgo"
	"unsafe"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/minimax"
	"github.com/baahl-nyu/lattigo/v6/utils/bignum"
)

// =========================================================================
// Polynomial type operations
// =========================================================================

//export NewPolynomialMonomial
func NewPolynomialMonomial(coeffs *C.double, n C.int, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goCoeffs := cDoublesToGoFloat64s(coeffs, n)
	poly := bignum.NewPolynomial(bignum.Monomial, goCoeffs, nil)
	return C.uintptr_t(cgo.NewHandle(&poly))
}

//export NewPolynomialChebyshev
func NewPolynomialChebyshev(
	coeffs *C.double, n C.int,
	intervalA C.double, intervalB C.double,
	errOut **C.char,
) C.uintptr_t {
	defer catchPanic(errOut)
	goCoeffs := cDoublesToGoFloat64s(coeffs, n)
	poly := bignum.NewPolynomial(
		bignum.Chebyshev, goCoeffs, [2]float64{float64(intervalA), float64(intervalB)},
	)
	return C.uintptr_t(cgo.NewHandle(&poly))
}

// =========================================================================
// Minimax composite polynomial (raw Lattigo, no caching or rescaling)
// =========================================================================

//export GenMinimaxCompositePolynomial
func GenMinimaxCompositePolynomial(
	prec C.uint,
	logAlpha C.int, logErr C.int,
	degreesPtr *C.int, numDegrees C.int,
	debug C.int,
	outCoeffs **C.double, outLen *C.int,
	outSeps **C.int, outNumPolys *C.int,
	errOut **C.char,
) {
	defer catchPanic(errOut)
	degrees := cIntsToGoInts(degreesPtr, numDegrees)

	coeffsBig := minimax.GenMinimaxCompositePolynomial(
		uint(prec), int(logAlpha), int(logErr), degrees, bignum.Sign, int(debug) != 0,
	)

	// Count total coefficients and build separator indices
	total := 0
	nPolys := len(coeffsBig)
	seps := make([]int, nPolys)
	for i, poly := range coeffsBig {
		seps[i] = total
		total += len(poly)
	}

	// Convert big.Float coefficients to float64
	flat := make([]float64, total)
	idx := 0
	for _, poly := range coeffsBig {
		for _, c := range poly {
			f, _ := c.Float64()
			flat[idx] = f
			idx++
		}
	}

	// Copy flat coefficients to C memory
	coeffSize := C.size_t(total) * C.size_t(unsafe.Sizeof(C.double(0)))
	coeffPtr := (*C.double)(C.malloc(coeffSize))
	coeffSlice := unsafe.Slice(coeffPtr, total)
	for i, v := range flat {
		coeffSlice[i] = C.double(v)
	}
	*outCoeffs = coeffPtr
	*outLen = C.int(total)

	// Copy separator indices to C memory
	sepsSize := C.size_t(nPolys) * C.size_t(unsafe.Sizeof(C.int(0)))
	sepsPtr := (*C.int)(C.malloc(sepsSize))
	sepsSlice := unsafe.Slice(sepsPtr, nPolys)
	for i, v := range seps {
		sepsSlice[i] = C.int(v)
	}
	*outSeps = sepsPtr
	*outNumPolys = C.int(nPolys)
}

