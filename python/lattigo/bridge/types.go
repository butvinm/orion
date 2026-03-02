package main

//#include <stdlib.h>
//#include <stdint.h>
import "C"

import (
	"runtime/cgo"

	orion "github.com/baahl-nyu/orion"
)

// =========================================================================
// Polynomial type operations
// =========================================================================

//export GeneratePolynomialMonomial
func GeneratePolynomialMonomial(coeffs *C.double, numCoeffs C.int, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goCoeffs := cDoublesToGoFloat64s(coeffs, numCoeffs)
	poly := orion.GenerateMonomial(goCoeffs)
	return C.uintptr_t(cgo.NewHandle(poly))
}

//export GeneratePolynomialChebyshev
func GeneratePolynomialChebyshev(coeffs *C.double, numCoeffs C.int, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goCoeffs := cDoublesToGoFloat64s(coeffs, numCoeffs)
	poly := orion.GenerateChebyshev(goCoeffs)
	return C.uintptr_t(cgo.NewHandle(poly))
}

// =========================================================================
// Minimax sign coefficients
// =========================================================================

//export GenerateMinimaxSignCoeffs
func GenerateMinimaxSignCoeffs(
	degreesPtr *C.int, numDegrees C.int,
	prec C.int,
	logAlpha C.int,
	logErr C.int,
	debug C.int,
	outLen *C.int,
	errOut **C.char,
) *C.double {
	defer catchPanic(errOut)
	degrees := cIntsToGoInts(degreesPtr, numDegrees)
	flat, err := orion.GenerateMinimaxSignCoeffs(degrees, uint(prec), int(logAlpha), int(logErr), int(debug) != 0)
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goFloat64sToCDoubles(flat)
	*outLen = length
	return ptr
}
