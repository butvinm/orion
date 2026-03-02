package main

//#include <stdlib.h>
//#include <stdint.h>
import "C"

import (
	"runtime/cgo"
	"unsafe"

	orionclient "github.com/baahl-nyu/orion/orionclient"
)

// =========================================================================
// Ciphertext type operations
// =========================================================================

//export CiphertextMarshal
func CiphertextMarshal(ctH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
	defer catchPanic(errOut)
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	data, err := ct.Marshal()
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export CiphertextUnmarshal
func CiphertextUnmarshal(data *C.char, dataLen C.ulong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goData := cBytesToGoSlice(data, dataLen)
	ct, err := orionclient.UnmarshalCiphertext(goData)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(ct))
}

//export CiphertextLevel
func CiphertextLevel(ctH C.uintptr_t) C.int {
	defer logPanic()
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	return C.int(ct.Level())
}

//export CiphertextScale
func CiphertextScale(ctH C.uintptr_t) C.ulonglong {
	defer logPanic()
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	return C.ulonglong(ct.Scale())
}

//export CiphertextSetScale
func CiphertextSetScale(ctH C.uintptr_t, scale C.ulonglong) {
	defer logPanic()
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	ct.SetScale(uint64(scale))
}

//export CiphertextSlots
func CiphertextSlots(ctH C.uintptr_t) C.int {
	defer logPanic()
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	return C.int(ct.Slots())
}

//export CiphertextDegree
func CiphertextDegree(ctH C.uintptr_t) C.int {
	defer logPanic()
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	return C.int(ct.Degree())
}

//export CiphertextShape
func CiphertextShape(ctH C.uintptr_t, outLen *C.int) *C.int {
	defer logPanic()
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	shape := ct.Shape()
	ptr, length := goIntsToCInts(shape)
	*outLen = length
	return ptr
}

//export CiphertextNumCiphertexts
func CiphertextNumCiphertexts(ctH C.uintptr_t) C.int {
	defer logPanic()
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	return C.int(ct.NumCiphertexts())
}

// CombineSingleCiphertexts takes N single-ct Ciphertext handles and combines
// them into one multi-ct Ciphertext. The input handles are NOT deleted.
//
//export CombineSingleCiphertexts
func CombineSingleCiphertexts(handles *C.uintptr_t, numHandles C.int, shapeDims *C.int, numDims C.int, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	n := int(numHandles)
	if n == 0 {
		setErr(errOut, "no ciphertext handles provided")
		return 0
	}
	hSlice := unsafe.Slice(handles, n)
	cts := make([]*orionclient.Ciphertext, n)
	for i := range cts {
		cts[i] = cgo.Handle(hSlice[i]).Value().(*orionclient.Ciphertext)
	}

	shape := cIntsToGoInts(shapeDims, numDims)
	combined := orionclient.CombineCiphertexts(cts, shape)
	return C.uintptr_t(cgo.NewHandle(combined))
}

// =========================================================================
// Plaintext type operations
// =========================================================================

//export PlaintextLevel
func PlaintextLevel(ptH C.uintptr_t) C.int {
	defer logPanic()
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	return C.int(pt.Level())
}

//export PlaintextScale
func PlaintextScale(ptH C.uintptr_t) C.ulonglong {
	defer logPanic()
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	return C.ulonglong(pt.Scale())
}

//export PlaintextSetScale
func PlaintextSetScale(ptH C.uintptr_t, scale C.ulonglong) {
	defer logPanic()
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	pt.SetScale(uint64(scale))
}

//export PlaintextSlots
func PlaintextSlots(ptH C.uintptr_t) C.int {
	defer logPanic()
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	return C.int(pt.Slots())
}

//export PlaintextShape
func PlaintextShape(ptH C.uintptr_t, outLen *C.int) *C.int {
	defer logPanic()
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	shape := pt.Shape()
	ptr, length := goIntsToCInts(shape)
	*outLen = length
	return ptr
}


// =========================================================================
// Polynomial type operations
// =========================================================================

//export GeneratePolynomialMonomial
func GeneratePolynomialMonomial(coeffs *C.double, numCoeffs C.int, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goCoeffs := cDoublesToGoFloat64s(coeffs, numCoeffs)
	poly := orionclient.GenerateMonomial(goCoeffs)
	return C.uintptr_t(cgo.NewHandle(poly))
}

//export GeneratePolynomialChebyshev
func GeneratePolynomialChebyshev(coeffs *C.double, numCoeffs C.int, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goCoeffs := cDoublesToGoFloat64s(coeffs, numCoeffs)
	poly := orionclient.GenerateChebyshev(goCoeffs)
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
	flat, err := orionclient.GenerateMinimaxSignCoeffs(degrees, uint(prec), int(logAlpha), int(logErr), int(debug) != 0)
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goFloat64sToCDoubles(flat)
	*outLen = length
	return ptr
}
