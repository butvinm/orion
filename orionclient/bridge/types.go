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
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	return C.int(ct.Level())
}

//export CiphertextScale
func CiphertextScale(ctH C.uintptr_t) C.ulonglong {
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	return C.ulonglong(ct.Scale())
}

//export CiphertextSetScale
func CiphertextSetScale(ctH C.uintptr_t, scale C.ulonglong) {
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	ct.SetScale(uint64(scale))
}

//export CiphertextSlots
func CiphertextSlots(ctH C.uintptr_t) C.int {
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	return C.int(ct.Slots())
}

//export CiphertextDegree
func CiphertextDegree(ctH C.uintptr_t) C.int {
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	return C.int(ct.Degree())
}

//export CiphertextShape
func CiphertextShape(ctH C.uintptr_t, outLen *C.int) *C.int {
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	shape := ct.Shape()
	ptr, length := goIntsToCInts(shape)
	*outLen = length
	return ptr
}

//export CiphertextNumCiphertexts
func CiphertextNumCiphertexts(ctH C.uintptr_t) C.int {
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	return C.int(ct.NumCiphertexts())
}

// =========================================================================
// Plaintext type operations
// =========================================================================

//export PlaintextLevel
func PlaintextLevel(ptH C.uintptr_t) C.int {
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	return C.int(pt.Level())
}

//export PlaintextScale
func PlaintextScale(ptH C.uintptr_t) C.ulonglong {
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	return C.ulonglong(pt.Scale())
}

//export PlaintextSetScale
func PlaintextSetScale(ptH C.uintptr_t, scale C.ulonglong) {
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	pt.SetScale(uint64(scale))
}

//export PlaintextSlots
func PlaintextSlots(ptH C.uintptr_t) C.int {
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	return C.int(pt.Slots())
}

//export PlaintextShape
func PlaintextShape(ptH C.uintptr_t, outLen *C.int) *C.int {
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	shape := pt.Shape()
	ptr, length := goIntsToCInts(shape)
	*outLen = length
	return ptr
}

// =========================================================================
// EvalKeyBundle operations
// =========================================================================

//export NewEvalKeyBundle
func NewEvalKeyBundle() C.uintptr_t {
	bundle := &orionclient.EvalKeyBundle{
		GaloisKeys:    make(map[uint64][]byte),
		BootstrapKeys: make(map[int][]byte),
	}
	return C.uintptr_t(cgo.NewHandle(bundle))
}

//export EvalKeyBundleSetRLK
func EvalKeyBundleSetRLK(keysH C.uintptr_t, data *C.char, dataLen C.ulong) {
	bundle := cgo.Handle(keysH).Value().(*orionclient.EvalKeyBundle)
	bundle.RLK = cBytesToGoSlice(data, dataLen)
}

//export EvalKeyBundleAddGaloisKey
func EvalKeyBundleAddGaloisKey(keysH C.uintptr_t, galEl C.ulonglong, data *C.char, dataLen C.ulong) {
	bundle := cgo.Handle(keysH).Value().(*orionclient.EvalKeyBundle)
	bundle.GaloisKeys[uint64(galEl)] = cBytesToGoSlice(data, dataLen)
}

//export EvalKeyBundleAddBootstrapKey
func EvalKeyBundleAddBootstrapKey(keysH C.uintptr_t, slots C.int, data *C.char, dataLen C.ulong) {
	bundle := cgo.Handle(keysH).Value().(*orionclient.EvalKeyBundle)
	bundle.BootstrapKeys[int(slots)] = cBytesToGoSlice(data, dataLen)
}

//export EvalKeyBundleSetBootLogP
func EvalKeyBundleSetBootLogP(keysH C.uintptr_t, logP *C.int, logPLen C.int) {
	bundle := cgo.Handle(keysH).Value().(*orionclient.EvalKeyBundle)
	bundle.BootLogP = cIntsToGoInts(logP, logPLen)
}

//export EvalKeyBundleGetRLK
func EvalKeyBundleGetRLK(keysH C.uintptr_t, outLen *C.ulong) *C.char {
	bundle := cgo.Handle(keysH).Value().(*orionclient.EvalKeyBundle)
	if bundle.RLK == nil {
		*outLen = 0
		return nil
	}
	ptr, length := goSliceToCBytes(bundle.RLK)
	*outLen = length
	return ptr
}

//export EvalKeyBundleGetGaloisKeyElements
func EvalKeyBundleGetGaloisKeyElements(keysH C.uintptr_t, outLen *C.int) *C.ulonglong {
	bundle := cgo.Handle(keysH).Value().(*orionclient.EvalKeyBundle)
	n := len(bundle.GaloisKeys)
	if n == 0 {
		*outLen = 0
		return nil
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.ulonglong(0)))
	ptr := (*C.ulonglong)(C.malloc(size))
	slice := unsafe.Slice(ptr, n)
	i := 0
	for el := range bundle.GaloisKeys {
		slice[i] = C.ulonglong(el)
		i++
	}
	*outLen = C.int(n)
	return ptr
}

//export EvalKeyBundleGetGaloisKey
func EvalKeyBundleGetGaloisKey(keysH C.uintptr_t, galEl C.ulonglong, outLen *C.ulong) *C.char {
	bundle := cgo.Handle(keysH).Value().(*orionclient.EvalKeyBundle)
	data, ok := bundle.GaloisKeys[uint64(galEl)]
	if !ok {
		*outLen = 0
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export EvalKeyBundleGetBootstrapSlots
func EvalKeyBundleGetBootstrapSlots(keysH C.uintptr_t, outLen *C.int) *C.int {
	bundle := cgo.Handle(keysH).Value().(*orionclient.EvalKeyBundle)
	n := len(bundle.BootstrapKeys)
	if n == 0 {
		*outLen = 0
		return nil
	}
	slots := make([]int, 0, n)
	for s := range bundle.BootstrapKeys {
		slots = append(slots, s)
	}
	ptr, length := goIntsToCInts(slots)
	*outLen = length
	return ptr
}

//export EvalKeyBundleGetBootstrapKey
func EvalKeyBundleGetBootstrapKey(keysH C.uintptr_t, slots C.int, outLen *C.ulong) *C.char {
	bundle := cgo.Handle(keysH).Value().(*orionclient.EvalKeyBundle)
	data, ok := bundle.BootstrapKeys[int(slots)]
	if !ok {
		*outLen = 0
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export EvalKeyBundleGetBootLogP
func EvalKeyBundleGetBootLogP(keysH C.uintptr_t, outLen *C.int) *C.int {
	bundle := cgo.Handle(keysH).Value().(*orionclient.EvalKeyBundle)
	ptr, length := goIntsToCInts(bundle.BootLogP)
	*outLen = length
	return ptr
}

// =========================================================================
// LinearTransform type operations
// =========================================================================

//export LinearTransformMarshal
func LinearTransformMarshal(ltH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
	lt := cgo.Handle(ltH).Value().(*orionclient.LinearTransform)
	data, err := lt.Marshal()
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export LinearTransformUnmarshal
func LinearTransformUnmarshal(data *C.char, dataLen C.ulong, errOut **C.char) C.uintptr_t {
	goData := cBytesToGoSlice(data, dataLen)
	lt, err := orionclient.LoadLinearTransform(goData)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(lt))
}

//export LinearTransformRequiredGaloisElements
func LinearTransformRequiredGaloisElements(ltH C.uintptr_t, paramsJSON *C.char, outLen *C.int, errOut **C.char) *C.ulonglong {
	lt := cgo.Handle(ltH).Value().(*orionclient.LinearTransform)
	params, err := parseParams(C.GoString(paramsJSON))
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	elements, err := lt.RequiredGaloisElements(params)
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	n := len(elements)
	if n == 0 {
		*outLen = 0
		return nil
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.ulonglong(0)))
	ptr := (*C.ulonglong)(C.malloc(size))
	slice := unsafe.Slice(ptr, n)
	for i, el := range elements {
		slice[i] = C.ulonglong(el)
	}
	*outLen = C.int(n)
	return ptr
}

// =========================================================================
// Polynomial type operations
// =========================================================================

//export GeneratePolynomialMonomial
func GeneratePolynomialMonomial(coeffs *C.double, numCoeffs C.int) C.uintptr_t {
	goCoeffs := cDoublesToGoFloat64s(coeffs, numCoeffs)
	poly := orionclient.GenerateMonomial(goCoeffs)
	return C.uintptr_t(cgo.NewHandle(poly))
}

//export GeneratePolynomialChebyshev
func GeneratePolynomialChebyshev(coeffs *C.double, numCoeffs C.int) C.uintptr_t {
	goCoeffs := cDoublesToGoFloat64s(coeffs, numCoeffs)
	poly := orionclient.GenerateChebyshev(goCoeffs)
	return C.uintptr_t(cgo.NewHandle(poly))
}
