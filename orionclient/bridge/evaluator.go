package main

//#include <stdlib.h>
//#include <stdint.h>
import "C"

import (
	"runtime/cgo"
	"unsafe"

	orionclient "github.com/baahl-nyu/orion/orionclient"
)

// NewEvaluator creates an Evaluator from JSON-encoded params and an EvalKeyBundle handle.
//
//export NewEvaluator
func NewEvaluator(paramsJSON *C.char, keysH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	params, err := parseParams(C.GoString(paramsJSON))
	if err != nil {
		setErr(errOut, "parsing params: "+err.Error())
		return 0
	}
	keys := cgo.Handle(keysH).Value().(*orionclient.EvalKeyBundle)
	eval, err := orionclient.NewEvaluator(params, *keys)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(eval))
}

//export EvaluatorClose
func EvaluatorClose(evalH C.uintptr_t) {
	defer logPanic()
	h := cgo.Handle(evalH)
	eval := h.Value().(*orionclient.Evaluator)
	eval.Close()
	// NOTE: Do NOT call h.Delete() here. EvaluatorClose only does resource cleanup.
	// The Python GoHandle.close() calls DeleteHandle separately to free the cgo
	// handle slot (two-step close pattern).
}

//export EvalEncode
func EvalEncode(evalH C.uintptr_t, values *C.double, numValues C.int, level C.int, scale C.ulonglong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	goValues := cDoublesToGoFloat64s(values, numValues)
	pt, err := eval.Encode(goValues, int(level), uint64(scale))
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(pt))
}

// =========================================================================
// Ciphertext-Ciphertext arithmetic
// =========================================================================

//export EvalAdd
func EvalAdd(evalH C.uintptr_t, ct0H C.uintptr_t, ct1H C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct0 := cgo.Handle(ct0H).Value().(*orionclient.Ciphertext)
	ct1 := cgo.Handle(ct1H).Value().(*orionclient.Ciphertext)
	res, err := eval.Add(ct0, ct1)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

//export EvalSub
func EvalSub(evalH C.uintptr_t, ct0H C.uintptr_t, ct1H C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct0 := cgo.Handle(ct0H).Value().(*orionclient.Ciphertext)
	ct1 := cgo.Handle(ct1H).Value().(*orionclient.Ciphertext)
	res, err := eval.Sub(ct0, ct1)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

//export EvalMul
func EvalMul(evalH C.uintptr_t, ct0H C.uintptr_t, ct1H C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct0 := cgo.Handle(ct0H).Value().(*orionclient.Ciphertext)
	ct1 := cgo.Handle(ct1H).Value().(*orionclient.Ciphertext)
	res, err := eval.Mul(ct0, ct1)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

// =========================================================================
// Ciphertext-Plaintext arithmetic
// =========================================================================

//export EvalAddPlaintext
func EvalAddPlaintext(evalH C.uintptr_t, ctH C.uintptr_t, ptH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	res, err := eval.AddPlaintext(ct, pt)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

//export EvalSubPlaintext
func EvalSubPlaintext(evalH C.uintptr_t, ctH C.uintptr_t, ptH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	res, err := eval.SubPlaintext(ct, pt)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

//export EvalMulPlaintext
func EvalMulPlaintext(evalH C.uintptr_t, ctH C.uintptr_t, ptH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	res, err := eval.MulPlaintext(ct, pt)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

// =========================================================================
// Scalar arithmetic
// =========================================================================

//export EvalAddScalar
func EvalAddScalar(evalH C.uintptr_t, ctH C.uintptr_t, scalar C.double, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	res, err := eval.AddScalar(ct, float64(scalar))
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

//export EvalMulScalar
func EvalMulScalar(evalH C.uintptr_t, ctH C.uintptr_t, scalar C.double, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	res, err := eval.MulScalar(ct, float64(scalar))
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

//export EvalNegate
func EvalNegate(evalH C.uintptr_t, ctH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	res, err := eval.Negate(ct)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

// =========================================================================
// Rotation and rescaling
// =========================================================================

//export EvalRotate
func EvalRotate(evalH C.uintptr_t, ctH C.uintptr_t, amount C.int, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	res, err := eval.Rotate(ct, int(amount))
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

//export EvalRescale
func EvalRescale(evalH C.uintptr_t, ctH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	res, err := eval.Rescale(ct)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

// =========================================================================
// Polynomial evaluation
// =========================================================================

//export EvalPoly
func EvalPoly(evalH C.uintptr_t, ctH C.uintptr_t, polyH C.uintptr_t, outScale C.ulonglong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	poly := cgo.Handle(polyH).Value().(*orionclient.Polynomial)
	res, err := eval.EvalPoly(ct, poly.Raw(), uint64(outScale))
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

// =========================================================================
// Linear transform evaluation
// =========================================================================

//export EvalLinearTransform
func EvalLinearTransform(evalH C.uintptr_t, ctH C.uintptr_t, ltH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	lt := cgo.Handle(ltH).Value().(*orionclient.LinearTransform)
	res, err := eval.EvalLinearTransform(ct, lt)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

// =========================================================================
// Bootstrap
// =========================================================================

//export EvalBootstrap
func EvalBootstrap(evalH C.uintptr_t, ctH C.uintptr_t, numSlots C.int, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	res, err := eval.Bootstrap(ct, int(numSlots))
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(res))
}

// =========================================================================
// Evaluator queries
// =========================================================================

//export EvalMaxSlots
func EvalMaxSlots(evalH C.uintptr_t) C.int {
	defer logPanic()
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	return C.int(eval.MaxSlots())
}

//export EvalGaloisElement
func EvalGaloisElement(evalH C.uintptr_t, rotation C.int) C.ulonglong {
	defer logPanic()
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	return C.ulonglong(eval.GaloisElement(int(rotation)))
}

//export EvalModuliChain
func EvalModuliChain(evalH C.uintptr_t, outLen *C.int) *C.ulonglong {
	defer logPanic()
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	chain := eval.ModuliChain()
	n := len(chain)
	if n == 0 {
		*outLen = 0
		return nil
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.ulonglong(0)))
	ptr := (*C.ulonglong)(C.malloc(size))
	slice := unsafe.Slice(ptr, n)
	for i, v := range chain {
		slice[i] = C.ulonglong(v)
	}
	*outLen = C.int(n)
	return ptr
}

//export EvalDefaultScale
func EvalDefaultScale(evalH C.uintptr_t) C.ulonglong {
	defer logPanic()
	eval := cgo.Handle(evalH).Value().(*orionclient.Evaluator)
	return C.ulonglong(eval.DefaultScale())
}
