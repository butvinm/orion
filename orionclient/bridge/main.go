package main

//#include <stdlib.h>
//#include <stdint.h>
import "C"

import (
	"runtime/cgo"
	"unsafe"
)

func main() {}

//export DeleteHandle
func DeleteHandle(h C.uintptr_t) {
	cgo.Handle(h).Delete()
}

//export FreeCArray
func FreeCArray(ptr unsafe.Pointer) {
	C.free(ptr)
}

// --- Internal helpers ---

// setErr sets the error output string. Caller must free with FreeCArray.
func setErr(errOut **C.char, msg string) {
	if errOut != nil {
		*errOut = C.CString(msg)
	}
}

// goSliceToCBytes copies a Go byte slice to C-allocated memory.
// Caller must free with FreeCArray.
func goSliceToCBytes(data []byte) (*C.char, C.ulong) {
	if len(data) == 0 {
		return nil, 0
	}
	ptr := C.CBytes(data)
	return (*C.char)(ptr), C.ulong(len(data))
}

// cBytesToGoSlice copies C bytes to a Go byte slice.
func cBytesToGoSlice(data *C.char, dataLen C.ulong) []byte {
	return C.GoBytes(unsafe.Pointer(data), C.int(dataLen))
}

// goFloat64sToCDoubles copies a Go float64 slice to C-allocated memory.
// Caller must free with FreeCArray.
func goFloat64sToCDoubles(vals []float64) (*C.double, C.int) {
	n := len(vals)
	if n == 0 {
		return nil, 0
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.double(0)))
	ptr := (*C.double)(C.malloc(size))
	slice := unsafe.Slice(ptr, n)
	for i, v := range vals {
		slice[i] = C.double(v)
	}
	return ptr, C.int(n)
}

// cDoublesToGoFloat64s copies C doubles to a Go float64 slice.
func cDoublesToGoFloat64s(vals *C.double, numVals C.int) []float64 {
	n := int(numVals)
	if n == 0 {
		return nil
	}
	cSlice := unsafe.Slice(vals, n)
	result := make([]float64, n)
	for i, v := range cSlice {
		result[i] = float64(v)
	}
	return result
}

// cIntsToGoInts copies C ints to a Go int slice.
func cIntsToGoInts(vals *C.int, numVals C.int) []int {
	n := int(numVals)
	if n == 0 {
		return nil
	}
	cSlice := unsafe.Slice(vals, n)
	result := make([]int, n)
	for i, v := range cSlice {
		result[i] = int(v)
	}
	return result
}

// goIntsToCInts copies a Go int slice to C-allocated memory.
// Caller must free with FreeCArray.
func goIntsToCInts(vals []int) (*C.int, C.int) {
	n := len(vals)
	if n == 0 {
		return nil, 0
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.int(0)))
	ptr := (*C.int)(C.malloc(size))
	slice := unsafe.Slice(ptr, n)
	for i, v := range vals {
		slice[i] = C.int(v)
	}
	return ptr, C.int(n)
}
