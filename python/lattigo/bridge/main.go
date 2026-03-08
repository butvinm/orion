package main

//#include <stdlib.h>
//#include <stdint.h>
import "C"

import (
	"fmt"
	"os"
	"runtime/cgo"
	"unsafe"
)

func main() {}

//export DeleteHandle
func DeleteHandle(h C.uintptr_t) {
	defer logPanic()
	cgo.Handle(h).Delete()
}

//export FreeCArray
func FreeCArray(ptr unsafe.Pointer) {
	C.free(ptr)
}

// --- Internal helpers ---

// catchPanic recovers from panics and reports them via errOut.
func catchPanic(errOut **C.char) {
	if r := recover(); r != nil {
		setErr(errOut, fmt.Sprintf("panic: %v", r))
	}
}

// logPanic recovers from panics and logs them to stderr.
// Use for bridge functions without errOut where panics must not cross the CGo boundary.
func logPanic() {
	if r := recover(); r != nil {
		fmt.Fprintf(os.Stderr, "orionclient bridge panic: %v\n", r)
	}
}

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
// Uses unsafe.Slice instead of C.GoBytes to support data larger than 2 GB
// (C.GoBytes takes C.int which is 32-bit, truncating lengths > 2^31).
func cBytesToGoSlice(data *C.char, dataLen C.ulong) []byte {
	n := int(dataLen)
	if n == 0 {
		return nil
	}
	src := unsafe.Slice((*byte)(unsafe.Pointer(data)), n)
	dst := make([]byte, n)
	copy(dst, src)
	return dst
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
