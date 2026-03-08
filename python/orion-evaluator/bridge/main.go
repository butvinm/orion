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

func catchPanic(errOut **C.char) {
	if r := recover(); r != nil {
		setErr(errOut, fmt.Sprintf("panic: %v", r))
	}
}

func logPanic() {
	if r := recover(); r != nil {
		fmt.Fprintf(os.Stderr, "orion-evaluator bridge panic: %v\n", r)
	}
}

func setErr(errOut **C.char, msg string) {
	if errOut != nil {
		*errOut = C.CString(msg)
	}
}

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
