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

func cBytesToGoSlice(data *C.char, dataLen C.ulong) []byte {
	return C.GoBytes(unsafe.Pointer(data), C.int(dataLen))
}
