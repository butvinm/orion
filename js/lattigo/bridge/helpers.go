//go:build js && wasm

package main

import "syscall/js"

// jsToFloat64Slice converts a JS Array or TypedArray to a Go float64 slice.
func jsToFloat64Slice(v js.Value) []float64 {
	length := v.Length()
	result := make([]float64, length)
	for i := 0; i < length; i++ {
		result[i] = v.Index(i).Float()
	}
	return result
}

// float64SliceToJS converts a Go float64 slice to a JS Float64Array.
func float64SliceToJS(s []float64) js.Value {
	arr := js.Global().Get("Float64Array").New(len(s))
	for i, v := range s {
		arr.SetIndex(i, v)
	}
	return arr
}

// bytesToJS converts a Go byte slice to a JS Uint8Array.
func bytesToJS(b []byte) js.Value {
	arr := js.Global().Get("Uint8Array").New(len(b))
	js.CopyBytesToJS(arr, b)
	return arr
}

// jsToBytes converts a JS Uint8Array to a Go byte slice.
func jsToBytes(v js.Value) []byte {
	length := v.Length()
	b := make([]byte, length)
	js.CopyBytesToGo(b, v)
	return b
}
