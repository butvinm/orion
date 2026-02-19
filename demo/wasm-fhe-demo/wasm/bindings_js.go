//go:build js && wasm

package main

import (
	"syscall/js"
)

// goBytesToJSUint8Array copies a Go byte slice into a JS Uint8Array.
func goBytesToJSUint8Array(data []byte) js.Value {
	arr := js.Global().Get("Uint8Array").New(len(data))
	js.CopyBytesToJS(arr, data)
	return arr
}

// jsArrayToIntSlice converts a JS Array of numbers to a Go []int slice.
func jsArrayToIntSlice(arr js.Value) []int {
	length := arr.Get("length").Int()
	result := make([]int, length)
	for i := 0; i < length; i++ {
		result[i] = arr.Index(i).Int()
	}
	return result
}

// promisify wraps a blocking Go function as a JS Promise via a goroutine.
func promisify(fn func() (js.Value, error)) js.Value {
	handler := js.FuncOf(func(_ js.Value, pArgs []js.Value) interface{} {
		resolve, reject := pArgs[0], pArgs[1]
		go func() {
			result, err := fn()
			if err != nil {
				reject.Invoke(err.Error())
				return
			}
			resolve.Invoke(result)
		}()
		return nil
	})
	return js.Global().Get("Promise").New(handler)
}

// orionInit: (logN, logQ[], logP[], logScale, h, ringType) => Promise<void>
func orionInit(_ js.Value, args []js.Value) interface{} {
	logN := args[0].Int()
	logQ := jsArrayToIntSlice(args[1])
	logP := jsArrayToIntSlice(args[2])
	logScale := args[3].Int()
	h := args[4].Int()
	ringType := args[5].String()

	return promisify(func() (js.Value, error) {
		if err := InitScheme(logN, logQ, logP, logScale, h, ringType); err != nil {
			return js.Undefined(), err
		}
		return js.Undefined(), nil
	})
}

// orionGetMaxSlots: () => number
func orionGetMaxSlots(_ js.Value, _ []js.Value) interface{} {
	return GetMaxSlots()
}

// orionSerializeRelinKey: () => Promise<Uint8Array>
func orionSerializeRelinKey(_ js.Value, _ []js.Value) interface{} {
	return promisify(func() (js.Value, error) {
		data, err := SerializeRelinKey()
		if err != nil {
			return js.Undefined(), err
		}
		return goBytesToJSUint8Array(data), nil
	})
}

// orionGenerateAndSerializeGaloisKey: (galEl: number) => Promise<Uint8Array>
func orionGenerateAndSerializeGaloisKey(_ js.Value, args []js.Value) interface{} {
	galEl := uint64(args[0].Float())

	return promisify(func() (js.Value, error) {
		data, err := GenerateAndSerializeGaloisKey(galEl)
		if err != nil {
			return js.Undefined(), err
		}
		return goBytesToJSUint8Array(data), nil
	})
}

// orionSerializeBootstrapKeys: (numSlots: number, logP: number[]) => Promise<Uint8Array>
func orionSerializeBootstrapKeys(_ js.Value, args []js.Value) interface{} {
	numSlots := args[0].Int()
	logP := jsArrayToIntSlice(args[1])

	return promisify(func() (js.Value, error) {
		data, err := SerializeBootstrapKeys(numSlots, logP)
		if err != nil {
			return js.Undefined(), err
		}
		return goBytesToJSUint8Array(data), nil
	})
}

func main() {
	g := js.Global()
	g.Set("orionInit", js.FuncOf(orionInit))
	g.Set("orionGetMaxSlots", js.FuncOf(orionGetMaxSlots))
	g.Set("orionSerializeRelinKey", js.FuncOf(orionSerializeRelinKey))
	g.Set("orionGenerateAndSerializeGaloisKey", js.FuncOf(orionGenerateAndSerializeGaloisKey))
	g.Set("orionSerializeBootstrapKeys", js.FuncOf(orionSerializeBootstrapKeys))

	// Block forever so the Go runtime stays alive in the browser.
	select {}
}
