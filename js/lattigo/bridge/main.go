//go:build js && wasm

package main

import "syscall/js"

func main() {
	ns := js.Global().Get("Object").New()

	// deleteHandle(handleID) — frees a Go-side handle. Idempotent.
	ns.Set("deleteHandle", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) < 1 {
			return nil
		}
		id := uint32(args[0].Int())
		Delete(id)
		return nil
	}))

	// Readiness signal — MUST be last registration.
	ns.Set("__ready", true)

	js.Global().Set("lattigo", ns)

	// Keep Go runtime alive.
	select {}
}
