//go:build js && wasm

package main

// Handle map for Go objects exposed to JavaScript.
// No mutex needed — Go WASM is single-threaded (goroutines are cooperative).

var (
	handleMap  = make(map[uint32]any)
	nextHandle uint32 = 1
)

// Store saves obj into the handle map and returns its ID.
func Store(obj any) uint32 {
	id := nextHandle
	nextHandle++
	handleMap[id] = obj
	return id
}

// Load retrieves the object for the given handle ID.
// Returns (nil, false) if the handle does not exist.
func Load(id uint32) (any, bool) {
	obj, ok := handleMap[id]
	return obj, ok
}

// Delete removes the handle. No-op if the handle does not exist (idempotent).
func Delete(id uint32) {
	delete(handleMap, id)
}
