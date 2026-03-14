//go:build js && wasm

package main

import "testing"

func TestStoreAndLoad(t *testing.T) {
	// Reset state for test isolation.
	handleMap = make(map[uint32]any)
	nextHandle = 1

	id := Store("hello")
	if id == 0 {
		t.Fatal("expected non-zero handle ID")
	}

	obj, ok := Load(id)
	if !ok {
		t.Fatalf("Load(%d) returned ok=false", id)
	}
	if obj.(string) != "hello" {
		t.Fatalf("Load(%d) = %v, want %q", id, obj, "hello")
	}
}

func TestStoreIncrementsID(t *testing.T) {
	handleMap = make(map[uint32]any)
	nextHandle = 1

	id1 := Store("a")
	id2 := Store("b")
	id3 := Store("c")

	if id1 >= id2 || id2 >= id3 {
		t.Fatalf("IDs not strictly increasing: %d, %d, %d", id1, id2, id3)
	}
}

func TestDelete(t *testing.T) {
	handleMap = make(map[uint32]any)
	nextHandle = 1

	id := Store("to-delete")

	Delete(id)

	_, ok := Load(id)
	if ok {
		t.Fatalf("Load(%d) returned ok=true after Delete", id)
	}
}

func TestDeleteIdempotent(t *testing.T) {
	handleMap = make(map[uint32]any)
	nextHandle = 1

	id := Store("once")
	Delete(id)
	// Second delete should be a no-op, not panic.
	Delete(id)

	_, ok := Load(id)
	if ok {
		t.Fatalf("Load(%d) returned ok=true after double Delete", id)
	}
}

func TestDeleteNonExistent(t *testing.T) {
	handleMap = make(map[uint32]any)
	nextHandle = 1

	// Should not panic.
	Delete(99999)
}

func TestLoadNonExistent(t *testing.T) {
	handleMap = make(map[uint32]any)
	nextHandle = 1

	_, ok := Load(42)
	if ok {
		t.Fatal("Load(42) returned ok=true for non-existent handle")
	}
}

func TestStoreMultipleTypes(t *testing.T) {
	handleMap = make(map[uint32]any)
	nextHandle = 1

	idStr := Store("string-val")
	idInt := Store(12345)
	idSlice := Store([]float64{1.0, 2.0, 3.0})

	obj1, ok1 := Load(idStr)
	obj2, ok2 := Load(idInt)
	obj3, ok3 := Load(idSlice)

	if !ok1 || !ok2 || !ok3 {
		t.Fatal("one or more loads failed")
	}
	if obj1.(string) != "string-val" {
		t.Errorf("string: got %v", obj1)
	}
	if obj2.(int) != 12345 {
		t.Errorf("int: got %v", obj2)
	}
	s := obj3.([]float64)
	if len(s) != 3 || s[0] != 1.0 || s[1] != 2.0 || s[2] != 3.0 {
		t.Errorf("slice: got %v", s)
	}
}
