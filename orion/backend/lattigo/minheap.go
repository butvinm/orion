package main

import (
	"container/heap"
	"fmt"
)

type MinHeap []int

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MinHeap) Push(x interface{}) {
	*h = append(*h, x.(int))
}

func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// HeapAllocator updated to store pointers
type HeapAllocator struct {
	nextInt       int                  // The next integer to allocate
	freedIntegers MinHeap              // Min-heap to store freed integers
	InterfaceMap  map[int]*interface{} // Map to store/retrieve pointers to structs
}

// NewHeapAllocator initializes and returns a new HeapAllocator.
func NewHeapAllocator() *HeapAllocator {
	allocator := &HeapAllocator{
		nextInt:       0,
		freedIntegers: MinHeap{},
		InterfaceMap:  make(map[int]*interface{}),
	}
	heap.Init(&allocator.freedIntegers)
	return allocator
}

// Add assigns the lowest available integer to the provided object and
// returns the integer. Now ensures we're storing a pointer.
func (ha *HeapAllocator) Add(obj interface{}) int {
	var allocated int
	if len(ha.freedIntegers) > 0 {
		// Reuse the smallest available integer from the heap
		allocated = heap.Pop(&ha.freedIntegers).(int)
	} else {
		// Allocate a new integer
		allocated = ha.nextInt
		ha.nextInt++
	}

	// Create a pointer to the interface value
	objCopy := obj
	objPtr := &objCopy

	// Store the pointer in the map
	ha.InterfaceMap[allocated] = objPtr
	return allocated
}

// Retrieve returns the associated object with integer.
func (ha *HeapAllocator) Retrieve(integer int) interface{} {
	if objPtr, exists := ha.InterfaceMap[integer]; exists {
		// Dereference the pointer to get the original interface value
		return *objPtr
	}
	panic(fmt.Sprintf("Heap object not found for integer: %d", integer))
}

// Delete removes the integer and its associated object from the allocator
// and adds the integer back to the pool of available integers.
func (ha *HeapAllocator) Delete(integer int) {
	if _, exists := ha.InterfaceMap[integer]; exists {
		heap.Push(&ha.freedIntegers, integer)
		delete(ha.InterfaceMap, integer)
	}
}

// Reset clears the allocator's state, reinitializing its fields.
func (ha *HeapAllocator) Reset() {
	ha.nextInt = 0
	ha.freedIntegers = MinHeap{} // Reinitialize the slice
	heap.Init(&ha.freedIntegers) // Reinitialize the heap properties
	ha.InterfaceMap = make(map[int]*interface{})
}

func (ha *HeapAllocator) GetLiveKeys() []int {
	keys := make([]int, 0, len(ha.InterfaceMap))
	for k := range ha.InterfaceMap {
		keys = append(keys, k)
	}
	return keys
}
