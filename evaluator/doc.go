// Package evaluator loads compiled Orion models (.orion v2 files) and runs
// FHE inference on ciphertexts by walking the computation graph.
//
// Model is loaded once via LoadModel and is immutable — safe to share
// across goroutines. Evaluator is created per-client via NewEvaluator
// and is NOT goroutine-safe (Lattigo buffers are reused internally).
//
// Supported ops: linear_transform, quad, polynomial, flatten, add, mult.
// Bootstrap is recognized but not yet implemented.
package evaluator
