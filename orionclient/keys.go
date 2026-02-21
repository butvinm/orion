package orionclient

// Manifest describes which evaluation keys a Client must generate.
// Mirrors Python's KeyManifest.
type Manifest struct {
	GaloisElements []uint64 `json:"galois_elements"`
	BootstrapSlots []int    `json:"bootstrap_slots"`
	BootLogP       []int    `json:"boot_logp,omitempty"`
	NeedsRLK       bool     `json:"needs_rlk"`
}

// EvalKeyBundle holds serialized evaluation keys for transport.
// The Client produces this; the Evaluator consumes it.
type EvalKeyBundle struct {
	RLK           []byte            // Serialized relinearization key (nil if not needed)
	GaloisKeys    map[uint64][]byte // galois_element -> serialized Galois key
	BootstrapKeys map[int][]byte    // slot_count -> serialized bootstrap keys
	BootLogP      []int             // Bootstrap auxiliary primes (per slot config)
}
