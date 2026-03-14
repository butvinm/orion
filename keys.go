package orion

// Manifest describes which evaluation keys a Client must generate.
// Mirrors Python's KeyManifest.
type Manifest struct {
	GaloisElements []uint64 `json:"galois_elements"`
	BootstrapSlots []int    `json:"bootstrap_slots"`
	BootLogP       []int    `json:"boot_logp,omitempty"`
	BtpLogN        int      `json:"btp_logn,omitempty"`
	NeedsRLK       bool     `json:"needs_rlk"`
}

