package orion

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Params holds CKKS scheme parameters in a serialization-friendly form.
// Mirrors Python's CKKSParams dataclass.
type Params struct {
	LogN            int    `json:"logn"`
	LogQ            []int  `json:"logq"`
	LogP            []int  `json:"logp"`
	LogDefaultScale int    `json:"log_default_scale"`
	H               int    `json:"h"`
	RingType        string `json:"ring_type"` // "standard" or "conjugate_invariant"
	BootLogP        []int  `json:"boot_logp,omitempty"`
	BtpLogN         int    `json:"btp_logn,omitempty"`
}

// NewCKKSParameters constructs Lattigo ckks.Parameters from Params.
func (p Params) NewCKKSParameters() (ckks.Parameters, error) {
	rt := ring.ConjugateInvariant
	switch p.RingType {
	case "standard":
		rt = ring.Standard
	case "conjugate_invariant", "conjugateinvariant", "":
		rt = ring.ConjugateInvariant
	default:
		return ckks.Parameters{}, fmt.Errorf("unknown ring type: %q", p.RingType)
	}

	lit := ckks.ParametersLiteral{
		LogN:            p.LogN,
		LogQ:            p.LogQ,
		LogP:            p.LogP,
		LogDefaultScale: p.LogDefaultScale,
		Xs:              ring.Ternary{H: p.H},
		RingType:        rt,
	}

	// Bootstrap requires all CKKS primes to satisfy q = 1 mod 2^(btp_logn+1).
	if p.BtpLogN > 0 {
		logNthRoot := p.BtpLogN + 1
		lit.LogNthRoot = logNthRoot
	}

	return ckks.NewParametersFromLiteral(lit)
}

// MaxSlots returns the maximum number of plaintext slots.
func (p Params) MaxSlots() int {
	if p.RingType == "standard" {
		return 1 << (p.LogN - 1)
	}
	return 1 << p.LogN
}

// DefaultScale returns 2^LogDefaultScale as a uint64.
func (p Params) DefaultScale() uint64 {
	return 1 << uint(p.LogDefaultScale)
}

// MaxLevel returns the maximum multiplicative level (len(LogQ) - 1).
func (p Params) MaxLevel() int {
	return len(p.LogQ) - 1
}
