package orion

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewCKKSParametersWithBtpLogN(t *testing.T) {
	// Standard ring, bootstrap-enabled params.
	p := Params{
		LogN:     14,
		LogQ:     []int{55, 40, 40, 40},
		LogP:     []int{61, 61},
		LogScale: 40,
		H:        192,
		RingType: "standard",
		BootLogP: []int{61, 61, 61, 61, 61, 61},
		BtpLogN:  14,
	}

	ckksParams, err := p.NewCKKSParameters()
	require.NoError(t, err)

	// LogNthRoot should be btp_logn + 1 = 15.
	assert.Equal(t, 15, ckksParams.LogNthRoot())
}

func TestNewCKKSParametersWithoutBtpLogN(t *testing.T) {
	// Non-bootstrap params: BtpLogN=0 (zero value), LogNthRoot should
	// default to LogN+1 for standard ring (Lattigo default).
	p := Params{
		LogN:     13,
		LogQ:     []int{29, 26, 26, 26, 26, 26},
		LogP:     []int{29, 29},
		LogScale: 26,
		H:        8192,
		RingType: "conjugate_invariant",
	}

	ckksParams, err := p.NewCKKSParameters()
	require.NoError(t, err)

	// For conjugate_invariant, Lattigo defaults LogNthRoot to LogN+2.
	assert.Equal(t, 15, ckksParams.LogNthRoot())
}

func TestBtpLogNFieldSerialization(t *testing.T) {
	// Verify BtpLogN is included in JSON when non-zero.
	p := Params{
		LogN:     14,
		LogQ:     []int{55, 40},
		LogP:     []int{61},
		LogScale: 40,
		H:        192,
		RingType: "standard",
		BtpLogN:  14,
	}
	assert.Equal(t, 14, p.BtpLogN)

	// Zero value should behave as "no bootstrap".
	p2 := Params{
		LogN:     13,
		LogQ:     []int{29, 26},
		LogP:     []int{29},
		LogScale: 26,
		H:        8192,
		RingType: "standard",
	}
	assert.Equal(t, 0, p2.BtpLogN)
}
