package orionclient

import (
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
)

// Plaintext wraps a Lattigo plaintext with shape metadata.
type Plaintext struct {
	raw   *rlwe.Plaintext
	shape []int
}

// Raw returns the underlying Lattigo plaintext.
func (p *Plaintext) Raw() *rlwe.Plaintext {
	return p.raw
}

// Shape returns the logical tensor shape.
func (p *Plaintext) Shape() []int {
	dst := make([]int, len(p.shape))
	copy(dst, p.shape)
	return dst
}

// Level returns the multiplicative level.
func (p *Plaintext) Level() int {
	return p.raw.Level()
}

// Scale returns the plaintext scale as uint64.
func (p *Plaintext) Scale() uint64 {
	v, _ := p.raw.Scale.Value.Uint64()
	return v
}

// SetScale sets the plaintext scale.
func (p *Plaintext) SetScale(scale uint64) {
	p.raw.Scale = rlwe.NewScale(scale)
}

// Slots returns the number of usable slots.
func (p *Plaintext) Slots() int {
	return 1 << p.raw.LogDimensions.Cols
}
