package orionclient

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
)

var ciphertextMagic = [8]byte{'O', 'R', 'T', 'X', 'T', 0x00, 0x01, 0x00}

// Ciphertext wraps one or more Lattigo ciphertexts with shape metadata.
// This is the unified type replacing both CipherText and CipherTensor.
type Ciphertext struct {
	cts   []*rlwe.Ciphertext
	shape []int
}

// NewCiphertext creates a Ciphertext from raw Lattigo ciphertexts and a shape.
func NewCiphertext(cts []*rlwe.Ciphertext, shape []int) *Ciphertext {
	return &Ciphertext{cts: cts, shape: shape}
}

// Raw returns the underlying Lattigo ciphertexts.
func (c *Ciphertext) Raw() []*rlwe.Ciphertext {
	return c.cts
}

// NumCiphertexts returns the number of underlying ciphertexts.
func (c *Ciphertext) NumCiphertexts() int {
	return len(c.cts)
}

// Shape returns the logical tensor shape.
func (c *Ciphertext) Shape() []int {
	dst := make([]int, len(c.shape))
	copy(dst, c.shape)
	return dst
}

// Level returns the multiplicative level of the first ciphertext.
func (c *Ciphertext) Level() int {
	if len(c.cts) == 0 {
		return -1
	}
	return c.cts[0].Level()
}

// Scale returns the scale of the first ciphertext as uint64.
func (c *Ciphertext) Scale() uint64 {
	if len(c.cts) == 0 {
		return 0
	}
	v, _ := c.cts[0].Scale.Value.Uint64()
	return v
}

// SetScale sets the scale on all underlying ciphertexts.
func (c *Ciphertext) SetScale(scale uint64) {
	s := rlwe.NewScale(scale)
	for _, ct := range c.cts {
		ct.Scale = s
	}
}

// Slots returns the number of usable slots of the first ciphertext.
func (c *Ciphertext) Slots() int {
	if len(c.cts) == 0 {
		return 0
	}
	return 1 << c.cts[0].LogDimensions.Cols
}

// Degree returns the polynomial degree of the first ciphertext.
func (c *Ciphertext) Degree() int {
	if len(c.cts) == 0 {
		return -1
	}
	return c.cts[0].Degree()
}

// Marshal serializes the Ciphertext to wire format.
//
// Wire format:
//
//	[8]  magic "ORTXT\x00\x01\x00"
//	[4]  num_ciphertexts  (uint32 LE)
//	[4]  shape_len        (uint32 LE)
//	[shape_len * 4] shape dims (int32 LE each)
//	for each ciphertext:
//	  [8]  ct_len  (uint64 LE)
//	  [ct_len] ct_data (Lattigo MarshalBinary)
//	[4]  CRC32 of everything above
func (c *Ciphertext) Marshal() ([]byte, error) {
	// Estimate capacity
	buf := make([]byte, 0, 1024)

	// Magic
	buf = append(buf, ciphertextMagic[:]...)

	// num_ciphertexts
	tmp4 := make([]byte, 4)
	binary.LittleEndian.PutUint32(tmp4, uint32(len(c.cts)))
	buf = append(buf, tmp4...)

	// shape_len
	binary.LittleEndian.PutUint32(tmp4, uint32(len(c.shape)))
	buf = append(buf, tmp4...)

	// shape dims
	for _, d := range c.shape {
		binary.LittleEndian.PutUint32(tmp4, uint32(int32(d)))
		buf = append(buf, tmp4...)
	}

	// ciphertexts
	tmp8 := make([]byte, 8)
	for i, ct := range c.cts {
		data, err := ct.MarshalBinary()
		if err != nil {
			return nil, fmt.Errorf("marshalling ciphertext %d: %w", i, err)
		}
		binary.LittleEndian.PutUint64(tmp8, uint64(len(data)))
		buf = append(buf, tmp8...)
		buf = append(buf, data...)
	}

	// CRC32
	checksum := crc32.ChecksumIEEE(buf)
	binary.LittleEndian.PutUint32(tmp4, checksum)
	buf = append(buf, tmp4...)

	return buf, nil
}

// UnmarshalCiphertext deserializes a Ciphertext from wire format.
func UnmarshalCiphertext(data []byte) (*Ciphertext, error) {
	if len(data) < 8+4+4+4 {
		return nil, fmt.Errorf("ciphertext data too short: %d bytes", len(data))
	}

	// Verify magic
	if string(data[:8]) != string(ciphertextMagic[:]) {
		return nil, fmt.Errorf("invalid ciphertext magic: %x", data[:8])
	}

	// Verify CRC32
	payload := data[:len(data)-4]
	storedCRC := binary.LittleEndian.Uint32(data[len(data)-4:])
	computedCRC := crc32.ChecksumIEEE(payload)
	if storedCRC != computedCRC {
		return nil, fmt.Errorf("CRC32 mismatch: stored=%08x computed=%08x", storedCRC, computedCRC)
	}

	off := 8

	// num_ciphertexts
	numCts := int(binary.LittleEndian.Uint32(data[off : off+4]))
	off += 4

	// shape_len
	shapeLen := int(binary.LittleEndian.Uint32(data[off : off+4]))
	off += 4

	// shape
	shape := make([]int, shapeLen)
	for i := range shape {
		if off+4 > len(payload) {
			return nil, fmt.Errorf("unexpected end of data reading shape")
		}
		shape[i] = int(int32(binary.LittleEndian.Uint32(data[off : off+4])))
		off += 4
	}

	// ciphertexts
	cts := make([]*rlwe.Ciphertext, numCts)
	for i := range cts {
		if off+8 > len(payload) {
			return nil, fmt.Errorf("unexpected end of data reading ciphertext %d length", i)
		}
		ctLen := int(binary.LittleEndian.Uint64(data[off : off+8]))
		off += 8

		if off+ctLen > len(payload) {
			return nil, fmt.Errorf("unexpected end of data reading ciphertext %d body", i)
		}

		ct := &rlwe.Ciphertext{}
		if err := ct.UnmarshalBinary(data[off : off+ctLen]); err != nil {
			return nil, fmt.Errorf("unmarshalling ciphertext %d: %w", i, err)
		}
		cts[i] = ct
		off += ctLen
	}

	return &Ciphertext{cts: cts, shape: shape}, nil
}
