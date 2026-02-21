package orionclient

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCiphertextMarshalRoundTrip(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	values := []float64{1.5, 2.5, 3.5, 4.5}
	pt, err := c.Encode(values, 3, c.DefaultScale())
	require.NoError(t, err)
	ct, err := c.Encrypt(pt)
	require.NoError(t, err)

	// Marshal
	data, err := ct.Marshal()
	require.NoError(t, err)
	assert.True(t, len(data) > 0)

	// Verify magic header
	assert.Equal(t, []byte("ORTXT\x00\x01\x00"), data[:8])

	// Unmarshal
	ct2, err := UnmarshalCiphertext(data)
	require.NoError(t, err)

	assert.Equal(t, ct.NumCiphertexts(), ct2.NumCiphertexts())
	assert.Equal(t, ct.Shape(), ct2.Shape())
	assert.Equal(t, ct.Level(), ct2.Level())
	assert.Equal(t, ct.Scale(), ct2.Scale())
	assert.Equal(t, ct.Degree(), ct2.Degree())

	// Decrypt the deserialized ciphertext
	pts, err := c.Decrypt(ct2)
	require.NoError(t, err)
	decoded, err := c.Decode(pts[0])
	require.NoError(t, err)

	for i := range values {
		assert.InDelta(t, values[i], decoded[i], 1e-4,
			"mismatch at index %d", i)
	}
}

func TestCiphertextMarshalMultiple(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	// Create a ciphertext with multiple underlying cts
	v1 := []float64{1.0, 2.0}
	v2 := []float64{3.0, 4.0}

	pt1, err := c.Encode(v1, 3, c.DefaultScale())
	require.NoError(t, err)
	ct1, err := c.Encrypt(pt1)
	require.NoError(t, err)

	pt2, err := c.Encode(v2, 3, c.DefaultScale())
	require.NoError(t, err)
	ct2, err := c.Encrypt(pt2)
	require.NoError(t, err)

	// Combine into a multi-ct Ciphertext
	combined := NewCiphertext(
		append(ct1.Raw(), ct2.Raw()...),
		[]int{2, 2},
	)
	assert.Equal(t, 2, combined.NumCiphertexts())

	// Marshal/unmarshal round-trip
	data, err := combined.Marshal()
	require.NoError(t, err)

	restored, err := UnmarshalCiphertext(data)
	require.NoError(t, err)

	assert.Equal(t, 2, restored.NumCiphertexts())
	assert.Equal(t, []int{2, 2}, restored.Shape())

	// Decrypt both
	pts, err := c.Decrypt(restored)
	require.NoError(t, err)
	require.Len(t, pts, 2)

	d1, err := c.Decode(pts[0])
	require.NoError(t, err)
	d2, err := c.Decode(pts[1])
	require.NoError(t, err)

	assert.InDelta(t, 1.0, d1[0], 1e-4)
	assert.InDelta(t, 2.0, d1[1], 1e-4)
	assert.InDelta(t, 3.0, d2[0], 1e-4)
	assert.InDelta(t, 4.0, d2[1], 1e-4)
}

func TestCiphertextMetadata(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	pt, err := c.Encode([]float64{1.0}, 2, c.DefaultScale())
	require.NoError(t, err)
	ct, err := c.Encrypt(pt)
	require.NoError(t, err)

	assert.Equal(t, 1, ct.NumCiphertexts())
	assert.Equal(t, []int{1}, ct.Shape())
	assert.Equal(t, 2, ct.Level())
	assert.True(t, ct.Scale() > 0)
	assert.True(t, ct.Slots() > 0)
	assert.Equal(t, 1, ct.Degree())
}

func TestCiphertextSetScale(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	pt, err := c.Encode([]float64{1.0}, 3, c.DefaultScale())
	require.NoError(t, err)
	ct, err := c.Encrypt(pt)
	require.NoError(t, err)

	newScale := uint64(1 << 25)
	ct.SetScale(newScale)
	assert.Equal(t, newScale, ct.Scale())
}

func TestUnmarshalCiphertextBadMagic(t *testing.T) {
	// Must be at least 20 bytes to pass the length check (8 magic + 4 numCts + 4 shapeLen + 4 CRC)
	data := []byte("BADMAGIC0000000000001234")
	_, err := UnmarshalCiphertext(data)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid ciphertext magic")
}

func TestUnmarshalCiphertextTooShort(t *testing.T) {
	_, err := UnmarshalCiphertext([]byte{1, 2, 3})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "too short")
}

func TestUnmarshalCiphertextBadCRC(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	pt, err := c.Encode([]float64{1.0}, 3, c.DefaultScale())
	require.NoError(t, err)
	ct, err := c.Encrypt(pt)
	require.NoError(t, err)

	data, err := ct.Marshal()
	require.NoError(t, err)

	// Corrupt the CRC
	data[len(data)-1] ^= 0xFF

	_, err = UnmarshalCiphertext(data)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "CRC32 mismatch")
}

func TestEmptyCiphertextMetadata(t *testing.T) {
	ct := NewCiphertext(nil, nil)
	assert.Equal(t, 0, ct.NumCiphertexts())
	assert.Equal(t, -1, ct.Level())
	assert.Equal(t, uint64(0), ct.Scale())
	assert.Equal(t, 0, ct.Slots())
	assert.Equal(t, -1, ct.Degree())
}
