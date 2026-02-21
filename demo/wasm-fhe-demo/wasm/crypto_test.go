package main

import (
	"math"
	"testing"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/orion/orionclient"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test parameters — small for speed.
var testParams = orionclient.Params{
	LogN:     13,
	LogQ:     []int{29, 26, 26, 26, 26, 26},
	LogP:     []int{29, 29},
	LogScale: 26,
	H:        8192,
	RingType: "conjugate_invariant",
}

func newTestClient(t *testing.T) *orionclient.Client {
	t.Helper()
	c, err := orionclient.New(testParams)
	require.NoError(t, err)
	t.Cleanup(func() { c.Close() })
	return c
}

func TestClientNew(t *testing.T) {
	c := newTestClient(t)
	assert.Greater(t, c.MaxSlots(), 0)
}

func TestMaxSlotsPowerOfTwo(t *testing.T) {
	c := newTestClient(t)
	slots := c.MaxSlots()
	assert.Greater(t, slots, 0)
	assert.Equal(t, 0, slots&(slots-1), "MaxSlots should be a power of 2")
}

func TestDefaultScale(t *testing.T) {
	c := newTestClient(t)
	expected := uint64(1) << uint(testParams.LogScale)
	assert.Equal(t, expected, c.DefaultScale())
}

func TestGenerateRLK(t *testing.T) {
	c := newTestClient(t)
	data, err := c.GenerateRLK()
	require.NoError(t, err)
	assert.Greater(t, len(data), 0)

	// Round-trip: unmarshal and verify.
	rlk := &rlwe.RelinearizationKey{}
	require.NoError(t, rlk.UnmarshalBinary(data))
}

func TestGenerateGaloisKey(t *testing.T) {
	c := newTestClient(t)
	galEl := c.GaloisElement(1)
	data, err := c.GenerateGaloisKey(galEl)
	require.NoError(t, err)
	assert.Greater(t, len(data), 0)

	gk := &rlwe.GaloisKey{}
	require.NoError(t, gk.UnmarshalBinary(data))
}

func TestMultipleGaloisKeys(t *testing.T) {
	c := newTestClient(t)
	rotations := []int{1, 2, 4, 8}
	for _, rot := range rotations {
		galEl := c.GaloisElement(rot)
		data, err := c.GenerateGaloisKey(galEl)
		require.NoError(t, err, "rotation %d", rot)
		assert.Greater(t, len(data), 0, "rotation %d", rot)
	}
}

func TestEncodeEncryptDecryptRoundTrip(t *testing.T) {
	c := newTestClient(t)

	original := []float64{0.1, 0.5, -0.3, 1.0, 0.0, -1.0, 0.42, 0.99}
	scale := c.DefaultScale()
	level := len(testParams.LogQ) - 1

	// Encode.
	pt, err := c.Encode(original, level, scale)
	require.NoError(t, err)

	// Encrypt.
	ct, err := c.Encrypt(pt)
	require.NoError(t, err)
	assert.Equal(t, 1, ct.NumCiphertexts())

	// Decrypt.
	pts, err := c.Decrypt(ct)
	require.NoError(t, err)
	require.Len(t, pts, 1)

	// Decode.
	result, err := c.Decode(pts[0])
	require.NoError(t, err)

	// Verify values match within CKKS approximation tolerance.
	for i, v := range original {
		assert.InDelta(t, v, result[i], 1e-3, "slot %d", i)
	}
}

func TestEncryptRawBytesRoundTrip(t *testing.T) {
	// Simulates the WASM encrypt/decrypt path: raw Lattigo MarshalBinary bytes.
	c := newTestClient(t)

	original := []float64{3.14, 2.71, 1.41}
	scale := c.DefaultScale()
	level := len(testParams.LogQ) - 1

	pt, err := c.Encode(original, level, scale)
	require.NoError(t, err)

	ct, err := c.Encrypt(pt)
	require.NoError(t, err)

	// Marshal single ciphertext to raw Lattigo bytes (what WASM orionEncrypt returns).
	rawBytes, err := ct.Raw()[0].MarshalBinary()
	require.NoError(t, err)
	assert.Greater(t, len(rawBytes), 0)

	// Unmarshal from raw bytes (what WASM orionDecrypt receives).
	rawCt := &rlwe.Ciphertext{}
	require.NoError(t, rawCt.UnmarshalBinary(rawBytes))

	// Wrap and decrypt.
	ct2 := orionclient.NewCiphertext([]*rlwe.Ciphertext{rawCt}, []int{c.MaxSlots()})
	pts, err := c.Decrypt(ct2)
	require.NoError(t, err)

	result, err := c.Decode(pts[0])
	require.NoError(t, err)

	for i, v := range original {
		assert.InDelta(t, v, result[i], 1e-3, "slot %d", i)
	}
}

func TestMultipleClientsCoexist(t *testing.T) {
	c1, err := orionclient.New(testParams)
	require.NoError(t, err)
	defer c1.Close()

	c2, err := orionclient.New(testParams)
	require.NoError(t, err)
	defer c2.Close()

	original := []float64{1.0, 2.0, 3.0}
	scale := c1.DefaultScale()
	level := len(testParams.LogQ) - 1

	pt1, err := c1.Encode(original, level, scale)
	require.NoError(t, err)
	ct1, err := c1.Encrypt(pt1)
	require.NoError(t, err)

	// Decrypt with c1 should work.
	pts, err := c1.Decrypt(ct1)
	require.NoError(t, err)
	result, err := c1.Decode(pts[0])
	require.NoError(t, err)
	for i, v := range original {
		assert.InDelta(t, v, result[i], 1e-3, "slot %d", i)
	}

	// Decrypt with c2 produces garbage (different secret key).
	pts2, err := c2.Decrypt(ct1)
	require.NoError(t, err)
	result2, err := c2.Decode(pts2[0])
	require.NoError(t, err)

	anyDifferent := false
	for i, v := range original {
		if math.Abs(v-result2[i]) > 1.0 {
			anyDifferent = true
			break
		}
	}
	assert.True(t, anyDifferent, "decrypting with wrong key should produce different values")
}
