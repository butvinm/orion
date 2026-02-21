package orionclient

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testParams returns small CKKS parameters suitable for fast tests.
func testParams() Params {
	return Params{
		LogN:     12,
		LogQ:     []int{45, 30, 30, 30},
		LogP:     []int{50},
		LogScale: 30,
		H:        64,
		RingType: "conjugate_invariant",
	}
}

func TestNewClient(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	assert.Equal(t, 1<<12, c.MaxSlots())
	assert.Equal(t, uint64(1<<30), c.DefaultScale())
}

func TestEncodeDecodeRoundTrip(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	values := make([]float64, 64)
	for i := range values {
		values[i] = float64(i) * 0.1
	}

	pt, err := c.Encode(values, 3, c.DefaultScale())
	require.NoError(t, err)

	decoded, err := c.Decode(pt)
	require.NoError(t, err)

	// CKKS is approximate -- check within tolerance
	for i := range values {
		assert.InDelta(t, values[i], decoded[i], 1e-4,
			"mismatch at index %d", i)
	}
}

func TestEncryptDecryptRoundTrip(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	values := []float64{1.0, 2.0, 3.0, 4.0}
	pt, err := c.Encode(values, 3, c.DefaultScale())
	require.NoError(t, err)

	ct, err := c.Encrypt(pt)
	require.NoError(t, err)

	assert.Equal(t, 1, ct.NumCiphertexts())
	assert.True(t, ct.Level() >= 0)
	assert.True(t, ct.Slots() > 0)
	assert.Equal(t, 1, ct.Degree())

	pts, err := c.Decrypt(ct)
	require.NoError(t, err)
	require.Len(t, pts, 1)

	decoded, err := c.Decode(pts[0])
	require.NoError(t, err)

	for i := range values {
		assert.InDelta(t, values[i], decoded[i], 1e-4,
			"mismatch at index %d", i)
	}
}

func TestSecretKeyPersistence(t *testing.T) {
	p := testParams()
	c1, err := New(p)
	require.NoError(t, err)

	// Encrypt with first client
	values := []float64{10.0, 20.0, 30.0}
	pt, err := c1.Encode(values, 3, c1.DefaultScale())
	require.NoError(t, err)
	ct, err := c1.Encrypt(pt)
	require.NoError(t, err)

	// Serialize ciphertext and secret key
	ctBytes, err := ct.Marshal()
	require.NoError(t, err)
	skBytes, err := c1.SecretKey()
	require.NoError(t, err)
	c1.Close()

	// Restore with second client
	c2, err := FromSecretKey(p, skBytes)
	require.NoError(t, err)
	defer c2.Close()

	ct2, err := UnmarshalCiphertext(ctBytes)
	require.NoError(t, err)

	pts, err := c2.Decrypt(ct2)
	require.NoError(t, err)
	decoded, err := c2.Decode(pts[0])
	require.NoError(t, err)

	for i := range values {
		assert.InDelta(t, values[i], decoded[i], 1e-4,
			"mismatch at index %d", i)
	}
}

func TestMultipleClientsCoexist(t *testing.T) {
	p := testParams()

	c1, err := New(p)
	require.NoError(t, err)
	defer c1.Close()

	c2, err := New(p)
	require.NoError(t, err)
	defer c2.Close()

	// Encrypt with c1
	v1 := []float64{1.0, 2.0}
	pt1, err := c1.Encode(v1, 3, c1.DefaultScale())
	require.NoError(t, err)
	ct1, err := c1.Encrypt(pt1)
	require.NoError(t, err)

	// Encrypt with c2
	v2 := []float64{100.0, 200.0}
	pt2, err := c2.Encode(v2, 3, c2.DefaultScale())
	require.NoError(t, err)
	ct2, err := c2.Encrypt(pt2)
	require.NoError(t, err)

	// Decrypt each with its own client
	pts1, err := c1.Decrypt(ct1)
	require.NoError(t, err)
	d1, err := c1.Decode(pts1[0])
	require.NoError(t, err)

	pts2, err := c2.Decrypt(ct2)
	require.NoError(t, err)
	d2, err := c2.Decode(pts2[0])
	require.NoError(t, err)

	assert.InDelta(t, 1.0, d1[0], 1e-4)
	assert.InDelta(t, 2.0, d1[1], 1e-4)
	assert.InDelta(t, 100.0, d2[0], 1e-4)
	assert.InDelta(t, 200.0, d2[1], 1e-4)
}

func TestGenerateRLK(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	rlk, err := c.GenerateRLK()
	require.NoError(t, err)
	assert.True(t, len(rlk) > 0)
}

func TestGenerateGaloisKey(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	galEl := c.GaloisElement(1)
	gk, err := c.GenerateGaloisKey(galEl)
	require.NoError(t, err)
	assert.True(t, len(gk) > 0)
}

func TestGenerateKeys(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	manifest := Manifest{
		GaloisElements: []uint64{
			c.GaloisElement(1),
			c.GaloisElement(2),
		},
		NeedsRLK: true,
	}

	bundle, err := c.GenerateKeys(manifest)
	require.NoError(t, err)
	assert.True(t, len(bundle.RLK) > 0)
	assert.Len(t, bundle.GaloisKeys, 2)
	for _, v := range bundle.GaloisKeys {
		assert.True(t, len(v) > 0)
	}
}

func TestCloseZerosSecretKey(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)

	// Close should zero the key
	c.Close()

	assert.Nil(t, c.sk)
	assert.Nil(t, c.keygen)
	assert.Nil(t, c.encryptor)
	assert.Nil(t, c.decryptor)
	assert.Nil(t, c.encoder)
}

func TestClosedClientErrors(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	c.Close()

	_, err = c.SecretKey()
	assert.Error(t, err)

	_, err = c.Encode([]float64{1.0}, 0, 1<<30)
	assert.Error(t, err)

	_, err = c.GenerateRLK()
	assert.Error(t, err)
}

func TestParamsMaxSlots(t *testing.T) {
	p := testParams()
	assert.Equal(t, 1<<12, p.MaxSlots())

	p.RingType = "standard"
	assert.Equal(t, 1<<11, p.MaxSlots())
}

func TestParamsMaxLevel(t *testing.T) {
	p := testParams()
	assert.Equal(t, 3, p.MaxLevel())
}

func TestParamsInvalidRingType(t *testing.T) {
	p := testParams()
	p.RingType = "invalid"
	_, err := p.NewCKKSParameters()
	assert.Error(t, err)
}

func TestPlaintextMetadata(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	values := make([]float64, 32)
	for i := range values {
		values[i] = math.Sin(float64(i))
	}

	pt, err := c.Encode(values, 2, c.DefaultScale())
	require.NoError(t, err)

	assert.Equal(t, 2, pt.Level())
	assert.Equal(t, c.DefaultScale(), pt.Scale())
	assert.True(t, pt.Slots() > 0)
	assert.Equal(t, []int{32}, pt.Shape())
}
