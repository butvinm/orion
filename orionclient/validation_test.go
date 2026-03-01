package orionclient

import (
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Phase 7 validation tests: no global state, memory leaks, secret key zeroing,
// wire format test vectors.

// TestDifferentParamsCoexist creates two Clients with different CKKS parameters
// simultaneously and verifies they operate independently.
func TestDifferentParamsCoexist(t *testing.T) {
	// Params set 1: conjugate_invariant, 4 levels
	p1 := Params{
		LogN:     12,
		LogQ:     []int{45, 30, 30, 30},
		LogP:     []int{50},
		LogScale: 30,
		H:        64,
		RingType: "conjugate_invariant",
	}

	// Params set 2: standard ring, 3 levels, different scale
	p2 := Params{
		LogN:     12,
		LogQ:     []int{45, 30, 30},
		LogP:     []int{50},
		LogScale: 25,
		H:        64,
		RingType: "standard",
	}

	c1, err := New(p1)
	require.NoError(t, err)
	defer c1.Close()

	c2, err := New(p2)
	require.NoError(t, err)
	defer c2.Close()

	// Verify different parameters
	assert.NotEqual(t, c1.MaxSlots(), c2.MaxSlots(),
		"clients should have different slot counts (CI vs standard)")
	assert.NotEqual(t, c1.DefaultScale(), c2.DefaultScale(),
		"clients should have different default scales")

	// Encrypt with client 1
	v1 := []float64{1.0, 2.0, 3.0}
	pt1, err := c1.Encode(v1, p1.MaxLevel(), c1.DefaultScale())
	require.NoError(t, err)
	ct1, err := c1.Encrypt(pt1)
	require.NoError(t, err)

	// Encrypt with client 2
	v2 := []float64{10.0, 20.0, 30.0}
	pt2, err := c2.Encode(v2, p2.MaxLevel(), c2.DefaultScale())
	require.NoError(t, err)
	ct2, err := c2.Encrypt(pt2)
	require.NoError(t, err)

	// Decrypt each with its own client - must be independent
	pts1, err := c1.Decrypt(ct1)
	require.NoError(t, err)
	d1, err := c1.Decode(pts1[0])
	require.NoError(t, err)

	pts2, err := c2.Decrypt(ct2)
	require.NoError(t, err)
	d2, err := c2.Decode(pts2[0])
	require.NoError(t, err)

	for i := range v1 {
		assert.InDelta(t, v1[i], d1[i], 1e-4, "c1 mismatch at %d", i)
	}
	for i := range v2 {
		assert.InDelta(t, v2[i], d2[i], 1e-2, "c2 mismatch at %d (lower scale)", i)
	}
}

// TestDifferentParamsEvaluatorsCoexist creates two Evaluators with different
// CKKS parameters and verifies they operate independently.
func TestDifferentParamsEvaluatorsCoexist(t *testing.T) {
	p1 := Params{
		LogN:     12,
		LogQ:     []int{45, 30, 30, 30},
		LogP:     []int{50},
		LogScale: 30,
		H:        64,
		RingType: "conjugate_invariant",
	}
	p2 := Params{
		LogN:     12,
		LogQ:     []int{45, 30, 30},
		LogP:     []int{50},
		LogScale: 25,
		H:        64,
		RingType: "standard",
	}

	c1, err := New(p1)
	require.NoError(t, err)
	defer c1.Close()

	c2, err := New(p2)
	require.NoError(t, err)
	defer c2.Close()

	// Generate keys for each
	m1 := Manifest{NeedsRLK: true, GaloisElements: []uint64{c1.GaloisElement(1)}}
	k1, err := c1.GenerateKeys(m1)
	require.NoError(t, err)
	e1, err := NewEvaluator(p1, *k1)
	require.NoError(t, err)
	defer e1.Close()

	m2 := Manifest{NeedsRLK: true, GaloisElements: []uint64{c2.GaloisElement(1)}}
	k2, err := c2.GenerateKeys(m2)
	require.NoError(t, err)
	e2, err := NewEvaluator(p2, *k2)
	require.NoError(t, err)
	defer e2.Close()

	// Verify different slot counts
	assert.NotEqual(t, e1.MaxSlots(), e2.MaxSlots())

	// Operate on each independently
	v1 := []float64{5.0, 10.0}
	pt1, err := c1.Encode(v1, p1.MaxLevel(), c1.DefaultScale())
	require.NoError(t, err)
	ct1, err := c1.Encrypt(pt1)
	require.NoError(t, err)

	ct1out, err := e1.AddScalar(ct1, 100.0)
	require.NoError(t, err)
	pts1, err := c1.Decrypt(ct1out)
	require.NoError(t, err)
	d1, err := c1.Decode(pts1[0])
	require.NoError(t, err)
	assert.InDelta(t, 105.0, d1[0], 0.1)

	v2 := []float64{50.0, 100.0}
	pt2, err := c2.Encode(v2, p2.MaxLevel(), c2.DefaultScale())
	require.NoError(t, err)
	ct2, err := c2.Encrypt(pt2)
	require.NoError(t, err)

	ct2out, err := e2.AddScalar(ct2, 1.0)
	require.NoError(t, err)
	pts2, err := c2.Decrypt(ct2out)
	require.NoError(t, err)
	d2, err := c2.Decode(pts2[0])
	require.NoError(t, err)
	assert.InDelta(t, 51.0, d2[0], 0.1)
}

// TestNoMemoryLeakInferenceLoop runs an encode/encrypt/add/decrypt loop
// and checks that Go heap allocation stabilizes (doesn't grow unboundedly).
func TestNoMemoryLeakInferenceLoop(t *testing.T) {
	p := testParams()
	c, err := New(p)
	require.NoError(t, err)
	defer c.Close()

	m := Manifest{NeedsRLK: true, GaloisElements: []uint64{c.GaloisElement(1)}}
	keys, err := c.GenerateKeys(m)
	require.NoError(t, err)
	eval, err := NewEvaluator(p, *keys)
	require.NoError(t, err)
	defer eval.Close()

	// Warm up: run a few iterations so runtime caches stabilize
	for i := 0; i < 5; i++ {
		v := []float64{float64(i), float64(i + 1)}
		pt, err := c.Encode(v, p.MaxLevel(), c.DefaultScale())
		require.NoError(t, err)
		ct, err := c.Encrypt(pt)
		require.NoError(t, err)
		ctOut, err := eval.AddScalar(ct, 1.0)
		require.NoError(t, err)
		_, err = c.Decrypt(ctOut)
		require.NoError(t, err)
	}

	// Force GC and record baseline
	runtime.GC()
	var baseline runtime.MemStats
	runtime.ReadMemStats(&baseline)

	// Run inference loop
	const iterations = 50
	for i := 0; i < iterations; i++ {
		v := []float64{float64(i), float64(i + 1)}
		pt, err := c.Encode(v, p.MaxLevel(), c.DefaultScale())
		require.NoError(t, err)
		ct, err := c.Encrypt(pt)
		require.NoError(t, err)
		ctOut, err := eval.AddScalar(ct, 1.0)
		require.NoError(t, err)
		_, err = c.Decrypt(ctOut)
		require.NoError(t, err)
	}

	// Force GC and measure
	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	// HeapInuse should not grow more than 50MB beyond baseline.
	// CKKS ring elements are large but should be GC'd between iterations.
	growth := int64(after.HeapInuse) - int64(baseline.HeapInuse)
	t.Logf("heap growth over %d iterations: %d bytes (%.1f MB)",
		iterations, growth, float64(growth)/1024/1024)

	const maxGrowthBytes = 50 * 1024 * 1024 // 50 MB
	assert.True(t, growth < maxGrowthBytes,
		"heap grew %d bytes (%.1f MB) over %d iterations, exceeds %d MB limit",
		growth, float64(growth)/1024/1024, iterations, maxGrowthBytes/(1024*1024))
}

// TestSecretKeyCoefficientsZeroed verifies that Client.Close() actually zeroes
// the polynomial coefficients of the secret key, not just nils the pointer.
func TestSecretKeyCoefficientsZeroed(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)

	// Capture references to the coefficient slices before Close
	qCoeffs := c.sk.Value.Q.Coeffs
	require.True(t, len(qCoeffs) > 0, "secret key must have Q coefficients")

	// Verify at least some coefficients are non-zero before Close
	hasNonZero := false
	for _, ring := range qCoeffs {
		for _, coeff := range ring {
			if coeff != 0 {
				hasNonZero = true
				break
			}
		}
		if hasNonZero {
			break
		}
	}
	assert.True(t, hasNonZero, "secret key should have non-zero coefficients before Close")

	// Close should zero the coefficients
	c.Close()

	// The underlying slice memory should now be all zeros
	for i, ring := range qCoeffs {
		for j, coeff := range ring {
			assert.Equal(t, uint64(0), coeff,
				"Q coefficient [%d][%d] not zeroed after Close", i, j)
		}
	}

	// Also verify nil references
	assert.Nil(t, c.sk)
	assert.Nil(t, c.keygen)
	assert.Nil(t, c.encryptor)
	assert.Nil(t, c.decryptor)
	assert.Nil(t, c.encoder)
}

// TestWireFormatTestVectors verifies the Ciphertext wire format structure:
// magic header and metadata encoding.
func TestWireFormatTestVectors(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	values := []float64{1.0, 2.0, 3.0}
	pt, err := c.Encode(values, 3, c.DefaultScale())
	require.NoError(t, err)
	ct, err := c.Encrypt(pt)
	require.NoError(t, err)

	data, err := ct.Marshal()
	require.NoError(t, err)

	// Verify magic header: "ORTXT\x00\x01\x00" (8 bytes)
	magic := data[:8]
	assert.Equal(t, byte('O'), magic[0])
	assert.Equal(t, byte('R'), magic[1])
	assert.Equal(t, byte('T'), magic[2])
	assert.Equal(t, byte('X'), magic[3])
	assert.Equal(t, byte('T'), magic[4])
	assert.Equal(t, byte(0x00), magic[5])
	assert.Equal(t, byte(0x01), magic[6]) // version 1
	assert.Equal(t, byte(0x00), magic[7])

	// After magic: uint32 numCiphertexts = 1
	assert.True(t, len(data) > 12)

	// Round-trip: unmarshal must reproduce identical metadata
	ct2, err := UnmarshalCiphertext(data)
	require.NoError(t, err)
	assert.Equal(t, ct.NumCiphertexts(), ct2.NumCiphertexts())
	assert.Equal(t, ct.Level(), ct2.Level())
	assert.Equal(t, ct.Scale(), ct2.Scale())
	assert.Equal(t, ct.Shape(), ct2.Shape())
	assert.Equal(t, ct.Slots(), ct2.Slots())
	assert.Equal(t, ct.Degree(), ct2.Degree())

	// Re-marshal must produce identical bytes (deterministic)
	data2, err := ct2.Marshal()
	require.NoError(t, err)
	assert.Equal(t, data, data2, "re-marshaled bytes must be identical")
}

// TestWireFormatCrossVersionRejection verifies that an unknown version
// in the magic header is rejected.
func TestWireFormatCrossVersionRejection(t *testing.T) {
	c, err := New(testParams())
	require.NoError(t, err)
	defer c.Close()

	pt, err := c.Encode([]float64{1.0}, 3, c.DefaultScale())
	require.NoError(t, err)
	ct, err := c.Encrypt(pt)
	require.NoError(t, err)

	data, err := ct.Marshal()
	require.NoError(t, err)

	// Corrupt version byte (index 6: 0x01 -> 0x02)
	corrupted := make([]byte, len(data))
	copy(corrupted, data)
	corrupted[6] = 0x02

	_, err = UnmarshalCiphertext(corrupted)
	assert.Error(t, err, "should reject unknown version")
}

// TestErrorPropagationInGo verifies that Go functions return proper errors
// (not panics) on invalid inputs.
func TestErrorPropagationInGo(t *testing.T) {
	// Invalid params should error, not panic
	badParams := Params{LogN: 3, LogQ: []int{10}, LogP: []int{10}, LogScale: 5, RingType: "standard"}
	_, err := New(badParams)
	assert.Error(t, err, "invalid params should produce an error")

	// Unknown ring type
	badRing := testParams()
	badRing.RingType = "quantum_resistant"
	_, err = New(badRing)
	assert.Error(t, err)

	// Operations on closed client
	c, err := New(testParams())
	require.NoError(t, err)
	c.Close()

	_, err = c.Encode([]float64{1.0}, 0, 1<<30)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "closed")

	_, err = c.Encrypt(nil)
	assert.Error(t, err)

	_, err = c.SecretKey()
	assert.Error(t, err)

	// Operations on closed evaluator
	p := testParams()
	c2, err := New(p)
	require.NoError(t, err)
	defer c2.Close()

	m := Manifest{NeedsRLK: true}
	k, err := c2.GenerateKeys(m)
	require.NoError(t, err)
	eval, err := NewEvaluator(p, *k)
	require.NoError(t, err)
	eval.Close()

	ct := encryptValues(t, c2, []float64{1.0})
	_, err = eval.Add(ct, ct)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "closed")
}
