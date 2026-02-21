package orionclient

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// helper: create a Client + Evaluator pair with rotation keys for steps 1,2.
func setupClientEvaluator(t *testing.T) (*Client, *Evaluator) {
	t.Helper()
	p := testParams()
	c, err := New(p)
	require.NoError(t, err)

	manifest := Manifest{
		GaloisElements: []uint64{
			c.GaloisElement(1),
			c.GaloisElement(2),
		},
		NeedsRLK: true,
	}
	keys, err := c.GenerateKeys(manifest)
	require.NoError(t, err)

	eval, err := NewEvaluator(p, *keys)
	require.NoError(t, err)

	return c, eval
}

// helper: encrypt float64 slice into a *Ciphertext at max level.
func encryptValues(t *testing.T, c *Client, values []float64) *Ciphertext {
	t.Helper()
	p := testParams()
	pt, err := c.Encode(values, p.MaxLevel(), c.DefaultScale())
	require.NoError(t, err)
	ct, err := c.Encrypt(pt)
	require.NoError(t, err)
	return ct
}

// helper: decrypt a *Ciphertext to float64 slice.
func decryptValues(t *testing.T, c *Client, ct *Ciphertext) []float64 {
	t.Helper()
	pts, err := c.Decrypt(ct)
	require.NoError(t, err)
	require.Len(t, pts, 1)
	decoded, err := c.Decode(pts[0])
	require.NoError(t, err)
	return decoded
}

func TestEvaluatorNewAndClose(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	defer eval.Close()

	assert.Equal(t, c.MaxSlots(), eval.MaxSlots())
}

func TestEvaluatorAdd(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	defer eval.Close()

	v1 := []float64{1.0, 2.0, 3.0, 4.0}
	v2 := []float64{10.0, 20.0, 30.0, 40.0}
	ct1 := encryptValues(t, c, v1)
	ct2 := encryptValues(t, c, v2)

	ctSum, err := eval.Add(ct1, ct2)
	require.NoError(t, err)

	result := decryptValues(t, c, ctSum)
	for i := range v1 {
		assert.InDelta(t, v1[i]+v2[i], result[i], 0.1, "Add mismatch at %d", i)
	}
}

func TestEvaluatorSub(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	defer eval.Close()

	v1 := []float64{10.0, 20.0, 30.0, 40.0}
	v2 := []float64{1.0, 2.0, 3.0, 4.0}
	ct1 := encryptValues(t, c, v1)
	ct2 := encryptValues(t, c, v2)

	ctDiff, err := eval.Sub(ct1, ct2)
	require.NoError(t, err)

	result := decryptValues(t, c, ctDiff)
	for i := range v1 {
		assert.InDelta(t, v1[i]-v2[i], result[i], 0.1, "Sub mismatch at %d", i)
	}
}

func TestEvaluatorMulCiphertext(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	defer eval.Close()

	v1 := []float64{1.0, 2.0, 3.0, 4.0}
	v2 := []float64{2.0, 3.0, 4.0, 5.0}
	ct1 := encryptValues(t, c, v1)
	ct2 := encryptValues(t, c, v2)

	ctProd, err := eval.Mul(ct1, ct2)
	require.NoError(t, err)

	result := decryptValues(t, c, ctProd)
	for i := range v1 {
		assert.InDelta(t, v1[i]*v2[i], result[i], 0.5, "Mul mismatch at %d", i)
	}
}

func TestEvaluatorAddScalar(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	defer eval.Close()

	v := []float64{1.0, 2.0, 3.0, 4.0}
	ct := encryptValues(t, c, v)

	ctOut, err := eval.AddScalar(ct, 10.0)
	require.NoError(t, err)

	result := decryptValues(t, c, ctOut)
	for i := range v {
		assert.InDelta(t, v[i]+10.0, result[i], 0.1, "AddScalar mismatch at %d", i)
	}
}

func TestEvaluatorMulScalar(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	defer eval.Close()

	v := []float64{1.0, 2.0, 3.0, 4.0}
	ct := encryptValues(t, c, v)

	ctOut, err := eval.MulScalar(ct, 3.0)
	require.NoError(t, err)

	result := decryptValues(t, c, ctOut)
	for i := range v {
		assert.InDelta(t, v[i]*3.0, result[i], 0.5, "MulScalar mismatch at %d", i)
	}
}

func TestEvaluatorNegate(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	defer eval.Close()

	v := []float64{1.0, 2.0, 3.0, 4.0}
	ct := encryptValues(t, c, v)

	ctNeg, err := eval.Negate(ct)
	require.NoError(t, err)

	result := decryptValues(t, c, ctNeg)
	for i := range v {
		assert.InDelta(t, -v[i], result[i], 0.5, "Negate mismatch at %d", i)
	}
}

func TestEvaluatorAddPlaintext(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	defer eval.Close()

	p := testParams()
	v := []float64{1.0, 2.0, 3.0, 4.0}
	ptVals := []float64{10.0, 20.0, 30.0, 40.0}
	ct := encryptValues(t, c, v)

	pt, err := eval.Encode(ptVals, p.MaxLevel(), c.DefaultScale())
	require.NoError(t, err)

	ctOut, err := eval.AddPlaintext(ct, pt)
	require.NoError(t, err)

	result := decryptValues(t, c, ctOut)
	for i := range v {
		assert.InDelta(t, v[i]+ptVals[i], result[i], 0.1, "AddPlaintext mismatch at %d", i)
	}
}

func TestEvaluatorMulPlaintext(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	defer eval.Close()

	p := testParams()
	v := []float64{1.0, 2.0, 3.0, 4.0}
	ptVals := []float64{2.0, 3.0, 4.0, 5.0}
	ct := encryptValues(t, c, v)

	pt, err := eval.Encode(ptVals, p.MaxLevel(), c.DefaultScale())
	require.NoError(t, err)

	ctOut, err := eval.MulPlaintext(ct, pt)
	require.NoError(t, err)

	result := decryptValues(t, c, ctOut)
	for i := range v {
		assert.InDelta(t, v[i]*ptVals[i], result[i], 0.5, "MulPlaintext mismatch at %d", i)
	}
}

func TestEvaluatorRotate(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	defer eval.Close()

	n := 8
	v := make([]float64, n)
	for i := range v {
		v[i] = float64(i + 1)
	}
	ct := encryptValues(t, c, v)

	ctRot, err := eval.Rotate(ct, 1)
	require.NoError(t, err)

	result := decryptValues(t, c, ctRot)
	// After rotation by 1: [2, 3, 4, ..., n, ...]
	for i := 0; i < n-1; i++ {
		assert.InDelta(t, v[i+1], result[i], 0.1, "Rotate mismatch at %d", i)
	}
}

func TestEvaluatorRescale(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	defer eval.Close()

	// Ciphertext-ciphertext multiplication roughly squares the scale.
	// Rescaling then reduces scale and drops one level.
	v1 := []float64{1.0, 2.0, 3.0, 4.0}
	v2 := []float64{1.0, 1.0, 1.0, 1.0}
	ct1 := encryptValues(t, c, v1)
	ct2 := encryptValues(t, c, v2)

	levelBefore := ct1.Level()

	ctMul, err := eval.Mul(ct1, ct2)
	require.NoError(t, err)

	ctResc, err := eval.Rescale(ctMul)
	require.NoError(t, err)

	assert.True(t, ctResc.Level() < levelBefore,
		"level should decrease after rescale: before=%d, after=%d", levelBefore, ctResc.Level())

	result := decryptValues(t, c, ctResc)
	for i := range v1 {
		assert.InDelta(t, v1[i]*v2[i], result[i], 0.5, "Rescale mismatch at %d", i)
	}
}

func TestEvaluatorPolynomialEval(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	defer eval.Close()

	// Evaluate p(x) = 2x + 1 (monomial basis)
	v := []float64{0.1, 0.2, 0.3, 0.4}
	ct := encryptValues(t, c, v)

	poly := GenerateMonomial([]float64{1.0, 2.0}) // 1 + 2x
	ctOut, err := eval.EvalPoly(ct, poly.Raw(), c.DefaultScale())
	require.NoError(t, err)

	result := decryptValues(t, c, ctOut)
	for i := range v {
		expected := 1.0 + 2.0*v[i]
		assert.InDelta(t, expected, result[i], 0.5, "poly eval mismatch at %d", i)
	}
}

func TestEvaluatorLinearTransform(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	defer eval.Close()

	p := testParams()

	// Create a simple diagonal linear transform: identity * 2.0
	// Diagonal 0 = [2.0, 2.0, 2.0, ...]
	slots := p.MaxSlots()
	diag := make([]float64, slots)
	for i := range diag {
		diag[i] = 2.0
	}

	lt, err := GenerateLinearTransform(p, map[int][]float64{0: diag}, p.MaxLevel(), 1.0)
	require.NoError(t, err)

	// Need Galois elements for the LT evaluation
	galEls, err := lt.RequiredGaloisElements(p)
	require.NoError(t, err)

	// Recreate evaluator with the required Galois keys
	manifest := Manifest{
		GaloisElements: galEls,
		NeedsRLK:       true,
	}
	keys, err := c.GenerateKeys(manifest)
	require.NoError(t, err)
	eval.Close()

	eval2, err := NewEvaluator(p, *keys)
	require.NoError(t, err)
	defer eval2.Close()

	v := []float64{1.0, 2.0, 3.0, 4.0}
	ct := encryptValues(t, c, v)

	ctOut, err := eval2.EvalLinearTransform(ct, lt)
	require.NoError(t, err)

	result := decryptValues(t, c, ctOut)
	for i := range v {
		assert.InDelta(t, v[i]*2.0, result[i], 1.0, "LT eval mismatch at %d", i)
	}
}

func TestLinearTransformMarshalRoundTrip(t *testing.T) {
	p := testParams()
	slots := p.MaxSlots()

	diag := make([]float64, slots)
	for i := range diag {
		diag[i] = float64(i) * 0.01
	}

	lt, err := GenerateLinearTransform(p, map[int][]float64{0: diag}, p.MaxLevel(), 1.0)
	require.NoError(t, err)

	data, err := lt.Marshal()
	require.NoError(t, err)
	assert.True(t, len(data) > 0)

	lt2, err := LoadLinearTransform(data)
	require.NoError(t, err)

	assert.Equal(t, lt.raw.LevelQ, lt2.raw.LevelQ)
	assert.Equal(t, lt.raw.LevelP, lt2.raw.LevelP)
	assert.Equal(t, lt.raw.LogBabyStepGiantStepRatio, lt2.raw.LogBabyStepGiantStepRatio)
	assert.Equal(t, len(lt.raw.Vec), len(lt2.raw.Vec))
}

func TestEvaluatorMultipleInstancesCoexist(t *testing.T) {
	p := testParams()

	c1, err := New(p)
	require.NoError(t, err)
	defer c1.Close()

	c2, err := New(p)
	require.NoError(t, err)
	defer c2.Close()

	// Create evaluators with their own keys
	m1 := Manifest{NeedsRLK: true, GaloisElements: []uint64{c1.GaloisElement(1)}}
	k1, err := c1.GenerateKeys(m1)
	require.NoError(t, err)
	e1, err := NewEvaluator(p, *k1)
	require.NoError(t, err)
	defer e1.Close()

	m2 := Manifest{NeedsRLK: true, GaloisElements: []uint64{c2.GaloisElement(1)}}
	k2, err := c2.GenerateKeys(m2)
	require.NoError(t, err)
	e2, err := NewEvaluator(p, *k2)
	require.NoError(t, err)
	defer e2.Close()

	// Use evaluator 1
	v1 := []float64{1.0, 2.0}
	ct1 := encryptValues(t, c1, v1)
	ct1Add, err := e1.AddScalar(ct1, 5.0)
	require.NoError(t, err)
	r1 := decryptValues(t, c1, ct1Add)
	assert.InDelta(t, 6.0, r1[0], 0.1)
	assert.InDelta(t, 7.0, r1[1], 0.1)

	// Use evaluator 2 independently
	v2 := []float64{100.0, 200.0}
	ct2 := encryptValues(t, c2, v2)
	ct2Add, err := e2.AddScalar(ct2, 5.0)
	require.NoError(t, err)
	r2 := decryptValues(t, c2, ct2Add)
	assert.InDelta(t, 105.0, r2[0], 0.1)
	assert.InDelta(t, 205.0, r2[1], 0.1)
}

func TestEvaluatorClosedErrors(t *testing.T) {
	c, eval := setupClientEvaluator(t)
	defer c.Close()
	eval.Close()

	v := []float64{1.0}
	ct := encryptValues(t, c, v)

	_, err := eval.Add(ct, ct)
	assert.Error(t, err)

	_, err = eval.Encode([]float64{1.0}, 0, 1<<30)
	assert.Error(t, err)
}
