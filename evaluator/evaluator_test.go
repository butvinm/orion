package evaluator

import (
	"math"
	"os"
	"testing"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/orion/orionclient"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestEvaluator loads the MLP model, generates keys, and creates an Evaluator.
// Returns the model, evaluator, and client (for encrypting test inputs).
func newTestEvaluator(t *testing.T) (*Model, *Evaluator, *orionclient.Client) {
	t.Helper()

	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	model, err := LoadModel(data)
	require.NoError(t, err)

	params, manifest, _ := model.ClientParams()

	client, err := orionclient.New(params)
	require.NoError(t, err)

	keys, err := client.GenerateKeys(manifest)
	require.NoError(t, err)

	eval, err := NewEvaluator(params, *keys)
	require.NoError(t, err)

	return model, eval, client
}

func TestNewEvaluatorAndClose(t *testing.T) {
	model, eval, client := newTestEvaluator(t)
	defer client.Close()
	_ = model

	// Evaluator should be usable.
	assert.NotNil(t, eval.eval)
	assert.NotNil(t, eval.encoder)
	assert.NotNil(t, eval.linEval)
	assert.NotNil(t, eval.polyEval)

	// Close should nil out all fields.
	eval.Close()
	assert.Nil(t, eval.eval)
	assert.Nil(t, eval.encoder)
	assert.Nil(t, eval.linEval)
	assert.Nil(t, eval.polyEval)
}

func TestForwardStubOpsReturnError(t *testing.T) {
	model, eval, client := newTestEvaluator(t)
	defer client.Close()
	defer eval.Close()

	// Create a dummy ciphertext via the client.
	_, _, inputLevel := model.ClientParams()
	maxSlots := model.params.MaxSlots()
	zeros := make([]float64, maxSlots)

	pt, err := client.Encode(zeros, inputLevel, client.DefaultScale())
	require.NoError(t, err)

	ct, err := client.Encrypt(pt)
	require.NoError(t, err)

	// Forward should fail: flatten is the input node (skipped), then fc1
	// is linear_transform which is a stub -> "not yet implemented" error.
	_, err = eval.Forward(model, ct.Raw()[0])
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not yet implemented")
}

func TestForwardClosedEvaluatorReturnsError(t *testing.T) {
	model, eval, client := newTestEvaluator(t)
	defer client.Close()

	// Close the evaluator first.
	eval.Close()

	// Create a dummy ciphertext.
	_, _, inputLevel := model.ClientParams()
	maxSlots := model.params.MaxSlots()
	zeros := make([]float64, maxSlots)

	pt, err := client.Encode(zeros, inputLevel, client.DefaultScale())
	require.NoError(t, err)

	ct, err := client.Encrypt(pt)
	require.NoError(t, err)

	_, err = eval.Forward(model, ct.Raw()[0])
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "evaluator is closed")
}

// encryptVector is a helper that encodes and encrypts a float64 slice at a given level.
func encryptVector(t *testing.T, client *orionclient.Client, values []float64, level int) *orionclient.Ciphertext {
	t.Helper()
	pt, err := client.Encode(values, level, client.DefaultScale())
	require.NoError(t, err)
	ct, err := client.Encrypt(pt)
	require.NoError(t, err)
	return ct
}

// decryptVector is a helper that decrypts a raw rlwe.Ciphertext and returns the decoded float64 slice.
func decryptVector(t *testing.T, client *orionclient.Client, ct *orionclient.Ciphertext) []float64 {
	t.Helper()
	pts, err := client.Decrypt(ct)
	require.NoError(t, err)
	require.Len(t, pts, 1)
	vals, err := client.Decode(pts[0])
	require.NoError(t, err)
	return vals
}

func TestEvalQuad(t *testing.T) {
	model, eval, client := newTestEvaluator(t)
	defer client.Close()
	defer eval.Close()
	_ = model

	maxSlots := client.MaxSlots()
	// Input: [0.5, -0.3, 1.0, 0.0, ...], expected: [0.25, 0.09, 1.0, 0.0, ...]
	input := make([]float64, maxSlots)
	input[0] = 0.5
	input[1] = -0.3
	input[2] = 1.0

	expected := make([]float64, maxSlots)
	for i := range input {
		expected[i] = input[i] * input[i]
	}

	// Encrypt at a level that allows MulRelin + Rescale (need at least 2 levels).
	// The MLP model's input level is typically 5, so we have plenty of room.
	_, _, inputLevel := model.ClientParams()
	ct := encryptVector(t, client, input, inputLevel)

	// Apply quad directly.
	result, err := eval.evalQuad(ct.Raw()[0])
	require.NoError(t, err)

	// Decrypt and compare.
	wrapped := orionclient.NewCiphertext([]*rlwe.Ciphertext{result}, nil)
	decoded := decryptVector(t, client, wrapped)

	numCheck := 10 // check first 10 slots
	for i := 0; i < numCheck; i++ {
		assert.InDelta(t, expected[i], decoded[i], 1e-3,
			"slot %d: expected %f, got %f", i, expected[i], decoded[i])
	}
}

func TestEvalAdd(t *testing.T) {
	model, eval, client := newTestEvaluator(t)
	defer client.Close()
	defer eval.Close()

	maxSlots := client.MaxSlots()
	a := make([]float64, maxSlots)
	b := make([]float64, maxSlots)
	expected := make([]float64, maxSlots)
	a[0] = 1.5
	a[1] = -2.0
	a[2] = 3.14
	b[0] = 0.5
	b[1] = 2.0
	b[2] = -1.14
	for i := range a {
		expected[i] = a[i] + b[i]
	}

	_, _, inputLevel := model.ClientParams()
	ctA := encryptVector(t, client, a, inputLevel)
	ctB := encryptVector(t, client, b, inputLevel)

	result, err := eval.evalAdd(ctA.Raw()[0], ctB.Raw()[0])
	require.NoError(t, err)

	wrapped := orionclient.NewCiphertext([]*rlwe.Ciphertext{result}, nil)
	decoded := decryptVector(t, client, wrapped)

	numCheck := 10
	for i := 0; i < numCheck; i++ {
		assert.InDelta(t, expected[i], decoded[i], 1e-3,
			"slot %d: expected %f, got %f", i, expected[i], decoded[i])
	}
}

func TestEvalMult(t *testing.T) {
	model, eval, client := newTestEvaluator(t)
	defer client.Close()
	defer eval.Close()

	maxSlots := client.MaxSlots()
	a := make([]float64, maxSlots)
	b := make([]float64, maxSlots)
	expected := make([]float64, maxSlots)
	a[0] = 2.0
	a[1] = -1.5
	a[2] = 0.7
	b[0] = 3.0
	b[1] = 2.0
	b[2] = -0.5
	for i := range a {
		expected[i] = a[i] * b[i]
	}

	_, _, inputLevel := model.ClientParams()
	ctA := encryptVector(t, client, a, inputLevel)
	ctB := encryptVector(t, client, b, inputLevel)

	result, err := eval.evalMult(ctA.Raw()[0], ctB.Raw()[0])
	require.NoError(t, err)

	wrapped := orionclient.NewCiphertext([]*rlwe.Ciphertext{result}, nil)
	decoded := decryptVector(t, client, wrapped)

	numCheck := 10
	for i := 0; i < numCheck; i++ {
		assert.InDelta(t, expected[i], decoded[i], 1e-3,
			"slot %d: expected %f, got %f", i, expected[i], decoded[i])
	}
}

func TestEvalFlatten(t *testing.T) {
	model, eval, client := newTestEvaluator(t)
	defer client.Close()
	defer eval.Close()

	maxSlots := client.MaxSlots()
	input := make([]float64, maxSlots)
	input[0] = 42.0
	input[1] = -7.5

	_, _, inputLevel := model.ClientParams()
	ct := encryptVector(t, client, input, inputLevel)

	// Flatten should return the same ciphertext (pointer equality).
	result, err := eval.evalFlatten(ct.Raw()[0])
	require.NoError(t, err)
	assert.Same(t, ct.Raw()[0], result, "flatten should return the same ciphertext pointer")

	// Verify values are preserved.
	wrapped := orionclient.NewCiphertext([]*rlwe.Ciphertext{result}, nil)
	decoded := decryptVector(t, client, wrapped)
	assert.InDelta(t, 42.0, decoded[0], 1e-3)
	assert.InDelta(t, -7.5, decoded[1], 1e-3)

	// Rest should be near zero.
	for i := 2; i < 10; i++ {
		assert.InDelta(t, 0.0, decoded[i], 1e-3)
	}
}

func TestEvalQuadLargeValues(t *testing.T) {
	// Test with a range of values to check numerical stability.
	model, eval, client := newTestEvaluator(t)
	defer client.Close()
	defer eval.Close()

	maxSlots := client.MaxSlots()
	input := make([]float64, maxSlots)
	for i := 0; i < 16; i++ {
		input[i] = float64(i-8) * 0.1 // [-0.8, -0.7, ..., 0.7]
	}

	_, _, inputLevel := model.ClientParams()
	ct := encryptVector(t, client, input, inputLevel)

	result, err := eval.evalQuad(ct.Raw()[0])
	require.NoError(t, err)

	wrapped := orionclient.NewCiphertext([]*rlwe.Ciphertext{result}, nil)
	decoded := decryptVector(t, client, wrapped)

	for i := 0; i < 16; i++ {
		expected := input[i] * input[i]
		assert.True(t, math.Abs(expected-decoded[i]) < 1e-3,
			"slot %d: expected %f, got %f", i, expected, decoded[i])
	}
}
