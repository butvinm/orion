package evaluator

import (
	"encoding/json"
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

func TestForwardNilModel(t *testing.T) {
	_, eval, client := newTestEvaluator(t)
	defer client.Close()
	defer eval.Close()

	ct := &rlwe.Ciphertext{}
	_, err := eval.Forward(nil, ct)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "model is nil")
}

func TestForwardNilInput(t *testing.T) {
	model, eval, client := newTestEvaluator(t)
	defer client.Close()
	defer eval.Close()

	_, err := eval.Forward(model, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "input ciphertext is nil")
}

func TestForwardBootstrapStubReturnsError(t *testing.T) {
	// Build a synthetic model with a bootstrap node to verify the stub returns an error.
	model, eval, client := newTestEvaluator(t)
	defer client.Close()
	defer eval.Close()

	// Inject a bootstrap node into the graph to test the error path.
	model.graph.Nodes["bootstrap_node"] = &Node{
		Name: "bootstrap_node",
		Op:   "bootstrap",
	}
	// Replace the graph order and input/output to route through the bootstrap node.
	model.graph.Order = []string{"flatten", "bootstrap_node"}
	model.graph.Input = "flatten"
	model.graph.Output = "bootstrap_node"
	model.graph.Inputs["bootstrap_node"] = []string{"flatten"}

	maxSlots := model.params.MaxSlots()
	zeros := make([]float64, maxSlots)
	_, _, inputLevel := model.ClientParams()

	pt, err := client.Encode(zeros, inputLevel, client.DefaultScale())
	require.NoError(t, err)

	ct, err := client.Encrypt(pt)
	require.NoError(t, err)

	_, err = eval.Forward(model, ct.Raw()[0])
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not yet implemented")
}

func TestForwardUnknownOpReturnsError(t *testing.T) {
	model, eval, client := newTestEvaluator(t)
	defer client.Close()
	defer eval.Close()

	// Inject an unknown op node.
	model.graph.Nodes["unknown_node"] = &Node{
		Name: "unknown_node",
		Op:   "nonexistent_op",
	}
	model.graph.Order = []string{"flatten", "unknown_node"}
	model.graph.Input = "flatten"
	model.graph.Output = "unknown_node"
	model.graph.Inputs["unknown_node"] = []string{"flatten"}

	maxSlots := model.params.MaxSlots()
	zeros := make([]float64, maxSlots)
	_, _, inputLevel := model.ClientParams()

	pt, err := client.Encode(zeros, inputLevel, client.DefaultScale())
	require.NoError(t, err)

	ct, err := client.Encrypt(pt)
	require.NoError(t, err)

	_, err = eval.Forward(model, ct.Raw()[0])
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unknown op")
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

// loadJSONFloats reads a JSON file containing a float64 array.
func loadJSONFloats(t *testing.T, path string) []float64 {
	t.Helper()
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	var values []float64
	require.NoError(t, json.Unmarshal(data, &values))
	return values
}

// loadModelAndEvaluate is a shared helper that loads a model, generates keys,
// encrypts input, runs Forward, decrypts, and returns (decoded, expected) slices.
func loadModelAndEvaluate(t *testing.T, modelPath, inputPath, expectedPath string) ([]float64, []float64) {
	t.Helper()

	data, err := os.ReadFile(modelPath)
	require.NoError(t, err)

	model, err := LoadModel(data)
	require.NoError(t, err)

	params, manifest, inputLevel := model.ClientParams()

	client, err := orionclient.New(params)
	require.NoError(t, err)
	defer client.Close()

	keys, err := client.GenerateKeys(manifest)
	require.NoError(t, err)

	eval, err := NewEvaluator(params, *keys)
	require.NoError(t, err)
	defer eval.Close()

	// Load input and pad to maxSlots.
	inputValues := loadJSONFloats(t, inputPath)
	maxSlots := model.params.MaxSlots()
	padded := make([]float64, maxSlots)
	copy(padded, inputValues)

	// Encode and encrypt.
	pt, err := client.Encode(padded, inputLevel, client.DefaultScale())
	require.NoError(t, err)

	ct, err := client.Encrypt(pt)
	require.NoError(t, err)

	// Run Forward.
	result, err := eval.Forward(model, ct.Raw()[0])
	require.NoError(t, err, "Forward failed")

	// Decrypt and decode.
	wrapped := orionclient.NewCiphertext([]*rlwe.Ciphertext{result}, nil)
	decoded := decryptVector(t, client, wrapped)

	expected := loadJSONFloats(t, expectedPath)
	return decoded, expected
}

func TestForwardMLP(t *testing.T) {
	decoded, expected := loadModelAndEvaluate(t,
		"testdata/mlp.orion", "testdata/mlp.input.json", "testdata/mlp.expected.json")

	// Tolerance 0.02: with logscale=26 and multiple FHE ops (2 LTs + quad + rescales),
	// CKKS noise accumulates to ~0.01. Observed max error is ~0.01 with these params.
	tolerance := 0.02

	t.Logf("MLP output (first %d values):", len(expected))
	for i, v := range expected {
		t.Logf("  [%d] expected=%.6f got=%.6f diff=%.6f", i, v, decoded[i], math.Abs(v-decoded[i]))
	}

	for i, v := range expected {
		assert.InDelta(t, v, decoded[i], tolerance,
			"slot %d: expected %f, got %f", i, v, decoded[i])
	}
}

func TestForwardSigmoid(t *testing.T) {
	decoded, expected := loadModelAndEvaluate(t,
		"testdata/sigmoid.orion", "testdata/sigmoid.input.json", "testdata/sigmoid.expected.json")

	// Tolerance 0.05: With logscale=26 parameters, 2 LTs + Chebyshev polynomial eval + rescales,
	// CKKS noise varies across random key generations. Observed range: 0.007–0.033 max error.
	// Higher than MLP because polynomial evaluation introduces more noise than simple squaring.
	tolerance := 0.05

	t.Logf("Sigmoid output (first %d values):", len(expected))
	for i, v := range expected {
		diff := math.Abs(v - decoded[i])
		t.Logf("  [%d] expected=%.6f got=%.6f diff=%.6f", i, v, decoded[i], diff)
	}

	for i, v := range expected {
		assert.InDelta(t, v, decoded[i], tolerance,
			"slot %d: expected %f, got %f", i, v, decoded[i])
	}
}

func TestForwardSigmoidUnfused(t *testing.T) {
	// This test exercises the fuse_modules=false code path in evalPolynomial.
	// When unfused, prescale and constant are applied before polynomial evaluation:
	//   x = x * prescale
	//   x = x + constant
	//   x = chebyshev(x)
	decoded, expected := loadModelAndEvaluate(t,
		"testdata/sigmoid_unfused.orion", "testdata/sigmoid_unfused.input.json", "testdata/sigmoid_unfused.expected.json")

	tolerance := 0.05

	t.Logf("Sigmoid unfused output (first %d values):", len(expected))
	for i, v := range expected {
		diff := math.Abs(v - decoded[i])
		t.Logf("  [%d] expected=%.6f got=%.6f diff=%.6f", i, v, decoded[i], diff)
	}

	for i, v := range expected {
		assert.InDelta(t, v, decoded[i], tolerance,
			"slot %d: expected %f, got %f", i, v, decoded[i])
	}
}

func TestMultipleEvaluatorsShareModel(t *testing.T) {
	// Load model once — it should be shareable across evaluators.
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	model, err := LoadModel(data)
	require.NoError(t, err)

	params, manifest, inputLevel := model.ClientParams()

	// Create two separate clients with independent key pairs.
	client1, err := orionclient.New(params)
	require.NoError(t, err)
	defer client1.Close()

	client2, err := orionclient.New(params)
	require.NoError(t, err)
	defer client2.Close()

	// Generate separate eval key bundles.
	keys1, err := client1.GenerateKeys(manifest)
	require.NoError(t, err)

	keys2, err := client2.GenerateKeys(manifest)
	require.NoError(t, err)

	// Create two evaluators sharing the same model.
	eval1, err := NewEvaluator(params, *keys1)
	require.NoError(t, err)
	defer eval1.Close()

	eval2, err := NewEvaluator(params, *keys2)
	require.NoError(t, err)
	defer eval2.Close()

	// Load input and expected output.
	inputValues := loadJSONFloats(t, "testdata/mlp.input.json")
	expectedValues := loadJSONFloats(t, "testdata/mlp.expected.json")

	maxSlots := model.params.MaxSlots()
	padded := make([]float64, maxSlots)
	copy(padded, inputValues)

	// Client 1 encrypts, evaluator 1 evaluates, client 1 decrypts.
	pt1, err := client1.Encode(padded, inputLevel, client1.DefaultScale())
	require.NoError(t, err)
	ct1, err := client1.Encrypt(pt1)
	require.NoError(t, err)

	result1, err := eval1.Forward(model, ct1.Raw()[0])
	require.NoError(t, err)

	wrapped1 := orionclient.NewCiphertext([]*rlwe.Ciphertext{result1}, nil)
	decoded1 := decryptVector(t, client1, wrapped1)

	// Client 2 encrypts, evaluator 2 evaluates, client 2 decrypts.
	pt2, err := client2.Encode(padded, inputLevel, client2.DefaultScale())
	require.NoError(t, err)
	ct2, err := client2.Encrypt(pt2)
	require.NoError(t, err)

	result2, err := eval2.Forward(model, ct2.Raw()[0])
	require.NoError(t, err)

	wrapped2 := orionclient.NewCiphertext([]*rlwe.Ciphertext{result2}, nil)
	decoded2 := decryptVector(t, client2, wrapped2)

	// Both should produce correct results independently.
	tolerance := 0.02
	for i, v := range expectedValues {
		assert.InDelta(t, v, decoded1[i], tolerance,
			"eval1 slot %d: expected %f, got %f", i, v, decoded1[i])
		assert.InDelta(t, v, decoded2[i], tolerance,
			"eval2 slot %d: expected %f, got %f", i, v, decoded2[i])
	}
}
