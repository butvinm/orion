package evaluator

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testContext holds raw Lattigo primitives for test encrypt/decrypt.
type testContext struct {
	ckksParams ckks.Parameters
	encoder    *ckks.Encoder
	encryptor  *rlwe.Encryptor
	decryptor  *rlwe.Decryptor
}

// newTestEvaluator loads the MLP model, generates keys via raw Lattigo, and
// creates an Evaluator using NewEvaluatorFromKeySet.
func newTestEvaluator(t *testing.T) (*Model, *Evaluator, *testContext) {
	t.Helper()

	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	model, err := LoadModel(data)
	require.NoError(t, err)

	params, manifest, _ := model.ClientParams()

	ckksParams, err := params.NewCKKSParameters()
	require.NoError(t, err)

	eval, ctx := setupEvaluator(t, ckksParams, manifest.GaloisElements, manifest.NeedsRLK)
	return model, eval, ctx
}

// setupEvaluator generates keys and creates an Evaluator + testContext.
func setupEvaluator(t *testing.T, ckksParams ckks.Parameters, galoisElements []uint64, needsRLK bool) (*Evaluator, *testContext) {
	t.Helper()

	kg := rlwe.NewKeyGenerator(ckksParams)
	sk := kg.GenSecretKeyNew()
	pk := kg.GenPublicKeyNew(sk)

	var rlk *rlwe.RelinearizationKey
	if needsRLK {
		rlk = kg.GenRelinearizationKeyNew(sk)
	}

	galoisKeys := make([]*rlwe.GaloisKey, 0, len(galoisElements))
	for _, ge := range galoisElements {
		galoisKeys = append(galoisKeys, kg.GenGaloisKeyNew(ge, sk))
	}

	evk := rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...)
	eval, err := NewEvaluatorFromKeySet(ckksParams, evk, nil)
	require.NoError(t, err)

	ctx := &testContext{
		ckksParams: ckksParams,
		encoder:    ckks.NewEncoder(ckksParams),
		encryptor:  ckks.NewEncryptor(ckksParams, pk),
		decryptor:  ckks.NewDecryptor(ckksParams, sk),
	}

	return eval, ctx
}

// encryptVector encodes and encrypts a float64 slice at a given level.
func encryptVector(t *testing.T, ctx *testContext, values []float64, level int) *rlwe.Ciphertext {
	t.Helper()
	pt := ckks.NewPlaintext(ctx.ckksParams, level)
	pt.Scale = ctx.ckksParams.DefaultScale()
	require.NoError(t, ctx.encoder.Encode(values, pt))
	ct := ckks.NewCiphertext(ctx.ckksParams, 1, level)
	require.NoError(t, ctx.encryptor.Encrypt(pt, ct))
	return ct
}

// decryptVector decrypts a ciphertext and returns the decoded float64 slice.
func decryptVector(t *testing.T, ctx *testContext, ct *rlwe.Ciphertext) []float64 {
	t.Helper()
	pt := ckks.NewPlaintext(ctx.ckksParams, ct.Level())
	ctx.decryptor.Decrypt(ct, pt)
	result := make([]float64, ctx.ckksParams.MaxSlots())
	require.NoError(t, ctx.encoder.Decode(pt, result))
	return result
}

func TestNewEvaluatorAndClose(t *testing.T) {
	model, eval, _ := newTestEvaluator(t)
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
	_, eval, _ := newTestEvaluator(t)
	defer eval.Close()

	ct := &rlwe.Ciphertext{}
	_, err := eval.Forward(nil, []*rlwe.Ciphertext{ct})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "model is nil")
}

func TestForwardEmptyInput(t *testing.T) {
	model, eval, _ := newTestEvaluator(t)
	defer eval.Close()

	_, err := eval.Forward(model, []*rlwe.Ciphertext{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "input ciphertext list is empty")
}

func TestForwardNilInputCT(t *testing.T) {
	model, eval, _ := newTestEvaluator(t)
	defer eval.Close()

	_, err := eval.Forward(model, []*rlwe.Ciphertext{nil})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "input ciphertext[0] is nil")
}

func TestForwardBootstrapWithoutKeysReturnsError(t *testing.T) {
	model, eval, ctx := newTestEvaluator(t)
	defer eval.Close()

	// Inject a bootstrap node into the graph to test error when no btpKeys provided.
	btpConfig := `{"input_level":3,"input_min":-1,"input_max":1,"prescale":1,"postscale":1,"constant":0,"slots":128}`
	model.graph.Nodes["bootstrap_node"] = &Node{
		Name:      "bootstrap_node",
		Op:        "bootstrap",
		ConfigRaw: json.RawMessage(btpConfig),
	}
	model.graph.Order = []string{"flatten", "bootstrap_node"}
	model.graph.Input = "flatten"
	model.graph.Output = "bootstrap_node"
	model.graph.Inputs["bootstrap_node"] = []string{"flatten"}

	maxSlots := ctx.ckksParams.MaxSlots()
	zeros := make([]float64, maxSlots)
	_, _, inputLevel := model.ClientParams()

	ct := encryptVector(t, ctx, zeros, inputLevel)

	_, err := eval.Forward(model, []*rlwe.Ciphertext{ct})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "bootstrap keys not provided")
}

func TestForwardUnknownOpReturnsError(t *testing.T) {
	model, eval, ctx := newTestEvaluator(t)
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

	maxSlots := ctx.ckksParams.MaxSlots()
	zeros := make([]float64, maxSlots)
	_, _, inputLevel := model.ClientParams()

	ct := encryptVector(t, ctx, zeros, inputLevel)

	_, err := eval.Forward(model, []*rlwe.Ciphertext{ct})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unknown op")
}

func TestForwardClosedEvaluatorReturnsError(t *testing.T) {
	model, eval, ctx := newTestEvaluator(t)

	// Close the evaluator first.
	eval.Close()

	_, _, inputLevel := model.ClientParams()
	maxSlots := ctx.ckksParams.MaxSlots()
	zeros := make([]float64, maxSlots)

	ct := encryptVector(t, ctx, zeros, inputLevel)

	_, err := eval.Forward(model, []*rlwe.Ciphertext{ct})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "evaluator is closed")
}

func TestEvalQuad(t *testing.T) {
	model, eval, ctx := newTestEvaluator(t)
	defer eval.Close()

	maxSlots := ctx.ckksParams.MaxSlots()
	input := make([]float64, maxSlots)
	input[0] = 0.5
	input[1] = -0.3
	input[2] = 1.0

	expected := make([]float64, maxSlots)
	for i := range input {
		expected[i] = input[i] * input[i]
	}

	_, _, inputLevel := model.ClientParams()
	ct := encryptVector(t, ctx, input, inputLevel)

	result, err := eval.evalQuad(ct)
	require.NoError(t, err)

	decoded := decryptVector(t, ctx, result)

	numCheck := 10
	for i := 0; i < numCheck; i++ {
		assert.InDelta(t, expected[i], decoded[i], 1e-3,
			"slot %d: expected %f, got %f", i, expected[i], decoded[i])
	}
}

func TestEvalAdd(t *testing.T) {
	model, eval, ctx := newTestEvaluator(t)
	defer eval.Close()

	maxSlots := ctx.ckksParams.MaxSlots()
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
	ctA := encryptVector(t, ctx, a, inputLevel)
	ctB := encryptVector(t, ctx, b, inputLevel)

	result, err := eval.evalAdd(ctA, ctB)
	require.NoError(t, err)

	decoded := decryptVector(t, ctx, result)

	numCheck := 10
	for i := 0; i < numCheck; i++ {
		assert.InDelta(t, expected[i], decoded[i], 1e-3,
			"slot %d: expected %f, got %f", i, expected[i], decoded[i])
	}
}

func TestEvalMult(t *testing.T) {
	model, eval, ctx := newTestEvaluator(t)
	defer eval.Close()

	maxSlots := ctx.ckksParams.MaxSlots()
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
	ctA := encryptVector(t, ctx, a, inputLevel)
	ctB := encryptVector(t, ctx, b, inputLevel)

	result, err := eval.evalMult(ctA, ctB)
	require.NoError(t, err)

	decoded := decryptVector(t, ctx, result)

	numCheck := 10
	for i := 0; i < numCheck; i++ {
		assert.InDelta(t, expected[i], decoded[i], 1e-3,
			"slot %d: expected %f, got %f", i, expected[i], decoded[i])
	}
}

func TestEvalFlatten(t *testing.T) {
	model, eval, ctx := newTestEvaluator(t)
	defer eval.Close()

	maxSlots := ctx.ckksParams.MaxSlots()
	input := make([]float64, maxSlots)
	input[0] = 42.0
	input[1] = -7.5

	_, _, inputLevel := model.ClientParams()
	ct := encryptVector(t, ctx, input, inputLevel)

	// Flatten should return a copy (not the same pointer) to prevent aliasing.
	result, err := eval.evalFlatten(ct)
	require.NoError(t, err)
	assert.NotSame(t, ct, result, "flatten should return a copy, not the same pointer")

	decoded := decryptVector(t, ctx, result)
	assert.InDelta(t, 42.0, decoded[0], 1e-3)
	assert.InDelta(t, -7.5, decoded[1], 1e-3)

	for i := 2; i < 10; i++ {
		assert.InDelta(t, 0.0, decoded[i], 1e-3)
	}
}

func TestEvalQuadLargeValues(t *testing.T) {
	model, eval, ctx := newTestEvaluator(t)
	defer eval.Close()

	maxSlots := ctx.ckksParams.MaxSlots()
	input := make([]float64, maxSlots)
	for i := 0; i < 16; i++ {
		input[i] = float64(i-8) * 0.1
	}

	_, _, inputLevel := model.ClientParams()
	ct := encryptVector(t, ctx, input, inputLevel)

	result, err := eval.evalQuad(ct)
	require.NoError(t, err)

	decoded := decryptVector(t, ctx, result)

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

// loadModelAndEvaluate loads a model, generates keys, encrypts input, runs
// Forward, decrypts, and returns (decoded, expected) slices.
func loadModelAndEvaluate(t *testing.T, modelPath, inputPath, expectedPath string) ([]float64, []float64) {
	t.Helper()

	data, err := os.ReadFile(modelPath)
	require.NoError(t, err)

	model, err := LoadModel(data)
	require.NoError(t, err)

	params, manifest, inputLevel := model.ClientParams()

	ckksParams, err := params.NewCKKSParameters()
	require.NoError(t, err)

	eval, ctx := setupEvaluator(t, ckksParams, manifest.GaloisElements, manifest.NeedsRLK)
	defer eval.Close()

	// Load input and pad to maxSlots.
	inputValues := loadJSONFloats(t, inputPath)
	maxSlots := ctx.ckksParams.MaxSlots()
	padded := make([]float64, maxSlots)
	copy(padded, inputValues)

	ct := encryptVector(t, ctx, padded, inputLevel)

	// Run Forward with single-element CT list.
	results, err := eval.Forward(model, []*rlwe.Ciphertext{ct})
	require.NoError(t, err, "Forward failed")
	require.Len(t, results, 1, "expected 1 output CT")

	decoded := decryptVector(t, ctx, results[0])

	expected := loadJSONFloats(t, expectedPath)
	return decoded, expected
}

func TestForwardMLP(t *testing.T) {
	decoded, expected := loadModelAndEvaluate(t,
		"testdata/mlp.orion", "testdata/mlp.input.json", "testdata/mlp.expected.json")

	tolerance := 0.025

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

	tolerance := 0.07

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
	decoded, expected := loadModelAndEvaluate(t,
		"testdata/sigmoid_unfused.orion", "testdata/sigmoid_unfused.input.json", "testdata/sigmoid_unfused.expected.json")

	tolerance := 0.03

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

func TestForwardConv2d(t *testing.T) {
	decoded, expected := loadModelAndEvaluate(t,
		"testdata/conv2d.orion", "testdata/conv2d.input.json", "testdata/conv2d.expected.json")

	tolerance := 0.1

	t.Logf("Conv2d output (first %d values):", len(expected))
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
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	model, err := LoadModel(data)
	require.NoError(t, err)

	params, manifest, inputLevel := model.ClientParams()

	ckksParams, err := params.NewCKKSParameters()
	require.NoError(t, err)

	// Create two evaluators with independent keys sharing the same model.
	eval1, ctx1 := setupEvaluator(t, ckksParams, manifest.GaloisElements, manifest.NeedsRLK)
	defer eval1.Close()

	eval2, ctx2 := setupEvaluator(t, ckksParams, manifest.GaloisElements, manifest.NeedsRLK)
	defer eval2.Close()

	// Load input and expected output.
	inputValues := loadJSONFloats(t, "testdata/mlp.input.json")
	expectedValues := loadJSONFloats(t, "testdata/mlp.expected.json")

	maxSlots := ctx1.ckksParams.MaxSlots()
	padded := make([]float64, maxSlots)
	copy(padded, inputValues)

	// Evaluator 1: encrypt, forward, decrypt.
	ct1 := encryptVector(t, ctx1, padded, inputLevel)
	results1, err := eval1.Forward(model, []*rlwe.Ciphertext{ct1})
	require.NoError(t, err)
	require.Len(t, results1, 1)
	decoded1 := decryptVector(t, ctx1, results1[0])

	// Evaluator 2: encrypt, forward, decrypt.
	ct2 := encryptVector(t, ctx2, padded, inputLevel)
	results2, err := eval2.Forward(model, []*rlwe.Ciphertext{ct2})
	require.NoError(t, err)
	require.Len(t, results2, 1)
	decoded2 := decryptVector(t, ctx2, results2[0])

	tolerance := 0.025
	for i, v := range expectedValues {
		assert.InDelta(t, v, decoded1[i], tolerance,
			"eval1 slot %d: expected %f, got %f", i, v, decoded1[i])
		assert.InDelta(t, v, decoded2[i], tolerance,
			"eval2 slot %d: expected %f, got %f", i, v, decoded2[i])
	}
}
