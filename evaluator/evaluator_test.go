package evaluator

import (
	"os"
	"testing"

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
