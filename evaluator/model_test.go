package evaluator

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadModelMLP(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	model, err := LoadModel(data)
	require.NoError(t, err)

	// Verify graph structure.
	assert.Equal(t, 4, len(model.graph.Nodes))
	assert.Equal(t, "flatten", model.graph.Input)
	assert.Equal(t, "fc2", model.graph.Output)

	// Verify 2 linear transforms (fc1 and fc2).
	assert.Equal(t, 2, len(model.transforms))
	assert.Contains(t, model.transforms, "fc1")
	assert.Contains(t, model.transforms, "fc2")

	// Each LT should have at least one diagonal ref.
	assert.Greater(t, len(model.transforms["fc1"]), 0)
	assert.Greater(t, len(model.transforms["fc2"]), 0)

	// Verify 2 biases (fc1 and fc2).
	assert.Equal(t, 2, len(model.biases))
	assert.Contains(t, model.biases, "fc1")
	assert.Contains(t, model.biases, "fc2")

	// Verify 0 polynomials (MLP uses quad, not polynomial).
	assert.Equal(t, 0, len(model.polys))
}

func TestLoadModelSigmoid(t *testing.T) {
	data, err := os.ReadFile("testdata/sigmoid.orion")
	require.NoError(t, err)

	model, err := LoadModel(data)
	require.NoError(t, err)

	// Verify 1 polynomial (act1 is sigmoid = chebyshev polynomial).
	assert.Equal(t, 1, len(model.polys))
	assert.Contains(t, model.polys, "act1")

	// Verify polynomial has coefficients.
	poly := model.polys["act1"]
	// The polynomial should have non-zero degree.
	assert.Greater(t, poly.Degree(), 0, "polynomial should have non-zero degree")

	// Verify 2 LTs still present.
	assert.Equal(t, 2, len(model.transforms))

	// Verify 2 biases.
	assert.Equal(t, 2, len(model.biases))
}

func TestClientParams(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	model, err := LoadModel(data)
	require.NoError(t, err)

	params, manifest, inputLevel := model.ClientParams()

	// Verify params match header.
	assert.Equal(t, 13, params.LogN)
	assert.Equal(t, []int{29, 26, 26, 26, 26, 26}, params.LogQ)
	assert.Equal(t, 8192, params.H)
	assert.Equal(t, "conjugate_invariant", params.RingType)

	// Verify manifest has galois elements.
	assert.Greater(t, len(manifest.GaloisElements), 0, "manifest should have galois elements")
	assert.True(t, manifest.NeedsRLK)

	// Verify galois elements are uint64 (converted from int in JSON).
	for _, ge := range manifest.GaloisElements {
		assert.Greater(t, ge, uint64(0), "galois element should be positive")
	}

	// Verify input level > 0.
	assert.Greater(t, inputLevel, 0)
	assert.Equal(t, 3, inputLevel)
}

func TestLoadModelTruncatedData(t *testing.T) {
	// Empty data.
	_, err := LoadModel([]byte{})
	assert.Error(t, err)

	// Wrong magic.
	badData := make([]byte, 100)
	copy(badData, "NOTMAGIC")
	_, err = LoadModel(badData)
	assert.Error(t, err)

	// Valid magic but truncated after that.
	truncated := make([]byte, 12)
	copy(truncated, []byte{'O', 'R', 'I', 'O', 'N', 0x00, 0x02, 0x00})
	// header length of 1000 but only 12 bytes total
	truncated[8] = 0xe8
	truncated[9] = 0x03
	truncated[10] = 0x00
	truncated[11] = 0x00
	_, err = LoadModel(truncated)
	assert.Error(t, err)
}
