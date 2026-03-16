package evaluator

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseContainerMLP(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	header, blobs, err := ParseContainer(data)
	require.NoError(t, err)

	// Verify version.
	assert.Equal(t, 2, header.Version)

	// Verify params.
	assert.Equal(t, 13, header.Params.LogN)
	assert.Equal(t, []int{29, 26, 26, 26, 26, 26}, header.Params.LogQ)
	assert.Equal(t, []int{29, 29}, header.Params.LogP)
	assert.Equal(t, 26, header.Params.LogDefaultScale)
	assert.Equal(t, 8192, header.Params.H)
	assert.Equal(t, "conjugate_invariant", header.Params.RingType)

	// Verify config.
	assert.Equal(t, "hybrid", header.Config.EmbeddingMethod)
	assert.True(t, header.Config.FuseModules)

	// Verify manifest.
	assert.True(t, header.Manifest.NeedsRLK)
	assert.Equal(t, 24, len(header.Manifest.GaloisElements))

	// Verify input level.
	assert.Equal(t, 3, header.InputLevel)

	// Verify graph structure.
	assert.Equal(t, "flatten", header.Graph.Input)
	assert.Equal(t, "fc2", header.Graph.Output)
	assert.Equal(t, 4, len(header.Graph.Nodes))
	assert.Equal(t, 3, len(header.Graph.Edges))

	// Verify node names and ops.
	nodeOps := make(map[string]string)
	for _, n := range header.Graph.Nodes {
		nodeOps[n.Name] = n.Op
	}
	assert.Equal(t, "flatten", nodeOps["flatten"])
	assert.Equal(t, "linear_transform", nodeOps["fc1"])
	assert.Equal(t, "quad", nodeOps["act1"])
	assert.Equal(t, "linear_transform", nodeOps["fc2"])

	// Verify blob count.
	assert.Equal(t, 4, header.BlobCount)
	assert.Equal(t, 4, len(blobs))
}

func TestParseContainerSigmoid(t *testing.T) {
	data, err := os.ReadFile("testdata/sigmoid.orion")
	require.NoError(t, err)

	header, blobs, err := ParseContainer(data)
	require.NoError(t, err)

	assert.Equal(t, 2, header.Version)
	assert.Equal(t, 4, len(header.Graph.Nodes))
	assert.Equal(t, 3, len(header.Graph.Edges))
	assert.Equal(t, 4, header.BlobCount)
	assert.Equal(t, 4, len(blobs))

	// Verify sigmoid model has polynomial node.
	var polyNode *HeaderNode
	for i := range header.Graph.Nodes {
		if header.Graph.Nodes[i].Op == "polynomial" {
			polyNode = &header.Graph.Nodes[i]
			break
		}
	}
	require.NotNil(t, polyNode, "sigmoid model should have a polynomial node")
	assert.Equal(t, "act1", polyNode.Name)
}

func TestParseContainerWrongMagic(t *testing.T) {
	data := make([]byte, 100)
	copy(data, "NOTMAGIC")

	_, _, err := ParseContainer(data)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid magic")
}

func TestParseContainerTruncatedData(t *testing.T) {
	// Too short for even the magic + header length.
	_, _, err := ParseContainer([]byte{0, 1, 2})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "too short")
}

func TestParseContainerTruncatedHeader(t *testing.T) {
	// Valid magic, but header length says 1000 bytes and we only have 20.
	data := make([]byte, 20)
	copy(data, magicV2[:])
	binary.LittleEndian.PutUint32(data[8:12], 1000) // header claims 1000 bytes

	_, _, err := ParseContainer(data)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "too short for header JSON")
}

func TestParseDiagonalBlob(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	header, blobs, err := ParseContainer(data)
	require.NoError(t, err)

	// maxSlots for conjugate_invariant with logn=13 is 2^13 = 8192.
	maxSlots := 1 << header.Params.LogN

	// Find fc1 node's diag_0_0 blob ref.
	var fc1 *HeaderNode
	for i := range header.Graph.Nodes {
		if header.Graph.Nodes[i].Name == "fc1" {
			fc1 = &header.Graph.Nodes[i]
			break
		}
	}
	require.NotNil(t, fc1)
	diagIdx, ok := fc1.BlobRefs["diag_0_0"]
	require.True(t, ok)

	diags, err := ParseDiagonalBlob(blobs[diagIdx], maxSlots)
	require.NoError(t, err)

	// Verify we have at least one diagonal.
	assert.Greater(t, len(diags), 0)

	// Verify indices are sorted.
	indices := make([]int, 0, len(diags))
	for idx := range diags {
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	for i := 1; i < len(indices); i++ {
		assert.Less(t, indices[i-1], indices[i], "diagonal indices should be strictly ascending")
	}

	// Verify each diagonal has maxSlots values.
	for idx, vals := range diags {
		assert.Equal(t, maxSlots, len(vals), "diagonal %d should have %d values", idx, maxSlots)
	}
}

func TestParseBiasBlob(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	header, blobs, err := ParseContainer(data)
	require.NoError(t, err)

	maxSlots := 1 << header.Params.LogN

	// Find fc1 node's bias blob ref.
	var fc1 *HeaderNode
	for i := range header.Graph.Nodes {
		if header.Graph.Nodes[i].Name == "fc1" {
			fc1 = &header.Graph.Nodes[i]
			break
		}
	}
	require.NotNil(t, fc1)
	biasIdx, ok := fc1.BlobRefs["bias"]
	require.True(t, ok)

	bias, err := ParseBiasBlob(blobs[biasIdx], maxSlots)
	require.NoError(t, err)

	assert.Equal(t, maxSlots, len(bias))

	// Bias should have some non-zero values (it's a trained network).
	hasNonZero := false
	for _, v := range bias {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "bias should have non-zero values")
}

func TestParseBiasBlobSizeMismatch(t *testing.T) {
	_, err := ParseBiasBlob([]byte{0, 1, 2}, 8192)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "size mismatch")
}

func TestParseDiagonalBlobTooShort(t *testing.T) {
	_, err := ParseDiagonalBlob([]byte{0, 1}, 8192)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "too short")
}

func TestParseLinearTransformConfig(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	header, _, err := ParseContainer(data)
	require.NoError(t, err)

	// Find fc1 node.
	var fc1 *HeaderNode
	for i := range header.Graph.Nodes {
		if header.Graph.Nodes[i].Name == "fc1" {
			fc1 = &header.Graph.Nodes[i]
			break
		}
	}
	require.NotNil(t, fc1)

	cfg, err := parseLinearTransformConfig(fc1.Config)
	require.NoError(t, err)

	assert.Equal(t, 2.0, cfg.BSGSRatio)
	assert.Equal(t, 8, cfg.OutputRotations)

	// Also test fc2.
	var fc2 *HeaderNode
	for i := range header.Graph.Nodes {
		if header.Graph.Nodes[i].Name == "fc2" {
			fc2 = &header.Graph.Nodes[i]
			break
		}
	}
	require.NotNil(t, fc2)

	cfg2, err := parseLinearTransformConfig(fc2.Config)
	require.NoError(t, err)

	assert.Equal(t, 2.0, cfg2.BSGSRatio)
	assert.Equal(t, 0, cfg2.OutputRotations)
}

func TestParsePolynomialConfig(t *testing.T) {
	data, err := os.ReadFile("testdata/sigmoid.orion")
	require.NoError(t, err)

	header, _, err := ParseContainer(data)
	require.NoError(t, err)

	// Find polynomial node.
	var polyNode *HeaderNode
	for i := range header.Graph.Nodes {
		if header.Graph.Nodes[i].Op == "polynomial" {
			polyNode = &header.Graph.Nodes[i]
			break
		}
	}
	require.NotNil(t, polyNode)

	cfg, err := parsePolynomialConfig(polyNode.Config)
	require.NoError(t, err)

	assert.Equal(t, "chebyshev", cfg.Basis)
	assert.Greater(t, len(cfg.Coeffs), 0, "coefficients should not be empty")
	assert.Equal(t, 8, len(cfg.Coeffs), "degree-7 Chebyshev should have 8 coefficients")
	assert.NotZero(t, cfg.Prescale)
}

func TestParseLinearTransformConfigInvalid(t *testing.T) {
	_, err := parseLinearTransformConfig(json.RawMessage(`{invalid`))
	assert.Error(t, err)
}

func TestParsePolynomialConfigInvalid(t *testing.T) {
	_, err := parsePolynomialConfig(json.RawMessage(`{invalid`))
	assert.Error(t, err)
}

func TestParseBootstrapConfigInvalid(t *testing.T) {
	_, err := parseBootstrapConfig(json.RawMessage(`{invalid`))
	assert.Error(t, err)
}
