package evaluator

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"strings"
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

	// Verify 2 linear transform configs (fc1 and fc2).
	assert.Equal(t, 2, len(model.ltConfigs))
	assert.Contains(t, model.ltConfigs, "fc1")
	assert.Contains(t, model.ltConfigs, "fc2")

	// Verify raw blobs are stored (diagonals are NOT pre-encoded).
	assert.Greater(t, len(model.rawBlobs), 0)

	// Verify diagonal blob refs point to valid raw blobs.
	for _, node := range model.graph.Nodes {
		if node.Op == "linear_transform" {
			for ref, blobIdx := range node.BlobRefs {
				if ref == "bias" || strings.HasPrefix(ref, "bias_") {
					continue
				}
				assert.Less(t, blobIdx, len(model.rawBlobs), "blob ref %q index out of range", ref)
				// Verify the blob is parseable.
				maxSlots := model.params.MaxSlots()
				diagMap, err := ParseDiagonalBlob(model.rawBlobs[blobIdx], maxSlots)
				assert.NoError(t, err, "parsing diagonal blob %q", ref)
				assert.Greater(t, len(diagMap), 0, "diagonal blob %q should have diagonals", ref)
			}
		}
	}

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

	// Verify 2 LT configs still present (diagonals not pre-encoded).
	assert.Equal(t, 2, len(model.ltConfigs))

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

func TestLoadModelWithBtpLogN(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	// Parse the original container to get the header JSON.
	header, blobs, err := ParseContainer(data)
	require.NoError(t, err)

	// Set btp_logn in the header params and manifest.
	header.Params.BtpLogN = 13
	header.Manifest.BtpLogN = 13

	// Re-serialize into a .orion binary.
	newData := rebuildContainer(t, header, blobs)

	model, err := LoadModel(newData)
	require.NoError(t, err)

	params, manifest, _ := model.ClientParams()

	// Verify btp_logn is correctly parsed.
	assert.Equal(t, 13, params.BtpLogN)
	assert.Equal(t, 13, manifest.BtpLogN)
}

func TestLoadModelWithoutBtpLogN(t *testing.T) {
	// Existing models without btp_logn should still load fine.
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	model, err := LoadModel(data)
	require.NoError(t, err)

	params, manifest, _ := model.ClientParams()

	// BtpLogN should be zero (not set in the original model).
	assert.Equal(t, 0, params.BtpLogN)
	assert.Equal(t, 0, manifest.BtpLogN)
}

// rebuildContainer serializes a header and blobs back into .orion v2 format.
func rebuildContainer(t *testing.T, header *CompiledHeader, blobs [][]byte) []byte {
	t.Helper()

	headerJSON, err := json.Marshal(header)
	require.NoError(t, err)

	// magic (8) + headerLen (4) + headerJSON + blobCount (4) + blobs
	size := 8 + 4 + len(headerJSON) + 4
	for _, b := range blobs {
		size += 8 + len(b)
	}

	buf := make([]byte, size)
	copy(buf[:8], magicV2[:])
	binary.LittleEndian.PutUint32(buf[8:12], uint32(len(headerJSON)))
	copy(buf[12:12+len(headerJSON)], headerJSON)

	offset := 12 + len(headerJSON)
	binary.LittleEndian.PutUint32(buf[offset:offset+4], uint32(len(blobs)))
	offset += 4

	for _, b := range blobs {
		binary.LittleEndian.PutUint64(buf[offset:offset+8], uint64(len(b)))
		offset += 8
		copy(buf[offset:offset+len(b)], b)
		offset += len(b)
	}

	return buf
}

func TestLoadModelInvalidData(t *testing.T) {
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
