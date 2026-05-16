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

	// Verify raw blobs are stored.
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

	// Verify 2 LT configs still present.
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

func TestParseClientParamsMatchesLoadModel(t *testing.T) {
	fixtures := []string{
		"testdata/mlp.orion",
		"testdata/conv2d.orion",
		"testdata/sigmoid.orion",
		"testdata/bootstrap_mlp.orion",
	}
	for _, fixture := range fixtures {
		t.Run(fixture, func(t *testing.T) {
			data, err := os.ReadFile(fixture)
			require.NoError(t, err)

			lightParams, lightManifest, lightInputLevel, err := ParseClientParams(data)
			require.NoError(t, err)

			model, err := LoadModel(data)
			require.NoError(t, err)
			heavyParams, heavyManifest, heavyInputLevel := model.ClientParams()

			assert.Equal(t, heavyParams, lightParams, "params mismatch")
			assert.Equal(t, heavyManifest, lightManifest, "manifest mismatch")
			assert.Equal(t, heavyInputLevel, lightInputLevel, "inputLevel mismatch")
		})
	}
}

func TestParseClientParamsSkipsLTEncoding(t *testing.T) {
	// ParseClientParams must NOT trigger eager LT encoding — that's the whole
	// point of this code path. Caller `bench keygen` OOMs at logn=16 if it
	// goes through LoadModel because the per-node embedDouble transient
	// exceeds 128 GB. We can't measure RSS directly in a unit test, but we
	// can assert that no Model is built and no preparedLTs map is allocated.
	// The contract is: ParseClientParams returns three values + error and
	// allocates no Lattigo handles. (If it did, the smoke test wouldn't
	// even pass at logn=15 in CI.)
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)
	_, _, _, err = ParseClientParams(data)
	require.NoError(t, err)
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

// TestPreparedLTsCachePresence verifies that LoadModel eagerly encodes all
// linear-transform diagonals into preparedLTs with the correct [col][row]
// shape and the correct LevelQ. This is the cache-presence half of Task 3.
func TestPreparedLTsCachePresence(t *testing.T) {
	for _, fname := range []string{"mlp.orion", "conv2d.orion", "sigmoid.orion", "sigmoid_unfused.orion"} {
		t.Run(fname, func(t *testing.T) {
			data, err := os.ReadFile("testdata/" + fname)
			require.NoError(t, err)

			model, err := LoadModel(data)
			require.NoError(t, err)

			require.NotNil(t, model.preparedLTs)

			ltNodes := 0
			for _, node := range model.graph.Nodes {
				if node.Op != "linear_transform" {
					continue
				}
				ltNodes++
				cfg := model.ltConfigs[node.Name]
				require.NotNil(t, cfg, "ltConfigs[%q] missing", node.Name)

				preparedCols, ok := model.preparedLTs[node.Name]
				require.True(t, ok, "preparedLTs[%q] missing", node.Name)

				// Shape: [NumInputCTs][NumOutputCTs].
				assert.Equal(t, cfg.NumInputCTs, len(preparedCols),
					"preparedLTs[%q] outer length must equal NumInputCTs", node.Name)
				for col, rowLTs := range preparedCols {
					assert.Equal(t, cfg.NumOutputCTs, len(rowLTs),
						"preparedLTs[%q][%d] inner length must equal NumOutputCTs", node.Name, col)
					for row, lt := range rowLTs {
						assert.Equal(t, node.Level, lt.LevelQ,
							"preparedLTs[%q][%d][%d] LevelQ should match node.Level", node.Name, col, row)
					}
				}
			}
			assert.Greater(t, ltNodes, 0, "fixture %s should contain at least one linear_transform node", fname)
			assert.Equal(t, ltNodes, len(model.preparedLTs),
				"preparedLTs should have one entry per linear_transform node")
		})
	}
}

// TestPreparedLTsHighLevelEncoding exercises the bootstrap-adjacent path where
// node.Level == params.MaxLevel(). The bootstrap_mlp fixture has fc1 at the
// top of its (logq=4) chain, which is the only fixture in the testdata set
// that reaches MaxLevel for any LT node.
func TestPreparedLTsHighLevelEncoding(t *testing.T) {
	data, err := os.ReadFile("testdata/bootstrap_mlp.orion")
	require.NoError(t, err)

	model, err := LoadModel(data)
	require.NoError(t, err)

	maxLevel := model.params.MaxLevel()
	require.Greater(t, maxLevel, 0)

	sawMax := false
	for _, node := range model.graph.Nodes {
		if node.Op != "linear_transform" {
			continue
		}
		preparedCols, ok := model.preparedLTs[node.Name]
		require.True(t, ok, "preparedLTs[%q] missing", node.Name)
		for _, rowLTs := range preparedCols {
			for _, lt := range rowLTs {
				assert.Equal(t, node.Level, lt.LevelQ)
				if node.Level == maxLevel {
					sawMax = true
				}
			}
		}
	}
	assert.True(t, sawMax,
		"bootstrap_mlp fixture should have at least one LT node at params.MaxLevel(); "+
			"if this fixture changes, pick another high-level model to keep the path covered")
}

// TestLoadModelCorruptedDiagonalBlob verifies that a malformed diag_* blob
// surfaces as a LoadModel error with node name + (row, col) context. The
// per-request Forward path can no longer produce these errors (Task 2).
func TestLoadModelCorruptedDiagonalBlob(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	header, blobs, err := ParseContainer(data)
	require.NoError(t, err)

	// Find fc1's diag_0_0 blob index.
	var diagIdx int = -1
	for _, n := range header.Graph.Nodes {
		if n.Name == "fc1" {
			diagIdx = n.BlobRefs["diag_0_0"]
		}
	}
	require.GreaterOrEqual(t, diagIdx, 0, "fc1.diag_0_0 ref not found")

	// Corrupt: replace the blob with 3 bytes (less than the 4-byte num_diags
	// header that ParseDiagonalBlob requires).
	corruptedBlobs := make([][]byte, len(blobs))
	copy(corruptedBlobs, blobs)
	corruptedBlobs[diagIdx] = []byte{0x01, 0x02, 0x03}

	newData := rebuildContainer(t, header, corruptedBlobs)
	_, err = LoadModel(newData)
	require.Error(t, err)

	msg := err.Error()
	assert.Contains(t, msg, "fc1", "error should mention the failing node name")
	assert.Contains(t, msg, "row=0", "error should mention the failing (row, col)")
	assert.Contains(t, msg, "col=0", "error should mention the failing (row, col)")
}

// TestLoadModelZeroNumCTsDefaultsToOne pins the negative-path-2 contract from
// the plan: zero NumInputCTs / NumOutputCTs defaults to 1 rather than
// erroring. This preserves the pre-Task-1 behavior at lines 117-122 of
// model.go.
func TestLoadModelZeroNumCTsDefaultsToOne(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	header, blobs, err := ParseContainer(data)
	require.NoError(t, err)

	// Rewrite fc1's config to have num_input_cts=0, num_output_cts=0.
	// The 1x1 diag_0_0 / bias_0 blobs already in the fixture satisfy the
	// defaulted-to-1 case.
	for i, n := range header.Graph.Nodes {
		if n.Name == "fc1" {
			var cfg LinearTransformConfig
			require.NoError(t, json.Unmarshal(n.Config, &cfg))
			cfg.NumInputCTs = 0
			cfg.NumOutputCTs = 0
			raw, err := json.Marshal(cfg)
			require.NoError(t, err)
			header.Graph.Nodes[i].Config = raw
			break
		}
	}

	newData := rebuildContainer(t, header, blobs)
	model, err := LoadModel(newData)
	require.NoError(t, err, "zero NumInputCTs/NumOutputCTs should default to 1, not error")

	cfg := model.ltConfigs["fc1"]
	require.NotNil(t, cfg)
	assert.Equal(t, 1, cfg.NumInputCTs)
	assert.Equal(t, 1, cfg.NumOutputCTs)

	preparedCols, ok := model.preparedLTs["fc1"]
	require.True(t, ok)
	assert.Equal(t, 1, len(preparedCols))
	assert.Equal(t, 1, len(preparedCols[0]))
}

// TestLoadModelMissingDiagonalBlobRef verifies that a missing diag_{row}_{col}
// ref surfaces at LoadModel (not Forward) with full (row, col) context.
func TestLoadModelMissingDiagonalBlobRef(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	header, blobs, err := ParseContainer(data)
	require.NoError(t, err)

	// Drop fc1's diag_0_0 ref entirely.
	for _, n := range header.Graph.Nodes {
		if n.Name == "fc1" {
			delete(n.BlobRefs, "diag_0_0")
		}
	}

	newData := rebuildContainer(t, header, blobs)
	_, err = LoadModel(newData)
	require.Error(t, err)

	msg := err.Error()
	assert.Contains(t, msg, "fc1")
	assert.Contains(t, msg, "diag_0_0")
}

// TestLoadModelOutOfRangeNodeLevel guards the level-validation path added in
// loadLinearTransformMetadata.
func TestLoadModelOutOfRangeNodeLevel(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	header, blobs, err := ParseContainer(data)
	require.NoError(t, err)

	// fc1's level is 3 in the fixture; bump it to 999 (clearly OOB).
	for i, n := range header.Graph.Nodes {
		if n.Name == "fc1" {
			header.Graph.Nodes[i].Level = 999
			break
		}
	}

	newData := rebuildContainer(t, header, blobs)
	_, err = LoadModel(newData)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "fc1")
	assert.Contains(t, err.Error(), "node level")
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
