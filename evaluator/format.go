package evaluator

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
)

// Magic bytes for the .orion v2 binary container format.
var magicV2 = [8]byte{'O', 'R', 'I', 'O', 'N', 0x00, 0x02, 0x00}

// CompiledHeader is the top-level JSON metadata in a .orion v2 file.
type CompiledHeader struct {
	Version    int            `json:"version"`
	Params     HeaderParams   `json:"params"`
	Config     HeaderConfig   `json:"config"`
	Manifest   HeaderManifest `json:"manifest"`
	InputLevel int            `json:"input_level"`
	Cost       HeaderCost     `json:"cost"`
	Graph      HeaderGraph    `json:"graph"`
	BlobCount  int            `json:"blob_count"`
}

// HeaderParams mirrors Python's CKKSParams.
type HeaderParams struct {
	LogN     int    `json:"logn"`
	LogQ     []int  `json:"logq"`
	LogP     []int  `json:"logp"`
	LogScale int    `json:"logscale"`
	H        int    `json:"h"`
	RingType string `json:"ring_type"`
	BootLogP []int  `json:"boot_logp"`
}

// HeaderConfig mirrors Python's CompilerConfig.
type HeaderConfig struct {
	Margin          int    `json:"margin"`
	EmbeddingMethod string `json:"embedding_method"`
	FuseModules     bool   `json:"fuse_modules"`
}

// HeaderManifest mirrors Python's KeyManifest.
type HeaderManifest struct {
	GaloisElements []int `json:"galois_elements"`
	BootstrapSlots []int `json:"bootstrap_slots"`
	BootLogP       []int `json:"boot_logp"`
	NeedsRLK       bool  `json:"needs_rlk"`
}

// HeaderCost mirrors Python's CostProfile.
type HeaderCost struct {
	BootstrapCount    int `json:"bootstrap_count"`
	GaloisKeyCount    int `json:"galois_key_count"`
	BootstrapKeyCount int `json:"bootstrap_key_count"`
}

// HeaderGraph contains the computation graph (nodes + edges).
type HeaderGraph struct {
	Input  string       `json:"input"`
	Output string       `json:"output"`
	Nodes  []HeaderNode `json:"nodes"`
	Edges  []HeaderEdge `json:"edges"`
}

// HeaderNode is a single node in the computation graph.
type HeaderNode struct {
	Name     string            `json:"name"`
	Op       string            `json:"op"`
	Level    int               `json:"level"`
	Depth    int               `json:"depth"`
	Shape    map[string][]int  `json:"shape"`
	Config   json.RawMessage   `json:"config"`
	BlobRefs map[string]int    `json:"blob_refs"`
}

// HeaderEdge is a directed edge in the computation graph.
type HeaderEdge struct {
	Src string `json:"src"`
	Dst string `json:"dst"`
}

// LinearTransformConfig holds parsed config for linear_transform nodes.
type LinearTransformConfig struct {
	BSGSRatio       float64 `json:"bsgs_ratio"`
	OutputRotations int     `json:"output_rotations"`
}

// PolynomialConfig holds parsed config for polynomial nodes.
type PolynomialConfig struct {
	Coeffs    []float64 `json:"coeffs"`
	Basis     string    `json:"basis"`
	Prescale  float64   `json:"prescale"`
	Postscale float64   `json:"postscale"`
	Constant  float64   `json:"constant"`
}

// BootstrapConfig holds parsed config for bootstrap nodes.
type BootstrapConfig struct {
	InputLevel int     `json:"input_level"`
	InputMin   float64 `json:"input_min"`
	InputMax   float64 `json:"input_max"`
	Prescale   float64 `json:"prescale"`
	Postscale  float64 `json:"postscale"`
	Constant   float64 `json:"constant"`
	Slots      int     `json:"slots"`
}

// ParseContainer parses a .orion v2 binary container, returning the header and raw blobs.
func ParseContainer(data []byte) (*CompiledHeader, [][]byte, error) {
	if len(data) < 12 {
		return nil, nil, errors.New("data too short for header")
	}

	// Verify magic bytes.
	var magic [8]byte
	copy(magic[:], data[:8])
	if magic != magicV2 {
		return nil, nil, fmt.Errorf("invalid magic: expected %v, got %v", magicV2, magic)
	}

	// Parse header length.
	headerLen := binary.LittleEndian.Uint32(data[8:12])
	offset := uint64(12)

	if uint64(len(data)) < offset+uint64(headerLen) {
		return nil, nil, fmt.Errorf("data too short for header JSON: need %d bytes at offset %d, have %d", headerLen, offset, len(data))
	}

	// Parse JSON header.
	var header CompiledHeader
	if err := json.Unmarshal(data[offset:offset+uint64(headerLen)], &header); err != nil {
		return nil, nil, fmt.Errorf("failed to parse header JSON: %w", err)
	}
	offset += uint64(headerLen)

	// Parse blob count.
	if uint64(len(data)) < offset+4 {
		return nil, nil, errors.New("data too short for blob count")
	}
	blobCount := binary.LittleEndian.Uint32(data[offset : offset+4])
	offset += 4

	if int(blobCount) != header.BlobCount {
		return nil, nil, fmt.Errorf("blob count mismatch: header says %d, container has %d", header.BlobCount, blobCount)
	}

	// Parse blobs.
	blobs := make([][]byte, 0, blobCount)
	for i := 0; i < int(blobCount); i++ {
		if uint64(len(data)) < offset+8 {
			return nil, nil, fmt.Errorf("data too short for blob %d length", i)
		}
		blobLen := binary.LittleEndian.Uint64(data[offset : offset+8])
		offset += 8

		if uint64(len(data)) < offset+blobLen {
			return nil, nil, fmt.Errorf("data too short for blob %d data: need %d bytes, have %d", i, blobLen, uint64(len(data))-offset)
		}
		blobs = append(blobs, data[offset:offset+blobLen])
		offset += blobLen
	}

	return &header, blobs, nil
}

// ParseDiagonalBlob parses a raw diagonal blob into a map of diagonal index to float64 values.
func ParseDiagonalBlob(data []byte, maxSlots int) (map[int][]float64, error) {
	if len(data) < 4 {
		return nil, errors.New("diagonal blob too short for num_diags")
	}

	numDiags := int(binary.LittleEndian.Uint32(data[:4]))
	offset := 4

	// Read diagonal indices (int32 LE).
	indicesSize := numDiags * 4
	if len(data) < offset+indicesSize {
		return nil, fmt.Errorf("diagonal blob too short for indices: need %d bytes at offset %d, have %d", indicesSize, offset, len(data))
	}

	indices := make([]int, numDiags)
	for i := 0; i < numDiags; i++ {
		indices[i] = int(int32(binary.LittleEndian.Uint32(data[offset+i*4 : offset+i*4+4])))
	}
	offset += indicesSize

	// Read values (float64 LE), numDiags * maxSlots values.
	valuesSize := numDiags * maxSlots * 8
	if len(data) < offset+valuesSize {
		return nil, fmt.Errorf("diagonal blob too short for values: need %d bytes at offset %d, have %d", valuesSize, offset, len(data))
	}

	result := make(map[int][]float64, numDiags)
	for i := 0; i < numDiags; i++ {
		vals := make([]float64, maxSlots)
		for j := 0; j < maxSlots; j++ {
			pos := offset + (i*maxSlots+j)*8
			vals[j] = math.Float64frombits(binary.LittleEndian.Uint64(data[pos : pos+8]))
		}
		result[indices[i]] = vals
	}

	return result, nil
}

// ParseBiasBlob parses a raw bias blob into a float64 slice.
func ParseBiasBlob(data []byte, maxSlots int) ([]float64, error) {
	expected := maxSlots * 8
	if len(data) != expected {
		return nil, fmt.Errorf("bias blob size mismatch: expected %d bytes, got %d", expected, len(data))
	}

	result := make([]float64, maxSlots)
	for i := 0; i < maxSlots; i++ {
		result[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8 : i*8+8]))
	}
	return result, nil
}

// parseLinearTransformConfig parses the config JSON for a linear_transform node.
func parseLinearTransformConfig(raw json.RawMessage) (*LinearTransformConfig, error) {
	var cfg LinearTransformConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse linear_transform config: %w", err)
	}
	return &cfg, nil
}

// parsePolynomialConfig parses the config JSON for a polynomial node.
// Prescale and Postscale default to 1.0 if omitted (Go zero-value 0.0 would
// destroy ciphertext data via multiplication by zero).
func parsePolynomialConfig(raw json.RawMessage) (*PolynomialConfig, error) {
	cfg := PolynomialConfig{
		Prescale:  1.0,
		Postscale: 1.0,
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse polynomial config: %w", err)
	}
	return &cfg, nil
}

// parseBootstrapConfig parses the config JSON for a bootstrap node.
// Prescale and Postscale default to 1.0 if omitted (Go zero-value 0.0 would
// destroy ciphertext data via multiplication by zero).
func parseBootstrapConfig(raw json.RawMessage) (*BootstrapConfig, error) {
	cfg := BootstrapConfig{Prescale: 1.0, Postscale: 1.0}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse bootstrap config: %w", err)
	}
	return &cfg, nil
}
