package evaluator

import (
	"fmt"
	"math"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/lintrans"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
	"github.com/baahl-nyu/lattigo/v6/utils/bignum"
	"github.com/baahl-nyu/orion/orionclient"
)

// Model holds a parsed and CKKS-encoded compiled model.
// It is immutable after LoadModel() and safe to share across goroutines.
type Model struct {
	header      *CompiledHeader
	clientParam orionclient.Params // cached for ClientParams()
	params      ckks.Parameters
	graph       *Graph
	transforms  map[string]map[string]lintrans.LinearTransformation // node -> ref -> LT
	biases      map[string]*rlwe.Plaintext                          // node -> bias
	polys       map[string]bignum.Polynomial                        // node -> polynomial
	ltConfigs   map[string]*LinearTransformConfig                   // node -> parsed LT config
	polyConfigs map[string]*PolynomialConfig                        // node -> parsed poly config
}

// LoadModel parses a .orion v2 file and CKKS-encodes diagonals, biases,
// and polynomials at load time. The returned Model is immutable.
func LoadModel(data []byte) (*Model, error) {
	// 1. Parse container.
	header, blobs, err := ParseContainer(data)
	if err != nil {
		return nil, fmt.Errorf("parsing container: %w", err)
	}

	// 2. Convert header params to CKKS parameters.
	p := headerToParams(header)
	ckksParams, err := p.NewCKKSParameters()
	if err != nil {
		return nil, fmt.Errorf("creating CKKS parameters: %w", err)
	}

	// 3. Create temporary encoder for encoding diagonals and biases.
	enc := ckks.NewEncoder(ckksParams)

	// 4. Build computation graph.
	graph, err := buildGraph(header)
	if err != nil {
		return nil, fmt.Errorf("building graph: %w", err)
	}

	maxSlots := ckksParams.MaxSlots()

	m := &Model{
		header:      header,
		clientParam: p,
		params:      ckksParams,
		graph:       graph,
		transforms:  make(map[string]map[string]lintrans.LinearTransformation),
		biases:      make(map[string]*rlwe.Plaintext),
		polys:       make(map[string]bignum.Polynomial),
		ltConfigs:   make(map[string]*LinearTransformConfig),
		polyConfigs: make(map[string]*PolynomialConfig),
	}

	// 5-6. Process each node based on op type.
	for _, node := range graph.Nodes {
		switch node.Op {
		case "linear_transform":
			if err := m.loadLinearTransform(node, blobs, ckksParams, enc, maxSlots); err != nil {
				return nil, fmt.Errorf("loading linear_transform %q: %w", node.Name, err)
			}
		case "polynomial":
			if err := m.loadPolynomial(node); err != nil {
				return nil, fmt.Errorf("loading polynomial %q: %w", node.Name, err)
			}
		case "flatten", "quad", "add", "mult", "bootstrap":
			// No pre-encoding needed for these ops.
		default:
			return nil, fmt.Errorf("unknown op type %q for node %q", node.Op, node.Name)
		}
	}

	return m, nil
}

// loadLinearTransform encodes diagonals and bias for a linear_transform node.
func (m *Model) loadLinearTransform(node *Node, blobs [][]byte, ckksParams ckks.Parameters, enc *ckks.Encoder, maxSlots int) error {
	cfg, err := parseLinearTransformConfig(node.ConfigRaw)
	if err != nil {
		return fmt.Errorf("parsing config: %w", err)
	}

	nodeTransforms := make(map[string]lintrans.LinearTransformation)

	// Validate node level is within the moduli chain.
	if node.Level < 0 || node.Level > ckksParams.MaxLevel() {
		return fmt.Errorf("node level %d out of range [0, %d]", node.Level, ckksParams.MaxLevel())
	}

	// Validate BSGS ratio is positive.
	if cfg.BSGSRatio <= 0 {
		return fmt.Errorf("bsgs_ratio must be positive, got %f", cfg.BSGSRatio)
	}

	for ref, blobIdx := range node.BlobRefs {
		if ref == "bias" {
			continue // handled separately below
		}

		if blobIdx < 0 || blobIdx >= len(blobs) {
			return fmt.Errorf("blob ref %q index %d out of range (have %d blobs)", ref, blobIdx, len(blobs))
		}

		diagMap, err := ParseDiagonalBlob(blobs[blobIdx], maxSlots)
		if err != nil {
			return fmt.Errorf("parsing diagonal blob %q: %w", ref, err)
		}

		// Build Lattigo diagonals (cast — both are map[int][]float64).
		diagonals := lintrans.Diagonals[float64](diagMap)

		ltparams := lintrans.Parameters{
			DiagonalsIndexList:        diagonals.DiagonalsIndexList(),
			LevelQ:                    node.Level,
			LevelP:                    ckksParams.MaxLevelP(),
			Scale:                     rlwe.NewScale(ckksParams.Q()[node.Level]),
			LogDimensions:             ring.Dimensions{Rows: 0, Cols: ckksParams.LogMaxSlots()},
			LogBabyStepGiantStepRatio: int(math.Log2(cfg.BSGSRatio)),
		}

		lt := lintrans.NewTransformation(ckksParams, ltparams)
		if err := lintrans.Encode(enc, diagonals, lt); err != nil {
			return fmt.Errorf("encoding linear transform %q: %w", ref, err)
		}

		nodeTransforms[ref] = lt
	}

	m.transforms[node.Name] = nodeTransforms
	m.ltConfigs[node.Name] = cfg

	// Encode bias if present.
	if biasIdx, ok := node.BlobRefs["bias"]; ok {
		if biasIdx < 0 || biasIdx >= len(blobs) {
			return fmt.Errorf("bias blob index %d out of range (have %d blobs)", biasIdx, len(blobs))
		}

		biasVec, err := ParseBiasBlob(blobs[biasIdx], maxSlots)
		if err != nil {
			return fmt.Errorf("parsing bias blob: %w", err)
		}

		biasLevel := node.Level - node.Depth
		if biasLevel < 0 {
			return fmt.Errorf("bias level %d is negative (node level=%d, depth=%d)", biasLevel, node.Level, node.Depth)
		}
		pt := ckks.NewPlaintext(ckksParams, biasLevel)
		pt.Scale = rlwe.NewScale(ckksParams.DefaultScale())

		if err := enc.Encode(biasVec, pt); err != nil {
			return fmt.Errorf("encoding bias: %w", err)
		}

		m.biases[node.Name] = pt
	}

	return nil
}

// loadPolynomial creates a bignum.Polynomial for a polynomial node.
func (m *Model) loadPolynomial(node *Node) error {
	cfg, err := parsePolynomialConfig(node.ConfigRaw)
	if err != nil {
		return fmt.Errorf("parsing config: %w", err)
	}

	if len(cfg.Coeffs) == 0 {
		return fmt.Errorf("polynomial coefficients are empty")
	}

	var poly bignum.Polynomial
	switch cfg.Basis {
	case "chebyshev":
		poly = bignum.NewPolynomial(bignum.Chebyshev, cfg.Coeffs, [2]float64{-1.0, 1.0})
	case "monomial":
		poly = bignum.NewPolynomial(bignum.Monomial, cfg.Coeffs, nil)
	default:
		return fmt.Errorf("unknown polynomial basis %q", cfg.Basis)
	}

	m.polys[node.Name] = poly
	m.polyConfigs[node.Name] = cfg
	return nil
}

// ClientParams returns the CKKS parameters, key manifest, and input level
// needed by a client to generate keys and encrypt input.
func (m *Model) ClientParams() (orionclient.Params, orionclient.Manifest, int) {
	// Convert galois elements from []int to []uint64.
	galoisElements := make([]uint64, len(m.header.Manifest.GaloisElements))
	for i, ge := range m.header.Manifest.GaloisElements {
		galoisElements[i] = uint64(ge)
	}

	manifest := orionclient.Manifest{
		GaloisElements: galoisElements,
		BootstrapSlots: m.header.Manifest.BootstrapSlots,
		BootLogP:       m.header.Manifest.BootLogP,
		NeedsRLK:       m.header.Manifest.NeedsRLK,
	}

	return m.clientParam, manifest, m.header.InputLevel
}

// headerToParams converts a CompiledHeader to orionclient.Params.
func headerToParams(header *CompiledHeader) orionclient.Params {
	return orionclient.Params{
		LogN:     header.Params.LogN,
		LogQ:     header.Params.LogQ,
		LogP:     header.Params.LogP,
		LogScale: header.Params.LogScale,
		H:        header.Params.H,
		RingType: header.Params.RingType,
		BootLogP: header.Params.BootLogP,
	}
}
