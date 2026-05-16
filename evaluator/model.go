package evaluator

import (
	"fmt"
	"math"
	"runtime"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/lintrans"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"

	orion "github.com/butvinm/orion/v2"
)

// Model holds a parsed compiled model with eagerly-encoded linear-transform
// diagonals. It is immutable after LoadModel() and safe to share across
// goroutines.
//
// Memory contract: linear-transform diagonals are CKKS-encoded once at load
// time and held resident on the Model. This trades steady resident memory
// (~7 GB at logn=15, ~13 GB at logn=16 for C3AE-class models — measured in
// docs/plans/20260516-pre-encode-lintrans.md) for elimination of the
// per-request transient spike inside Forward() (previously ~85 GB for C3AE
// conv2 at logn=15 from `lintrans.Encode` churn in `embedDouble`).
//
// Error contract: malformed `diag_*` blobs surface as LoadModel errors with
// node name + (row, col) context, not as deferred Forward errors. This is a
// fail-fast guarantee.
type Model struct {
	header      *CompiledHeader
	clientParam orion.Params // cached for ClientParams()
	params      ckks.Parameters
	graph       *Graph
	rawBlobs    [][]byte                                 // raw blob data (sub-slices of input, no copy)
	biases      map[string][]*rlwe.Plaintext             // node -> per-output-CT biases
	polys       map[string]bignum.Polynomial             // node -> polynomial
	ltConfigs   map[string]*LinearTransformConfig        // node -> parsed LT config
	polyConfigs map[string]*PolynomialConfig             // node -> parsed poly config
	preparedLTs map[string][][]lintrans.LinearTransformation // node -> [col][row] pre-encoded LTs
}

// LoadModel parses a .orion v2 file, stores raw blob data, and CKKS-encodes
// biases, polynomials, and all linear-transform diagonals at load time. The
// returned Model is immutable and shareable across goroutines. See Model's
// doc comment for the memory/error trade-off.
func LoadModel(data []byte) (*Model, error) {
	// 1. Parse container.
	header, blobs, err := ParseContainer(data)
	if err != nil {
		return nil, fmt.Errorf("parsing container: %w", err)
	}

	// 2. Validate format version.
	if header.Version != 2 {
		return nil, fmt.Errorf("unsupported format version %d (expected 2)", header.Version)
	}

	// 3. Convert header params to CKKS parameters.
	p := headerToParams(header)
	ckksParams, err := p.NewCKKSParameters()
	if err != nil {
		return nil, fmt.Errorf("creating CKKS parameters: %w", err)
	}

	// 4. Create temporary encoder for encoding diagonals and biases.
	enc := ckks.NewEncoder(ckksParams)

	// 5. Build computation graph.
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
		rawBlobs:    blobs,
		biases:      make(map[string][]*rlwe.Plaintext),
		polys:       make(map[string]bignum.Polynomial),
		ltConfigs:   make(map[string]*LinearTransformConfig),
		polyConfigs: make(map[string]*PolynomialConfig),
		preparedLTs: make(map[string][][]lintrans.LinearTransformation),
	}

	// Process each node: parse configs, encode biases (small), load polynomials,
	// and eagerly CKKS-encode all linear-transform diagonals.
	for _, node := range graph.Nodes {
		switch node.Op {
		case "linear_transform":
			if err := m.loadLinearTransformMetadata(node, blobs, ckksParams, enc, maxSlots); err != nil {
				return nil, fmt.Errorf("loading linear_transform %q: %w", node.Name, err)
			}
			// Reclaim per-node embedDouble transients before encoding the next
			// LT node — without this, transients can stack across nodes and
			// load-time peak RSS becomes the sum rather than the max.
			runtime.GC()
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

// loadLinearTransformMetadata parses config, encodes biases, and eagerly
// CKKS-encodes all linear-transform diagonals for a single linear_transform
// node. Diagonals are stored on the Model as preparedLTs[node.Name][col][row]
// — the exact shape that lintrans.Evaluator.EvaluateManyNew consumes per col.
//
// Encoding here moves the ~85 GB transient `embedDouble` spike (issue #21) out
// of the per-request Forward() path. Errors carry node name + (row, col)
// context for fail-fast diagnostics.
func (m *Model) loadLinearTransformMetadata(node *Node, blobs [][]byte, ckksParams ckks.Parameters, enc *ckks.Encoder, maxSlots int) error {
	cfg, err := parseLinearTransformConfig(node.ConfigRaw)
	if err != nil {
		return fmt.Errorf("parsing config: %w", err)
	}

	// Validate node level is within the moduli chain.
	if node.Level < 0 || node.Level > ckksParams.MaxLevel() {
		return fmt.Errorf("node level %d out of range [0, %d]", node.Level, ckksParams.MaxLevel())
	}

	// Validate BSGS ratio is positive.
	if cfg.BSGSRatio <= 0 {
		return fmt.Errorf("bsgs_ratio must be positive, got %f", cfg.BSGSRatio)
	}

	// Validate NumInputCTs/NumOutputCTs are positive. Zero or negative values
	// from the config default to 1 (preserves prior behavior).
	if cfg.NumInputCTs <= 0 {
		cfg.NumInputCTs = 1
	}
	if cfg.NumOutputCTs <= 0 {
		cfg.NumOutputCTs = 1
	}

	// Validate blob refs point to valid indices (diagonal blobs are parsed
	// below; this catches missing/oob refs early).
	for ref, blobIdx := range node.BlobRefs {
		if ref == "bias" || len(ref) > 5 && ref[:5] == "bias_" {
			continue
		}
		if blobIdx < 0 || blobIdx >= len(blobs) {
			return fmt.Errorf("blob ref %q index %d out of range (have %d blobs)", ref, blobIdx, len(blobs))
		}
	}

	m.ltConfigs[node.Name] = cfg

	// Pre-encode diagonals. Shape: [NumInputCTs][NumOutputCTs] —
	// EvaluateManyNew consumes one inner slice per input col.
	ltParamsTemplate := lintrans.Parameters{
		LevelQ:                    node.Level,
		LevelP:                    ckksParams.MaxLevelP(),
		Scale:                     rlwe.NewScale(ckksParams.Q()[node.Level]),
		LogDimensions:             ring.Dimensions{Rows: 0, Cols: ckksParams.LogMaxSlots()},
		LogBabyStepGiantStepRatio: int(math.Log2(cfg.BSGSRatio)),
	}

	preparedCols := make([][]lintrans.LinearTransformation, cfg.NumInputCTs)
	for col := 0; col < cfg.NumInputCTs; col++ {
		rowLTs := make([]lintrans.LinearTransformation, cfg.NumOutputCTs)
		for row := 0; row < cfg.NumOutputCTs; row++ {
			ref := fmt.Sprintf("diag_%d_%d", row, col)
			blobIdx, ok := node.BlobRefs[ref]
			if !ok {
				return fmt.Errorf("missing blob ref %q (row=%d, col=%d)", ref, row, col)
			}
			if blobIdx < 0 || blobIdx >= len(blobs) {
				return fmt.Errorf("blob ref %q (row=%d, col=%d) index %d out of range (have %d blobs)", ref, row, col, blobIdx, len(blobs))
			}

			diagMap, err := ParseDiagonalBlob(blobs[blobIdx], maxSlots)
			if err != nil {
				return fmt.Errorf("parsing diagonal blob %q (row=%d, col=%d): %w", ref, row, col, err)
			}

			diagonals := lintrans.Diagonals[float64](diagMap)
			ltparams := ltParamsTemplate
			ltparams.DiagonalsIndexList = diagonals.DiagonalsIndexList()

			lt := lintrans.NewTransformation(ckksParams, ltparams)
			if err := lintrans.Encode(enc, diagonals, lt); err != nil {
				return fmt.Errorf("encoding linear transform %q (row=%d, col=%d): %w", ref, row, col, err)
			}
			rowLTs[row] = lt

			// Reclaim per-diagonal embedDouble transients (BRedConstants /
			// ModuliChain / NewPoly slices that Lattigo allocates inside
			// each lintrans.Encode call — see issue #21). Without per-
			// diagonal GC, transients accumulate within one node's encode
			// loop and load-time peak RSS scales with N_diagonals × per-
			// diagonal transient. On logn=16 conv2 (~4379 diagonals) the
			// accumulated transient OOMs a 128 GB box.
			diagMap = nil
			runtime.GC()
		}
		preparedCols[col] = rowLTs
	}
	m.preparedLTs[node.Name] = preparedCols

	// Encode per-row biases (biases are small).
	biasLevel := node.Level - node.Depth
	if biasLevel < 0 {
		return fmt.Errorf("bias level %d is negative (node level=%d, depth=%d)", biasLevel, node.Level, node.Depth)
	}

	var biasPts []*rlwe.Plaintext
	for row := 0; row < cfg.NumOutputCTs; row++ {
		ref := fmt.Sprintf("bias_%d", row)
		biasIdx, ok := node.BlobRefs[ref]
		if !ok {
			continue
		}
		if biasIdx < 0 || biasIdx >= len(blobs) {
			return fmt.Errorf("bias blob %q index %d out of range (have %d blobs)", ref, biasIdx, len(blobs))
		}

		biasVec, err := ParseBiasBlob(blobs[biasIdx], maxSlots)
		if err != nil {
			return fmt.Errorf("parsing bias blob %q: %w", ref, err)
		}

		pt := ckks.NewPlaintext(ckksParams, biasLevel)
		pt.Scale = rlwe.NewScale(ckksParams.DefaultScale())

		if err := enc.Encode(biasVec, pt); err != nil {
			return fmt.Errorf("encoding bias %q: %w", ref, err)
		}

		// Grow slice if needed
		for len(biasPts) <= row {
			biasPts = append(biasPts, nil)
		}
		biasPts[row] = pt
	}

	if len(biasPts) > 0 {
		m.biases[node.Name] = biasPts
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

// ParseClientParams parses ONLY the header of a .orion v2 file and returns
// the CKKS parameters, key manifest, and input level — the exact tuple
// Model.ClientParams() returns, but WITHOUT loading biases, polynomials, or
// (after Task 1) eagerly encoding linear-transform diagonals.
//
// Use this from any caller that only needs params/manifest/inputLevel —
// keygen, encrypt, decrypt, or any client-side scaffolding. The full
// LoadModel path is reserved for the inference side (Evaluator.Forward),
// which is the only path that needs the encoded LTs and incurs the
// ~7 GB / ~13 GB resident cost at logn=15 / logn=16.
func ParseClientParams(data []byte) (orion.Params, orion.Manifest, int, error) {
	header, _, err := ParseContainer(data)
	if err != nil {
		return orion.Params{}, orion.Manifest{}, 0, fmt.Errorf("parsing container: %w", err)
	}
	if header.Version != 2 {
		return orion.Params{}, orion.Manifest{}, 0, fmt.Errorf("unsupported format version %d (expected 2)", header.Version)
	}

	params := headerToParams(header)

	galoisElements := make([]uint64, len(header.Manifest.GaloisElements))
	for i, ge := range header.Manifest.GaloisElements {
		galoisElements[i] = uint64(ge)
	}
	manifest := orion.Manifest{
		GaloisElements: galoisElements,
		BootstrapSlots: header.Manifest.BootstrapSlots,
		BootLogP:       header.Manifest.BootLogP,
		BtpLogN:        header.Manifest.BtpLogN,
		NeedsRLK:       header.Manifest.NeedsRLK,
	}

	return params, manifest, header.InputLevel, nil
}

// ClientParams returns the CKKS parameters, key manifest, and input level
// needed by a client to generate keys and encrypt input.
//
// If you only need the client params (not a ready-to-infer Model), prefer
// ParseClientParams(data) — it avoids the LoadModel cost.
func (m *Model) ClientParams() (orion.Params, orion.Manifest, int) {
	// Convert galois elements from []int to []uint64.
	galoisElements := make([]uint64, len(m.header.Manifest.GaloisElements))
	for i, ge := range m.header.Manifest.GaloisElements {
		galoisElements[i] = uint64(ge)
	}

	manifest := orion.Manifest{
		GaloisElements: galoisElements,
		BootstrapSlots: m.header.Manifest.BootstrapSlots,
		BootLogP:       m.header.Manifest.BootLogP,
		BtpLogN:        m.header.Manifest.BtpLogN,
		NeedsRLK:       m.header.Manifest.NeedsRLK,
	}

	return m.clientParam, manifest, m.header.InputLevel
}

// headerToParams converts a CompiledHeader to orion.Params.
func headerToParams(header *CompiledHeader) orion.Params {
	return orion.Params{
		LogN:     header.Params.LogN,
		LogQ:     header.Params.LogQ,
		LogP:     header.Params.LogP,
		LogDefaultScale: header.Params.LogDefaultScale,
		H:        header.Params.H,
		RingType: header.Params.RingType,
		BootLogP: header.Params.BootLogP,
		BtpLogN:  header.Params.BtpLogN,
	}
}
