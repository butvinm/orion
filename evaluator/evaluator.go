package evaluator

import (
	"fmt"
	"math"
	"math/bits"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/lintrans"
	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/polynomial"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
	"github.com/baahl-nyu/lattigo/v6/utils"
)

// Evaluator runs FHE inference on a compiled Model by walking the computation graph.
// It is NOT goroutine-safe (Lattigo buffers are reused internally).
type Evaluator struct {
	params   ckks.Parameters
	encoder  *ckks.Encoder
	eval     *ckks.Evaluator
	linEval  *lintrans.Evaluator
	polyEval *polynomial.Evaluator

	// Bootstrap support: keys stored for lazy bootstrapper initialization.
	btpKeys       *bootstrapping.EvaluationKeys
	bootstrappers map[int]*bootstrapping.Evaluator // logSlots -> bootstrapper
}

// NewEvaluatorFromKeySet creates an Evaluator from Lattigo types directly.
// btpKeys may be nil when the model does not use bootstrap.
func NewEvaluatorFromKeySet(ckksParams ckks.Parameters, keys *rlwe.MemEvaluationKeySet, btpKeys *bootstrapping.EvaluationKeys) (*Evaluator, error) {
	eval := ckks.NewEvaluator(ckksParams, keys)
	enc := ckks.NewEncoder(ckksParams)
	polyEval := polynomial.NewEvaluator(ckksParams, eval)
	linEval := lintrans.NewEvaluator(eval)

	return &Evaluator{
		params:        ckksParams,
		encoder:       enc,
		eval:          eval,
		linEval:       linEval,
		polyEval:      polyEval,
		btpKeys:       btpKeys,
		bootstrappers: make(map[int]*bootstrapping.Evaluator),
	}, nil
}

// Close releases evaluator resources.
func (e *Evaluator) Close() {
	e.eval = nil
	e.encoder = nil
	e.linEval = nil
	e.polyEval = nil
	e.btpKeys = nil
	e.bootstrappers = nil
}

// Forward runs FHE inference on the model's computation graph.
// inputs is a list of ciphertexts (one per input CT slot).
// For single-CT models, pass a single-element slice.
func (e *Evaluator) Forward(model *Model, inputs []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	if e.eval == nil {
		return nil, fmt.Errorf("evaluator is closed")
	}
	if model == nil {
		return nil, fmt.Errorf("model is nil")
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("input ciphertext list is empty")
	}
	for i, ct := range inputs {
		if ct == nil {
			return nil, fmt.Errorf("input ciphertext[%d] is nil", i)
		}
	}

	results := make(map[string][]*rlwe.Ciphertext)

	// Make the raw inputs available under a virtual key.
	const virtualInput = "__input__"
	results[virtualInput] = inputs

	// Walk the graph in topological order.
	for _, name := range model.graph.Order {
		node := model.graph.Nodes[name]
		predNames := model.graph.Inputs[name]

		// The graph's input node has no predecessors in the edge list;
		// wire it to the raw input ciphertexts.
		if name == model.graph.Input && len(predNames) == 0 {
			predNames = []string{virtualInput}
		}

		var err error
		var result []*rlwe.Ciphertext

		switch node.Op {
		case "flatten":
			if len(predNames) != 1 {
				return nil, fmt.Errorf("flatten %q: expected 1 input, got %d", name, len(predNames))
			}
			result, err = e.evalFlattenMulti(results[predNames[0]])

		case "quad":
			if len(predNames) != 1 {
				return nil, fmt.Errorf("quad %q: expected 1 input, got %d", name, len(predNames))
			}
			result, err = e.evalQuadMulti(results[predNames[0]])

		case "add":
			if len(predNames) != 2 {
				return nil, fmt.Errorf("add %q: expected 2 inputs, got %d", name, len(predNames))
			}
			result, err = e.evalAddMulti(results[predNames[0]], results[predNames[1]])

		case "mult":
			if len(predNames) != 2 {
				return nil, fmt.Errorf("mult %q: expected 2 inputs, got %d", name, len(predNames))
			}
			result, err = e.evalMultMulti(results[predNames[0]], results[predNames[1]])

		case "linear_transform":
			if len(predNames) != 1 {
				return nil, fmt.Errorf("linear_transform %q: expected 1 input, got %d", name, len(predNames))
			}
			result, err = e.evalLinearTransform(model, node, results[predNames[0]])

		case "polynomial":
			if len(predNames) != 1 {
				return nil, fmt.Errorf("polynomial %q: expected 1 input, got %d", name, len(predNames))
			}
			result, err = e.evalPolynomialMulti(model, node, results[predNames[0]])

		case "bootstrap":
			if len(predNames) != 1 {
				return nil, fmt.Errorf("bootstrap %q: expected 1 input, got %d", name, len(predNames))
			}
			result, err = e.evalBootstrapMulti(model, node, results[predNames[0]])

		default:
			return nil, fmt.Errorf("unknown op %q for node %q", node.Op, name)
		}

		if err != nil {
			return nil, fmt.Errorf("evaluating node %q (op=%s): %w", name, node.Op, err)
		}

		results[name] = result
	}

	out, ok := results[model.graph.Output]
	if !ok {
		return nil, fmt.Errorf("output node %q not found in results", model.graph.Output)
	}
	return out, nil
}

// evalFlattenMulti copies each CT in the list.
func (e *Evaluator) evalFlattenMulti(cts []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	out := make([]*rlwe.Ciphertext, len(cts))
	for i, ct := range cts {
		out[i] = ct.CopyNew()
	}
	return out, nil
}

// evalFlatten is kept for direct unit tests on single CTs.
func (e *Evaluator) evalFlatten(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	return ct.CopyNew(), nil
}

// evalQuadMulti computes ct^2 for each CT in the list.
func (e *Evaluator) evalQuadMulti(cts []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	out := make([]*rlwe.Ciphertext, len(cts))
	for i, ct := range cts {
		var err error
		out[i], err = e.evalQuad(ct)
		if err != nil {
			return nil, fmt.Errorf("ct[%d]: %w", i, err)
		}
	}
	return out, nil
}

// evalQuad computes ct^2 via MulRelin + Rescale.
func (e *Evaluator) evalQuad(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	result, err := e.eval.MulRelinNew(ct, ct)
	if err != nil {
		return nil, fmt.Errorf("MulRelin: %w", err)
	}
	if err := e.eval.Rescale(result, result); err != nil {
		return nil, fmt.Errorf("Rescale: %w", err)
	}
	return result, nil
}

// evalAddMulti adds two CT lists element-wise.
func (e *Evaluator) evalAddMulti(cts0, cts1 []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	if len(cts0) != len(cts1) {
		return nil, fmt.Errorf("add CT count mismatch: %d vs %d", len(cts0), len(cts1))
	}
	out := make([]*rlwe.Ciphertext, len(cts0))
	for i := range cts0 {
		var err error
		out[i], err = e.evalAdd(cts0[i], cts1[i])
		if err != nil {
			return nil, fmt.Errorf("ct[%d]: %w", i, err)
		}
	}
	return out, nil
}

// evalAdd adds two ciphertexts element-wise.
func (e *Evaluator) evalAdd(ct0, ct1 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	result, err := e.eval.AddNew(ct0, ct1)
	if err != nil {
		return nil, fmt.Errorf("Add: %w", err)
	}
	return result, nil
}

// evalMultMulti multiplies two CT lists element-wise.
func (e *Evaluator) evalMultMulti(cts0, cts1 []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	if len(cts0) != len(cts1) {
		return nil, fmt.Errorf("mult CT count mismatch: %d vs %d", len(cts0), len(cts1))
	}
	out := make([]*rlwe.Ciphertext, len(cts0))
	for i := range cts0 {
		var err error
		out[i], err = e.evalMult(cts0[i], cts1[i])
		if err != nil {
			return nil, fmt.Errorf("ct[%d]: %w", i, err)
		}
	}
	return out, nil
}

// evalMult multiplies two ciphertexts with relinearization and rescaling.
func (e *Evaluator) evalMult(ct0, ct1 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	result, err := e.eval.MulRelinNew(ct0, ct1)
	if err != nil {
		return nil, fmt.Errorf("MulRelin: %w", err)
	}
	if err := e.eval.Rescale(result, result); err != nil {
		return nil, fmt.Errorf("Rescale: %w", err)
	}
	return result, nil
}

// evalLinearTransform evaluates a linear transform node with blocked matrix-vector multiply.
// For multi-CT inputs, uses EvaluateManyNew to share BSGS rotations across row blocks.
//
// Diagonals are parsed from raw blobs and CKKS-encoded on demand for each block,
// then discarded after evaluation. This avoids the ~23x memory blowup from
// pre-encoding all diagonals at model load time.
func (e *Evaluator) evalLinearTransform(model *Model, node *Node, inputs []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	cfg, ok := model.ltConfigs[node.Name]
	if !ok {
		return nil, fmt.Errorf("no config found for linear_transform node %q", node.Name)
	}

	maxSlots := model.params.MaxSlots()
	numInputCTs := cfg.NumInputCTs
	numOutputCTs := cfg.NumOutputCTs

	if len(inputs) != numInputCTs {
		return nil, fmt.Errorf("linear_transform %q: expected %d input CTs, got %d", node.Name, numInputCTs, len(inputs))
	}

	outputs := make([]*rlwe.Ciphertext, numOutputCTs)

	for col := 0; col < numInputCTs; col++ {
		// Gather all row LTs for this column, encode on demand.
		rowLTs := make([]lintrans.LinearTransformation, numOutputCTs)
		for row := 0; row < numOutputCTs; row++ {
			ref := fmt.Sprintf("diag_%d_%d", row, col)
			blobIdx, ok := node.BlobRefs[ref]
			if !ok {
				return nil, fmt.Errorf("missing blob ref %q for node %q", ref, node.Name)
			}
			if blobIdx < 0 || blobIdx >= len(model.rawBlobs) {
				return nil, fmt.Errorf("blob ref %q index %d out of range (have %d blobs)", ref, blobIdx, len(model.rawBlobs))
			}

			diagMap, err := ParseDiagonalBlob(model.rawBlobs[blobIdx], maxSlots)
			if err != nil {
				return nil, fmt.Errorf("parsing diagonal blob %q: %w", ref, err)
			}

			diagonals := lintrans.Diagonals[float64](diagMap)
			ltparams := lintrans.Parameters{
				DiagonalsIndexList:        diagonals.DiagonalsIndexList(),
				LevelQ:                    node.Level,
				LevelP:                    model.params.MaxLevelP(),
				Scale:                     rlwe.NewScale(model.params.Q()[node.Level]),
				LogDimensions:             ring.Dimensions{Rows: 0, Cols: model.params.LogMaxSlots()},
				LogBabyStepGiantStepRatio: int(math.Log2(cfg.BSGSRatio)),
			}

			lt := lintrans.NewTransformation(model.params, ltparams)
			if err := lintrans.Encode(e.encoder, diagonals, lt); err != nil {
				return nil, fmt.Errorf("encoding linear transform %q: %w", ref, err)
			}
			rowLTs[row] = lt
		}

		// EvaluateManyNew shares BSGS rotations across all row blocks for this input CT.
		partials, err := e.linEval.EvaluateManyNew(inputs[col], rowLTs)
		if err != nil {
			return nil, fmt.Errorf("evaluating LT column %d for node %q: %w", col, node.Name, err)
		}

		// Accumulate partials into outputs.
		for row := 0; row < numOutputCTs; row++ {
			if outputs[row] == nil {
				outputs[row] = partials[row]
			} else {
				outputs[row], err = e.eval.AddNew(outputs[row], partials[row])
				if err != nil {
					return nil, fmt.Errorf("accumulating LT block (%d,%d): %w", row, col, err)
				}
			}
		}
	}

	// Rescale + bias + output rotations per output CT.
	biases := model.biases[node.Name]
	for row := 0; row < numOutputCTs; row++ {
		if err := e.eval.Rescale(outputs[row], outputs[row]); err != nil {
			return nil, fmt.Errorf("rescale after LT (row %d): %w", row, err)
		}

		if biases != nil && row < len(biases) && biases[row] != nil {
			var err error
			outputs[row], err = e.eval.AddNew(outputs[row], biases[row])
			if err != nil {
				return nil, fmt.Errorf("adding bias (row %d): %w", row, err)
			}
		}

		if cfg.OutputRotations > 0 {
			for i := 0; i < cfg.OutputRotations; i++ {
				rotation := maxSlots / (1 << (i + 1))
				rotated, err := e.eval.RotateNew(outputs[row], rotation)
				if err != nil {
					return nil, fmt.Errorf("output rotation step %d (rot=%d, row %d): %w", i, rotation, row, err)
				}
				outputs[row], err = e.eval.AddNew(outputs[row], rotated)
				if err != nil {
					return nil, fmt.Errorf("accumulating output rotation step %d (row %d): %w", i, row, err)
				}
			}
		}
	}

	return outputs, nil
}

// evalPolynomialMulti evaluates a polynomial on each CT in the list.
func (e *Evaluator) evalPolynomialMulti(model *Model, node *Node, cts []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	out := make([]*rlwe.Ciphertext, len(cts))
	for i, ct := range cts {
		var err error
		out[i], err = e.evalPolynomial(model, node, ct)
		if err != nil {
			return nil, fmt.Errorf("ct[%d]: %w", i, err)
		}
	}
	return out, nil
}

// evalPolynomial evaluates a polynomial node: optional prescale/constant, polynomial evaluation,
// optional postscale. Follows the Python Chebyshev class: prescale first, then constant, then poly eval.
//
// When the model is compiled with fuse_modules=true, the fuser absorbs prescale/constant into the
// preceding linear layer's weights/bias. In that case, prescale/constant must NOT be applied here
// (the Python Chebyshev.forward checks `if not self.fused:` before applying them).
func (e *Evaluator) evalPolynomial(model *Model, node *Node, ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	cfg, ok := model.polyConfigs[node.Name]
	if !ok {
		return nil, fmt.Errorf("no config found for polynomial node %q", node.Name)
	}

	poly, ok := model.polys[node.Name]
	if !ok {
		return nil, fmt.Errorf("no polynomial found for node %q", node.Name)
	}

	// Copy input to avoid modifying it (Lattigo's polynomial evaluator may modify in-place).
	work := ckks.NewCiphertext(e.params, 1, ct.Level())
	work.Copy(ct)

	// Apply prescale/constant only if modules were NOT fused during compilation.
	// When fused, prescale and constant are already absorbed into the preceding LT's weights/bias.
	if !model.header.Config.FuseModules {
		var err error
		// Apply prescale: scalar multiply then rescale.
		// MulNew(ct, float64) encodes the scalar at ct's scale, so the result
		// has scale = ct.Scale^2. Rescale brings it back to ~DefaultScale.
		// This consumes 1 level (accounted for by Python's set_depth += 1).
		if cfg.Prescale != 1 {
			work, err = e.eval.MulNew(work, cfg.Prescale)
			if err != nil {
				return nil, fmt.Errorf("prescale mul: %w", err)
			}
			if err = e.eval.Rescale(work, work); err != nil {
				return nil, fmt.Errorf("prescale rescale: %w", err)
			}
		}

		// Apply constant offset (scalar add, no level consumed).
		if cfg.Constant != 0 {
			work, err = e.eval.AddNew(work, cfg.Constant)
			if err != nil {
				return nil, fmt.Errorf("constant add: %w", err)
			}
		}
	}

	// Evaluate polynomial. Target scale = DefaultScale so output matches expected scale.
	targetScale := e.params.DefaultScale()
	result, err := e.polyEval.Evaluate(work, poly, targetScale)
	if err != nil {
		return nil, fmt.Errorf("polynomial evaluate: %w", err)
	}

	// Apply postscale if needed (scalar multiply + rescale, consumes 1 level).
	if cfg.Postscale != 1 {
		result, err = e.eval.MulNew(result, cfg.Postscale)
		if err != nil {
			return nil, fmt.Errorf("postscale mul: %w", err)
		}
		if err := e.eval.Rescale(result, result); err != nil {
			return nil, fmt.Errorf("postscale rescale: %w", err)
		}
	}

	return result, nil
}

// getBootstrapper returns (or lazily creates) a bootstrapping.Evaluator for the
// given slot count, using the model's manifest parameters.
func (e *Evaluator) getBootstrapper(model *Model, slots int) (*bootstrapping.Evaluator, error) {
	if slots <= 0 || slots&(slots-1) != 0 {
		return nil, fmt.Errorf("bootstrap slots must be a power of 2, got %d", slots)
	}
	logSlots := bits.Len(uint(slots)) - 1
	if btp, ok := e.bootstrappers[logSlots]; ok {
		return btp, nil
	}

	if e.btpKeys == nil {
		return nil, fmt.Errorf("bootstrap keys not provided but bootstrap op requires them")
	}

	manifest := model.header.Manifest
	btpLit := bootstrapping.ParametersLiteral{
		LogN:     utils.Pointy(manifest.BtpLogN),
		LogP:     copyIntSlice(manifest.BootLogP),
		Xs:       e.params.Xs(),
		LogSlots: utils.Pointy(logSlots),
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(e.params, btpLit)
	if err != nil {
		return nil, fmt.Errorf("creating bootstrap parameters (slots=%d): %w", slots, err)
	}

	btp, err := bootstrapping.NewEvaluator(btpParams, e.btpKeys)
	if err != nil {
		return nil, fmt.Errorf("creating bootstrapper (slots=%d): %w", slots, err)
	}

	e.bootstrappers[logSlots] = btp
	return btp, nil
}

// copyIntSlice returns a copy of the input slice.
func copyIntSlice(s []int) []int {
	if s == nil {
		return nil
	}
	out := make([]int, len(s))
	copy(out, s)
	return out
}

// evalBootstrapMulti bootstraps each CT in the list.
func (e *Evaluator) evalBootstrapMulti(model *Model, node *Node, cts []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	out := make([]*rlwe.Ciphertext, len(cts))
	for i, ct := range cts {
		var err error
		out[i], err = e.evalBootstrap(model, node, ct)
		if err != nil {
			return nil, fmt.Errorf("ct[%d]: %w", i, err)
		}
	}
	return out, nil
}

// evalBootstrap implements the bootstrap operation:
//  1. Parse BootstrapConfig
//  2. Constant shift (center values around 0)
//  3. Prescale (map to [-1, 1]) via plaintext multiply + rescale
//  4. Set sparse LogDimensions.Cols
//  5. Bootstrap (refresh modulus chain)
//  6. Sparse-slot postscale (integer multiply, restore dims)
//  7. Range-mapping postscale (integer multiply, no rescale)
//  8. Un-shift constant
func (e *Evaluator) evalBootstrap(model *Model, node *Node, ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	cfg, err := parseBootstrapConfig(node.ConfigRaw)
	if err != nil {
		return nil, fmt.Errorf("parsing bootstrap config: %w", err)
	}

	bootstrapper, err := e.getBootstrapper(model, cfg.Slots)
	if err != nil {
		return nil, err
	}

	// Work on a copy to avoid aliasing.
	work := ct.CopyNew()

	// Step 2: Constant shift — center values around 0.
	if cfg.Constant != 0 {
		work, err = e.eval.AddNew(work, cfg.Constant)
		if err != nil {
			return nil, fmt.Errorf("bootstrap constant shift: %w", err)
		}
	}

	// Step 3: Prescale — map values to [-1, 1] and zero inactive slots.
	if work.Level() < 1 {
		return nil, fmt.Errorf("bootstrap requires input level >= 1 for prescale, got level %d", work.Level())
	}
	{
		level := work.Level()
		ql := e.params.Q()[level]
		prescalePt := ckks.NewPlaintext(e.params, level)
		prescalePt.Scale = rlwe.NewScale(ql)

		prescaleVec := make([]float64, e.params.MaxSlots())
		for i := 0; i < cfg.Slots; i++ {
			prescaleVec[i] = cfg.Prescale
		}
		if err := e.encoder.Encode(prescaleVec, prescalePt); err != nil {
			return nil, fmt.Errorf("encoding prescale: %w", err)
		}

		work, err = e.eval.MulNew(work, prescalePt)
		if err != nil {
			return nil, fmt.Errorf("prescale mul: %w", err)
		}
		if err = e.eval.Rescale(work, work); err != nil {
			return nil, fmt.Errorf("prescale rescale: %w", err)
		}
	}

	// Step 4: Set sparse LogDimensions for bootstrap.
	work.LogDimensions.Cols = bootstrapper.LogMaxSlots()

	// Step 5: Bootstrap — refresh modulus chain.
	work, err = bootstrapper.Bootstrap(work)
	if err != nil {
		return nil, fmt.Errorf("bootstrap: %w", err)
	}

	// Step 6: Sparse-slot postscale — compensate for sparse slot packing.
	sparsePostscale := 1 << (e.params.LogMaxSlots() - bootstrapper.LogMaxSlots())
	if sparsePostscale > 1 {
		work, err = e.eval.MulNew(work, sparsePostscale)
		if err != nil {
			return nil, fmt.Errorf("sparse postscale mul: %w", err)
		}
	}

	// Restore full LogDimensions.
	work.LogDimensions.Cols = e.params.LogMaxSlots()

	// Step 7: Range-mapping postscale — integer multiply (no rescale needed).
	rangePostscale := int(cfg.Postscale)
	if cfg.Postscale != float64(rangePostscale) || rangePostscale < 1 {
		return nil, fmt.Errorf("bootstrap postscale must be a positive integer, got %f", cfg.Postscale)
	}
	if rangePostscale > 1 {
		work, err = e.eval.MulNew(work, rangePostscale)
		if err != nil {
			return nil, fmt.Errorf("range postscale mul: %w", err)
		}
	}

	// Step 8: Un-shift constant.
	if cfg.Constant != 0 {
		work, err = e.eval.AddNew(work, -cfg.Constant)
		if err != nil {
			return nil, fmt.Errorf("bootstrap constant un-shift: %w", err)
		}
	}

	return work, nil
}
