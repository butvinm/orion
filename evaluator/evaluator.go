package evaluator

import (
	"fmt"
	"math/bits"
	"sort"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/lintrans"
	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/polynomial"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
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
// The input ciphertext must contain already-padded slot values.
func (e *Evaluator) Forward(model *Model, input *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	if e.eval == nil {
		return nil, fmt.Errorf("evaluator is closed")
	}
	if model == nil {
		return nil, fmt.Errorf("model is nil")
	}
	if input == nil {
		return nil, fmt.Errorf("input ciphertext is nil")
	}

	results := make(map[string]*rlwe.Ciphertext)

	// Make the raw input available under a virtual key so the graph's input
	// node can consume it like any other node.
	const virtualInput = "__input__"
	results[virtualInput] = input

	// Walk the graph in topological order.
	for _, name := range model.graph.Order {
		node := model.graph.Nodes[name]
		inputs := model.graph.Inputs[name]

		// The graph's input node has no predecessors in the edge list;
		// wire it to the raw input ciphertext.
		if name == model.graph.Input && len(inputs) == 0 {
			inputs = []string{virtualInput}
		}

		var err error
		var result *rlwe.Ciphertext

		switch node.Op {
		case "flatten":
			if len(inputs) != 1 {
				return nil, fmt.Errorf("flatten %q: expected 1 input, got %d", name, len(inputs))
			}
			result, err = e.evalFlatten(results[inputs[0]])

		case "quad":
			if len(inputs) != 1 {
				return nil, fmt.Errorf("quad %q: expected 1 input, got %d", name, len(inputs))
			}
			result, err = e.evalQuad(results[inputs[0]])

		case "add":
			if len(inputs) != 2 {
				return nil, fmt.Errorf("add %q: expected 2 inputs, got %d", name, len(inputs))
			}
			result, err = e.evalAdd(results[inputs[0]], results[inputs[1]])

		case "mult":
			if len(inputs) != 2 {
				return nil, fmt.Errorf("mult %q: expected 2 inputs, got %d", name, len(inputs))
			}
			result, err = e.evalMult(results[inputs[0]], results[inputs[1]])

		case "linear_transform":
			if len(inputs) != 1 {
				return nil, fmt.Errorf("linear_transform %q: expected 1 input, got %d", name, len(inputs))
			}
			result, err = e.evalLinearTransform(model, node, results[inputs[0]])

		case "polynomial":
			if len(inputs) != 1 {
				return nil, fmt.Errorf("polynomial %q: expected 1 input, got %d", name, len(inputs))
			}
			result, err = e.evalPolynomial(model, node, results[inputs[0]])

		case "bootstrap":
			if len(inputs) != 1 {
				return nil, fmt.Errorf("bootstrap %q: expected 1 input, got %d", name, len(inputs))
			}
			result, err = e.evalBootstrap(model, node, results[inputs[0]])

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

// evalFlatten is a no-op — the input ciphertext already contains the flattened data.
// Returns a copy so the caller owns the result exclusively. This prevents
// pointer aliasing when the same ciphertext feeds multiple downstream nodes
// (e.g., residual connections with fan-out).
func (e *Evaluator) evalFlatten(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	return ct.CopyNew(), nil
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

// evalAdd adds two ciphertexts element-wise.
func (e *Evaluator) evalAdd(ct0, ct1 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	result, err := e.eval.AddNew(ct0, ct1)
	if err != nil {
		return nil, fmt.Errorf("Add: %w", err)
	}
	return result, nil
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

// evalLinearTransform evaluates a linear transform node: multi-block LT accumulation,
// rescale, bias addition, and optional output rotations.
func (e *Evaluator) evalLinearTransform(model *Model, node *Node, ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	cfg, ok := model.ltConfigs[node.Name]
	if !ok {
		return nil, fmt.Errorf("no config found for linear_transform node %q", node.Name)
	}

	// Look up pre-encoded LTs for this node.
	nodeTransforms, ok := model.transforms[node.Name]
	if !ok || len(nodeTransforms) == 0 {
		return nil, fmt.Errorf("no linear transforms found for node %q", node.Name)
	}

	// Evaluate each LT block and accumulate.
	// Sort refs for deterministic accumulation order (Go map iteration is random).
	refs := make([]string, 0, len(nodeTransforms))
	for ref := range nodeTransforms {
		refs = append(refs, ref)
	}
	sort.Strings(refs)

	var result *rlwe.Ciphertext
	for _, ref := range refs {
		lt := nodeTransforms[ref]
		partial, err := e.linEval.EvaluateNew(ct, lt)
		if err != nil {
			return nil, fmt.Errorf("evaluating LT block %q: %w", ref, err)
		}
		if result == nil {
			result = partial
		} else {
			result, err = e.eval.AddNew(result, partial)
			if err != nil {
				return nil, fmt.Errorf("accumulating LT block %q: %w", ref, err)
			}
		}
	}

	// Rescale after LT evaluation.
	if err := e.eval.Rescale(result, result); err != nil {
		return nil, fmt.Errorf("rescale after LT: %w", err)
	}

	// Add bias if present.
	if bias, ok := model.biases[node.Name]; ok {
		var err error
		result, err = e.eval.AddNew(result, bias)
		if err != nil {
			return nil, fmt.Errorf("adding bias: %w", err)
		}
	}

	// Apply output rotations (hybrid embedding fold-down).
	if cfg.OutputRotations > 0 {
		maxSlots := model.params.MaxSlots()
		for i := 0; i < cfg.OutputRotations; i++ {
			rotation := maxSlots / (1 << (i + 1))
			rotated, err := e.eval.RotateNew(result, rotation)
			if err != nil {
				return nil, fmt.Errorf("output rotation step %d (rot=%d): %w", i, rotation, err)
			}
			result, err = e.eval.AddNew(result, rotated)
			if err != nil {
				return nil, fmt.Errorf("accumulating output rotation step %d: %w", i, err)
			}
		}
	}

	return result, nil
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
	// The prescale plaintext has cfg.Prescale in active slots and 0.0 in inactive
	// slots. Even when prescale==1, the multiply is necessary to zero inactive
	// slots for clean sparse bootstrapping (matching the reference implementation
	// which unconditionally multiplies by prescale_ptxt).
	// This consumes 1 level. The compiler guarantees input_level >= 1
	// (level_dag.py:238-240 rejects bootstrap placement at level 0).
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
	// Postscale must be a positive integer (Python compiler uses math.ceil(absmax)).
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
