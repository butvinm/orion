package evaluator

import (
	"fmt"
	"sort"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/lintrans"
	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/polynomial"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

// Evaluator runs FHE inference on a compiled Model by walking the computation graph.
// It is NOT goroutine-safe (Lattigo buffers are reused internally).
type Evaluator struct {
	params   ckks.Parameters
	encoder  *ckks.Encoder
	eval     *ckks.Evaluator
	linEval  *lintrans.Evaluator
	polyEval *polynomial.Evaluator
}

// NewEvaluatorFromKeySet creates an Evaluator from Lattigo types directly.
func NewEvaluatorFromKeySet(ckksParams ckks.Parameters, keys *rlwe.MemEvaluationKeySet) (*Evaluator, error) {
	eval := ckks.NewEvaluator(ckksParams, keys)
	enc := ckks.NewEncoder(ckksParams)
	polyEval := polynomial.NewEvaluator(ckksParams, eval)
	linEval := lintrans.NewEvaluator(eval)

	return &Evaluator{
		params:   ckksParams,
		encoder:  enc,
		eval:     eval,
		linEval:  linEval,
		polyEval: polyEval,
	}, nil
}

// Close releases evaluator resources.
func (e *Evaluator) Close() {
	e.eval = nil
	e.encoder = nil
	e.linEval = nil
	e.polyEval = nil
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

	// Assign input ciphertext to the graph's input node.
	results[model.graph.Input] = input

	// Walk the graph in topological order.
	for _, name := range model.graph.Order {
		if name == model.graph.Input {
			continue
		}

		node := model.graph.Nodes[name]
		inputs := model.graph.Inputs[name]

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
			return nil, fmt.Errorf("op %q not yet implemented for node %q", node.Op, name)

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
