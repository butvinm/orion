package orionclient

import (
	"fmt"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/lintrans"
	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/polynomial"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
	"github.com/baahl-nyu/lattigo/v6/utils/bignum"
)

// Evaluator performs FHE operations on ciphertexts. It holds evaluation keys
// and exposes arithmetic, rotation, rescaling, polynomial evaluation,
// linear transforms, and bootstrapping. Multiple Evaluator instances can coexist.
type Evaluator struct {
	params     Params
	ckksParams ckks.Parameters
	encoder    *ckks.Encoder
	evaluator  *ckks.Evaluator
	evalKeys   *rlwe.MemEvaluationKeySet

	polyEval *polynomial.Evaluator
	linEval  *lintrans.Evaluator

	// Per-slot-count bootstrappers, lazily created from loaded keys.
	bootstrappers map[int]*bootstrapping.Evaluator
}

// NewEvaluator creates an Evaluator from parameters and a key bundle.
// The key bundle must contain at minimum the RLK. Galois keys and
// bootstrap keys are loaded from the bundle as provided.
func NewEvaluator(p Params, keys EvalKeyBundle) (*Evaluator, error) {
	ckksParams, err := p.NewCKKSParameters()
	if err != nil {
		return nil, fmt.Errorf("creating CKKS parameters: %w", err)
	}

	// Deserialize relinearization key
	var rlk *rlwe.RelinearizationKey
	if keys.RLK != nil {
		rlk = &rlwe.RelinearizationKey{}
		if err := rlk.UnmarshalBinary(keys.RLK); err != nil {
			return nil, fmt.Errorf("unmarshalling RLK: %w", err)
		}
	}

	// Build evaluation key set with RLK and all Galois keys
	galoisKeys := make([]*rlwe.GaloisKey, 0, len(keys.GaloisKeys))
	for _, gkData := range keys.GaloisKeys {
		gk := &rlwe.GaloisKey{}
		if err := gk.UnmarshalBinary(gkData); err != nil {
			return nil, fmt.Errorf("unmarshalling Galois key: %w", err)
		}
		galoisKeys = append(galoisKeys, gk)
	}

	evalKeys := rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...)

	eval := ckks.NewEvaluator(ckksParams, evalKeys)
	enc := ckks.NewEncoder(ckksParams)

	polyEval := polynomial.NewEvaluator(ckksParams, eval)
	linEval := lintrans.NewEvaluator(eval)

	e := &Evaluator{
		params:        p,
		ckksParams:    ckksParams,
		encoder:       enc,
		evaluator:     eval,
		evalKeys:      evalKeys,
		polyEval:      polyEval,
		linEval:       linEval,
		bootstrappers: make(map[int]*bootstrapping.Evaluator),
	}

	// Load bootstrap keys
	for slots, bkData := range keys.BootstrapKeys {
		if err := e.loadBootstrapKey(slots, bkData, keys.BootLogP); err != nil {
			return nil, fmt.Errorf("loading bootstrap keys for %d slots: %w", slots, err)
		}
	}

	return e, nil
}

// Close releases evaluator resources.
func (e *Evaluator) Close() {
	e.evaluator = nil
	e.encoder = nil
	e.evalKeys = nil
	e.polyEval = nil
	e.linEval = nil
	e.bootstrappers = nil
}

// Encode encodes float64 values into a Plaintext at the given level and scale.
func (e *Evaluator) Encode(values []float64, level int, scale uint64) (*Plaintext, error) {
	if e.encoder == nil {
		return nil, fmt.Errorf("evaluator is closed")
	}
	pt := ckks.NewPlaintext(e.ckksParams, level)
	pt.Scale = rlwe.NewScale(scale)
	if err := e.encoder.Encode(values, pt); err != nil {
		return nil, fmt.Errorf("encoding: %w", err)
	}
	return &Plaintext{raw: pt, shape: []int{len(values)}}, nil
}

// --- Ciphertext-Ciphertext arithmetic ---

// Add adds two ciphertexts element-wise, returning a new ciphertext.
func (e *Evaluator) Add(ct0, ct1 *Ciphertext) (*Ciphertext, error) {
	return e.binaryOp(ct0, ct1, func(a, b *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
		return e.evaluator.AddNew(a, b)
	})
}

// Sub subtracts ct1 from ct0 element-wise, returning a new ciphertext.
func (e *Evaluator) Sub(ct0, ct1 *Ciphertext) (*Ciphertext, error) {
	return e.binaryOp(ct0, ct1, func(a, b *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
		return e.evaluator.SubNew(a, b)
	})
}

// Mul multiplies two ciphertexts with relinearization, returning a new ciphertext.
func (e *Evaluator) Mul(ct0, ct1 *Ciphertext) (*Ciphertext, error) {
	return e.binaryOp(ct0, ct1, func(a, b *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
		return e.evaluator.MulRelinNew(a, b)
	})
}

// --- Ciphertext-Plaintext arithmetic ---

// AddPlaintext adds a plaintext to a ciphertext, returning a new ciphertext.
func (e *Evaluator) AddPlaintext(ct *Ciphertext, pt *Plaintext) (*Ciphertext, error) {
	return e.unaryOpPt(ct, pt, func(c *rlwe.Ciphertext, p *rlwe.Plaintext) (*rlwe.Ciphertext, error) {
		return e.evaluator.AddNew(c, p)
	})
}

// SubPlaintext subtracts a plaintext from a ciphertext, returning a new ciphertext.
func (e *Evaluator) SubPlaintext(ct *Ciphertext, pt *Plaintext) (*Ciphertext, error) {
	return e.unaryOpPt(ct, pt, func(c *rlwe.Ciphertext, p *rlwe.Plaintext) (*rlwe.Ciphertext, error) {
		return e.evaluator.SubNew(c, p)
	})
}

// MulPlaintext multiplies a ciphertext by a plaintext, returning a new ciphertext.
func (e *Evaluator) MulPlaintext(ct *Ciphertext, pt *Plaintext) (*Ciphertext, error) {
	return e.unaryOpPt(ct, pt, func(c *rlwe.Ciphertext, p *rlwe.Plaintext) (*rlwe.Ciphertext, error) {
		return e.evaluator.MulNew(c, p)
	})
}

// --- Scalar arithmetic ---

// AddScalar adds a float64 scalar to a ciphertext, returning a new ciphertext.
func (e *Evaluator) AddScalar(ct *Ciphertext, scalar float64) (*Ciphertext, error) {
	return e.unaryOpScalar(ct, func(c *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
		return e.evaluator.AddNew(c, scalar)
	})
}

// MulScalar multiplies a ciphertext by a float64 scalar, returning a new ciphertext.
func (e *Evaluator) MulScalar(ct *Ciphertext, scalar float64) (*Ciphertext, error) {
	return e.unaryOpScalar(ct, func(c *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
		return e.evaluator.MulNew(c, scalar)
	})
}

// Negate negates a ciphertext, returning a new ciphertext.
func (e *Evaluator) Negate(ct *Ciphertext) (*Ciphertext, error) {
	return e.MulScalar(ct, -1.0)
}

// --- Rotation and rescaling ---

// Rotate rotates a ciphertext by the given amount, returning a new ciphertext.
func (e *Evaluator) Rotate(ct *Ciphertext, amount int) (*Ciphertext, error) {
	return e.unaryOpScalar(ct, func(c *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
		return e.evaluator.RotateNew(c, amount)
	})
}

// Rescale rescales a ciphertext (reduces level by 1), returning a new ciphertext.
func (e *Evaluator) Rescale(ct *Ciphertext) (*Ciphertext, error) {
	return e.unaryOpScalar(ct, func(c *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
		out := c.CopyNew()
		if err := e.evaluator.Rescale(out, out); err != nil {
			return nil, err
		}
		return out, nil
	})
}

// --- Polynomial evaluation ---

// EvalPoly evaluates a polynomial on a ciphertext, returning a new ciphertext.
func (e *Evaluator) EvalPoly(ct *Ciphertext, poly bignum.Polynomial, outScale uint64) (*Ciphertext, error) {
	if e.polyEval == nil {
		return nil, fmt.Errorf("evaluator is closed")
	}
	if ct.NumCiphertexts() != 1 {
		return nil, fmt.Errorf("EvalPoly requires single underlying ciphertext, got %d", ct.NumCiphertexts())
	}

	// Copy input to avoid modifying it
	ctTmp := ckks.NewCiphertext(e.ckksParams, 1, ct.cts[0].Level())
	ctTmp.Copy(ct.cts[0])

	res, err := e.polyEval.Evaluate(ctTmp, poly, rlwe.NewScale(outScale))
	if err != nil {
		return nil, fmt.Errorf("polynomial evaluation: %w", err)
	}

	return NewCiphertext([]*rlwe.Ciphertext{res}, ct.Shape()), nil
}

// --- Linear transform ---

// EvalLinearTransform evaluates a linear transform on a ciphertext.
func (e *Evaluator) EvalLinearTransform(ct *Ciphertext, lt *LinearTransform) (*Ciphertext, error) {
	if e.linEval == nil {
		return nil, fmt.Errorf("evaluator is closed")
	}
	if ct.NumCiphertexts() != 1 {
		return nil, fmt.Errorf("EvalLinearTransform requires single underlying ciphertext, got %d", ct.NumCiphertexts())
	}

	// Refresh the lintrans evaluator with current keys
	e.linEval = lintrans.NewEvaluator(e.evaluator.WithKey(e.evalKeys))

	res, err := e.linEval.EvaluateNew(ct.cts[0], lt.raw)
	if err != nil {
		return nil, fmt.Errorf("linear transform evaluation: %w", err)
	}

	return NewCiphertext([]*rlwe.Ciphertext{res}, ct.Shape()), nil
}

// MaxSlots returns the maximum number of plaintext slots.
func (e *Evaluator) MaxSlots() int {
	return e.ckksParams.MaxSlots()
}

// GaloisElement returns the Galois element for a given rotation step.
func (e *Evaluator) GaloisElement(rotation int) uint64 {
	return e.ckksParams.GaloisElement(rotation)
}

// --- Internal helpers ---

// binaryOp applies a binary operation to pairs of underlying ciphertexts.
func (e *Evaluator) binaryOp(ct0, ct1 *Ciphertext, op func(*rlwe.Ciphertext, *rlwe.Ciphertext) (*rlwe.Ciphertext, error)) (*Ciphertext, error) {
	if e.evaluator == nil {
		return nil, fmt.Errorf("evaluator is closed")
	}
	if ct0.NumCiphertexts() != ct1.NumCiphertexts() {
		return nil, fmt.Errorf("ciphertext count mismatch: %d vs %d", ct0.NumCiphertexts(), ct1.NumCiphertexts())
	}

	results := make([]*rlwe.Ciphertext, ct0.NumCiphertexts())
	for i := range results {
		out, err := op(ct0.cts[i], ct1.cts[i])
		if err != nil {
			return nil, err
		}
		results[i] = out
	}
	return NewCiphertext(results, ct0.Shape()), nil
}

// unaryOpPt applies a ciphertext-plaintext operation to each underlying ct.
func (e *Evaluator) unaryOpPt(ct *Ciphertext, pt *Plaintext, op func(*rlwe.Ciphertext, *rlwe.Plaintext) (*rlwe.Ciphertext, error)) (*Ciphertext, error) {
	if e.evaluator == nil {
		return nil, fmt.Errorf("evaluator is closed")
	}

	results := make([]*rlwe.Ciphertext, ct.NumCiphertexts())
	for i := range results {
		out, err := op(ct.cts[i], pt.raw)
		if err != nil {
			return nil, err
		}
		results[i] = out
	}
	return NewCiphertext(results, ct.Shape()), nil
}

// unaryOpScalar applies a unary operation to each underlying ciphertext.
func (e *Evaluator) unaryOpScalar(ct *Ciphertext, op func(*rlwe.Ciphertext) (*rlwe.Ciphertext, error)) (*Ciphertext, error) {
	if e.evaluator == nil {
		return nil, fmt.Errorf("evaluator is closed")
	}

	results := make([]*rlwe.Ciphertext, ct.NumCiphertexts())
	for i := range results {
		out, err := op(ct.cts[i])
		if err != nil {
			return nil, err
		}
		results[i] = out
	}
	return NewCiphertext(results, ct.Shape()), nil
}
