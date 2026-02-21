package orionclient

import (
	"fmt"
	"math"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/utils"
)

// Bootstrap refreshes a ciphertext's multiplicative level via bootstrapping.
// numSlots specifies which pre-loaded bootstrapper to use.
func (e *Evaluator) Bootstrap(ct *Ciphertext, numSlots int) (*Ciphertext, error) {
	if e.evaluator == nil {
		return nil, fmt.Errorf("evaluator is closed")
	}
	if ct.NumCiphertexts() != 1 {
		return nil, fmt.Errorf("Bootstrap requires single underlying ciphertext, got %d", ct.NumCiphertexts())
	}

	btp, ok := e.bootstrappers[numSlots]
	if !ok {
		return nil, fmt.Errorf("no bootstrapper loaded for %d slots", numSlots)
	}

	ctBtp := ct.cts[0].CopyNew()
	ctBtp.LogDimensions.Cols = btp.LogMaxSlots()

	ctOut, err := btp.Bootstrap(ctBtp)
	if err != nil {
		return nil, fmt.Errorf("bootstrap: %w", err)
	}

	// Post-scale for slot count difference
	postscale := int(1 << (e.ckksParams.LogMaxSlots() - btp.LogMaxSlots()))
	if err := e.evaluator.Mul(ctOut, postscale, ctOut); err != nil {
		return nil, fmt.Errorf("bootstrap post-scale: %w", err)
	}

	ctOut.LogDimensions.Cols = e.ckksParams.LogMaxSlots()

	return NewCiphertext([]*rlwe.Ciphertext{ctOut}, ct.Shape()), nil
}

// loadBootstrapKey loads serialized bootstrap keys for a given slot count.
func (e *Evaluator) loadBootstrapKey(slots int, data []byte, logP []int) error {
	if _, exists := e.bootstrappers[slots]; exists {
		return nil
	}

	btpLit := bootstrapping.ParametersLiteral{
		LogN:     utils.Pointy(e.ckksParams.LogN()),
		LogP:     logP,
		Xs:       e.ckksParams.Xs(),
		LogSlots: utils.Pointy(int(math.Log2(float64(slots)))),
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(e.ckksParams, btpLit)
	if err != nil {
		return fmt.Errorf("creating bootstrap parameters: %w", err)
	}

	var btpKeys bootstrapping.EvaluationKeys
	if err := btpKeys.UnmarshalBinary(data); err != nil {
		return fmt.Errorf("unmarshalling bootstrap keys: %w", err)
	}

	btpEval, err := bootstrapping.NewEvaluator(btpParams, &btpKeys)
	if err != nil {
		return fmt.Errorf("creating bootstrap evaluator: %w", err)
	}

	e.bootstrappers[slots] = btpEval
	return nil
}
