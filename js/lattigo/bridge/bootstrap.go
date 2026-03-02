//go:build js && wasm

package main

import (
	"encoding/json"
	"fmt"
	"syscall/js"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
	"github.com/baahl-nyu/lattigo/v6/utils"
)

// btpLitJSON is the JSON schema for bootstrapping.ParametersLiteral.
// All fields are optional — omitted fields use Lattigo's defaults.
type btpLitJSON struct {
	LogN     *int  `json:"LogN"`
	LogP     []int `json:"LogP"`
	H        *int  `json:"H"`        // Maps to Xs: ring.Ternary{H: value}
	LogSlots *int  `json:"LogSlots"`
}

// newBootstrapParametersFromLiteral(paramsHID: number, btpLitJSON: string) → {handle: number} | {error: string}
// Constructs bootstrapping.Parameters from the residual CKKS params and a JSON literal.
// Sync — just parameter construction and validation.
func newBootstrapParametersFromLiteral(_ js.Value, args []js.Value) any {
	if len(args) < 2 {
		return errorResult("newBootstrapParametersFromLiteral: need paramsHID and btpLitJSON arguments")
	}

	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("newBootstrapParametersFromLiteral: invalid params handle")
	}
	params := obj.(*ckks.Parameters)

	jsonStr := args[1].String()
	var lit btpLitJSON
	if err := json.Unmarshal([]byte(jsonStr), &lit); err != nil {
		return errorResult(fmt.Sprintf("newBootstrapParametersFromLiteral: parsing JSON: %v", err))
	}

	btpLit := bootstrapping.ParametersLiteral{}
	if lit.LogN != nil {
		btpLit.LogN = utils.Pointy(*lit.LogN)
	}
	if lit.LogP != nil {
		btpLit.LogP = lit.LogP
	}
	if lit.H != nil {
		btpLit.Xs = ring.Ternary{H: *lit.H}
	}
	if lit.LogSlots != nil {
		btpLit.LogSlots = utils.Pointy(*lit.LogSlots)
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(*params, btpLit)
	if err != nil {
		return errorResult(fmt.Sprintf("newBootstrapParametersFromLiteral: %v", err))
	}

	return handleResult(Store(&btpParams))
}

// btpParamsGenEvaluationKeys(btpParamsHID: number, skHID: number) → Promise<{evkHID: number, btpEvkHID: number}>
// Async — key generation is heavy (5–30s). Returns a Promise.
// evkHID: the *rlwe.MemEvaluationKeySet embedded in the bootstrap EvaluationKeys.
// btpEvkHID: the full *bootstrapping.EvaluationKeys (includes ring switching keys).
func btpParamsGenEvaluationKeys(_ js.Value, args []js.Value) any {
	if len(args) < 2 {
		return errorResult("btpParamsGenEvaluationKeys: need btpParamsHID and skHID arguments")
	}

	btpParamsHID := uint32(args[0].Int())
	skHID := uint32(args[1].Int())

	handler := js.FuncOf(func(_ js.Value, pArgs []js.Value) any {
		resolve, reject := pArgs[0], pArgs[1]
		go func() {
			defer func() {
				if r := recover(); r != nil {
					reject.Invoke(fmt.Sprintf("btpParamsGenEvaluationKeys panic: %v", r))
				}
			}()

			obj, ok := Load(btpParamsHID)
			if !ok {
				reject.Invoke("btpParamsGenEvaluationKeys: invalid btp params handle")
				return
			}
			btpParams := obj.(*bootstrapping.Parameters)

			skObj, ok := Load(skHID)
			if !ok {
				reject.Invoke("btpParamsGenEvaluationKeys: invalid secret key handle")
				return
			}
			sk := skObj.(*rlwe.SecretKey)

			btpKeys, _, err := btpParams.GenEvaluationKeys(sk)
			if err != nil {
				reject.Invoke(fmt.Sprintf("btpParamsGenEvaluationKeys: %v", err))
				return
			}

			evkHID := Store(btpKeys.MemEvaluationKeySet)
			btpEvkHID := Store(btpKeys)

			result := js.Global().Get("Object").New()
			result.Set("evkHID", int(evkHID))
			result.Set("btpEvkHID", int(btpEvkHID))
			resolve.Invoke(result)
		}()
		return nil
	})
	return js.Global().Get("Promise").New(handler)
}
