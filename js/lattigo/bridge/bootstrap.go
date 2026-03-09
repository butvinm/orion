//go:build js && wasm

package main

import (
	"fmt"
	"syscall/js"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
	"github.com/baahl-nyu/lattigo/v6/utils"
)

// newBootstrapParams(paramsHID, logN?, logP?, h?, logSlots?)
// → {handle: number} | {error: string}
// All args after paramsHID are optional (pass null/undefined to use defaults).
func newBootstrapParams(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("newBootstrapParams: need paramsHID argument")
	}

	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("newBootstrapParams: invalid params handle")
	}
	params := obj.(*ckks.Parameters)

	btpLit := bootstrapping.ParametersLiteral{}

	if len(args) > 1 && !args[1].IsUndefined() && !args[1].IsNull() && args[1].Int() > 0 {
		btpLit.LogN = utils.Pointy(args[1].Int())
	}
	if len(args) > 2 && !args[2].IsUndefined() && !args[2].IsNull() {
		btpLit.LogP = jsToIntSlice(args[2])
	}
	if len(args) > 3 && !args[3].IsUndefined() && !args[3].IsNull() && args[3].Int() > 0 {
		btpLit.Xs = ring.Ternary{H: args[3].Int()}
	}
	if len(args) > 4 && !args[4].IsUndefined() && !args[4].IsNull() && args[4].Int() > 0 {
		btpLit.LogSlots = utils.Pointy(args[4].Int())
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(*params, btpLit)
	if err != nil {
		return errorResult(fmt.Sprintf("newBootstrapParams: %v", err))
	}

	return handleResult(Store(&btpParams))
}

// bootstrapParamsGenEvalKeys(btpParamsHID, skHID) → Promise<{evkHID, btpEvkHID}>
// Async — key generation is heavy (5–30s). Returns a Promise.
func bootstrapParamsGenEvalKeys(_ js.Value, args []js.Value) any {
	if len(args) < 2 {
		return errorResult("bootstrapParamsGenEvalKeys: need btpParamsHID and skHID arguments")
	}

	btpParamsHID := uint32(args[0].Int())
	skHID := uint32(args[1].Int())

	handler := js.FuncOf(func(_ js.Value, pArgs []js.Value) any {
		resolve, reject := pArgs[0], pArgs[1]
		go func() {
			defer func() {
				if r := recover(); r != nil {
					reject.Invoke(fmt.Sprintf("bootstrapParamsGenEvalKeys panic: %v", r))
				}
			}()

			obj, ok := Load(btpParamsHID)
			if !ok {
				reject.Invoke("bootstrapParamsGenEvalKeys: invalid btp params handle")
				return
			}
			btpParams := obj.(*bootstrapping.Parameters)

			skObj, ok := Load(skHID)
			if !ok {
				reject.Invoke("bootstrapParamsGenEvalKeys: invalid secret key handle")
				return
			}
			sk := skObj.(*rlwe.SecretKey)

			btpKeys, _, err := btpParams.GenEvaluationKeys(sk)
			if err != nil {
				reject.Invoke(fmt.Sprintf("bootstrapParamsGenEvalKeys: %v", err))
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
	promise := js.Global().Get("Promise").New(handler)
	handler.Release()
	return promise
}

// bootstrapEvalKeysMarshal(btpEvkHID) → Uint8Array | {error: string}
func bootstrapEvalKeysMarshal(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorResult("bootstrapEvalKeysMarshal: missing btpEvkHID argument")
	}
	obj, ok := Load(uint32(args[0].Int()))
	if !ok {
		return errorResult("bootstrapEvalKeysMarshal: invalid bootstrap evaluation keys handle")
	}
	btpKeys := obj.(*bootstrapping.EvaluationKeys)
	data, err := btpKeys.MarshalBinary()
	if err != nil {
		return errorResult(fmt.Sprintf("bootstrapEvalKeysMarshal: %v", err))
	}
	return bytesToJS(data)
}
