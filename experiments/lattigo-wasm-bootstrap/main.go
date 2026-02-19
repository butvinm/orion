//go:build js && wasm

package main

import (
	"fmt"
	"math"
	"syscall/js"
	"time"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
	"github.com/baahl-nyu/lattigo/v6/utils"
)

func main() {
	js.Global().Set("lattigoBootstrapTest", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		handler := js.FuncOf(func(_ js.Value, pArgs []js.Value) interface{} {
			resolve, reject := pArgs[0], pArgs[1]
			go func() {
				defer func() {
					if r := recover(); r != nil {
						reject.Invoke(fmt.Sprintf("panic: %v", r))
					}
				}()

				log := func(msg string) {
					js.Global().Get("console").Call("log", msg)
				}

				// Step 1: Create params
				log("Creating CKKS params...")
				t0 := time.Now()
				params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
					LogN:            13,
					LogQ:            []int{29, 26, 26, 26, 26, 26},
					LogP:            []int{29, 29},
					LogDefaultScale: 26,
					Xs:              ring.Ternary{H: 8192},
					RingType:        ring.ConjugateInvariant,
				})
				if err != nil {
					reject.Invoke(fmt.Sprintf("params error: %v", err))
					return
				}
				paramsTime := time.Since(t0)
				log(fmt.Sprintf("  params: %v", paramsTime))

				// Step 2: Generate secret key
				log("Generating secret key...")
				t1 := time.Now()
				kgen := rlwe.NewKeyGenerator(params)
				sk := kgen.GenSecretKeyNew()
				keygenTime := time.Since(t1)
				log(fmt.Sprintf("  keygen: %v", keygenTime))

				// Step 3: Generate RLK
				log("Generating RLK...")
				t2 := time.Now()
				rlk := kgen.GenRelinearizationKeyNew(sk)
				rlkTime := time.Since(t2)
				rlkData, _ := rlk.MarshalBinary()
				log(fmt.Sprintf("  rlk: %v (%d bytes = %.1f MB)", rlkTime, len(rlkData), float64(len(rlkData))/1024/1024))

				// Step 4: Generate a few Galois keys
				log("Generating Galois keys...")
				rotations := []int{1, 2, 4, 8, 16, 32, 64, 128}
				var totalGaloisTime time.Duration
				var totalGaloisBytes int
				for _, rot := range rotations {
					t := time.Now()
					galEl := params.GaloisElement(rot)
					gk := kgen.GenGaloisKeyNew(galEl, sk)
					elapsed := time.Since(t)
					totalGaloisTime += elapsed
					gkData, _ := gk.MarshalBinary()
					totalGaloisBytes += len(gkData)
				}
				log(fmt.Sprintf("  %d galois keys: %v (%d bytes = %.1f MB)", len(rotations), totalGaloisTime, totalGaloisBytes, float64(totalGaloisBytes)/1024/1024))

				// Step 5: Bootstrap key generation (THE BIG ONE)
				// For ConjugateInvariant ring, bootstrap needs LogN+1
				btpLogN := params.LogN()
				if params.RingType() == ring.ConjugateInvariant {
					btpLogN = params.LogN() + 1
				}
				numSlots := 256
				log(fmt.Sprintf("Generating bootstrap keys (slots=%d, btpLogN=%d)...", numSlots, btpLogN))
				t4 := time.Now()
				btpLit := bootstrapping.ParametersLiteral{
					LogN:     utils.Pointy(btpLogN),
					LogP:     []int{29, 29},
					Xs:       params.Xs(),
					LogSlots: utils.Pointy(int(math.Log2(float64(numSlots)))),
				}

				btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpLit)
				if err != nil {
					reject.Invoke(fmt.Sprintf("btp params error: %v", err))
					return
				}
				btpParamsTime := time.Since(t4)
				log(fmt.Sprintf("  btp params: %v", btpParamsTime))

				t5 := time.Now()
				btpKeys, _, err := btpParams.GenEvaluationKeys(sk)
				if err != nil {
					reject.Invoke(fmt.Sprintf("btp keygen error: %v", err))
					return
				}
				btpKeygenTime := time.Since(t5)
				log(fmt.Sprintf("  btp keygen: %v", btpKeygenTime))

				t6 := time.Now()
				btpData, err := btpKeys.MarshalBinary()
				if err != nil {
					reject.Invoke(fmt.Sprintf("btp marshal error: %v", err))
					return
				}
				btpSerializeTime := time.Since(t6)
				log(fmt.Sprintf("  btp serialize: %v (%d bytes = %.1f MB)", btpSerializeTime, len(btpData), float64(len(btpData))/1024/1024))

				totalTime := time.Since(t0)
				result := fmt.Sprintf(
					"DONE in %v | params:%v keygen:%v rlk:%v(%dB) galois(%d):%v(%dB) btp_params:%v btp_keygen:%v btp_serial:%v(%dB)",
					totalTime,
					paramsTime, keygenTime, rlkTime, len(rlkData),
					len(rotations), totalGaloisTime, totalGaloisBytes,
					btpParamsTime, btpKeygenTime, btpSerializeTime, len(btpData),
				)
				resolve.Invoke(result)
			}()
			return nil
		})
		return js.Global().Get("Promise").New(handler)
	}))

	select {}
}
