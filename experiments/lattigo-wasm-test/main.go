//go:build js && wasm

package main

import (
	"fmt"
	"syscall/js"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

func main() {
	// Register a simple test function
	js.Global().Set("lattigoTest", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		// Create CKKS parameters matching the MLP demo
		params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN:            13,
			LogQ:            []int{29, 26, 26, 26, 26, 26},
			LogP:            []int{29, 29},
			LogDefaultScale: 26,
			Xs:              ring.Ternary{H: 8192},
			RingType:        ring.ConjugateInvariant,
		})
		if err != nil {
			return fmt.Sprintf("params error: %v", err)
		}

		// Generate keys
		kgen := rlwe.NewKeyGenerator(params)
		sk := kgen.GenSecretKeyNew()
		_ = kgen.GenPublicKeyNew(sk)

		// Encode + Encrypt + Decrypt round-trip
		encoder := ckks.NewEncoder(params)
		encryptor := rlwe.NewEncryptor(params, sk)
		decryptor := rlwe.NewDecryptor(params, sk)

		values := make([]float64, params.MaxSlots())
		for i := range values {
			values[i] = float64(i) * 0.001
		}

		pt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(values, pt); err != nil {
			return fmt.Sprintf("encode error: %v", err)
		}

		ct, err := encryptor.EncryptNew(pt)
		if err != nil {
			return fmt.Sprintf("encrypt error: %v", err)
		}

		ptDec := decryptor.DecryptNew(ct)
		result := make([]float64, params.MaxSlots())
		if err := encoder.Decode(ptDec, result); err != nil {
			return fmt.Sprintf("decode error: %v", err)
		}

		return fmt.Sprintf("OK: slots=%d, first_value=%.6f", params.MaxSlots(), result[0])
	}))

	// Keep Go runtime alive
	select {}
}
