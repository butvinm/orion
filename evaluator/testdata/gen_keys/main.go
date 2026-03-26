// gen_keys generates evaluation key fixtures for the bootstrap_mlp test model.
// Produces: bootstrap_mlp.evk (MemEvaluationKeySet), bootstrap_mlp.btpkeys
// (bootstrapping.EvaluationKeys), bootstrap_mlp.sk (SecretKey).
//
// Usage: go run ./evaluator/testdata/gen_keys
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/utils"

	"github.com/butvinm/orion/v2/evaluator"
)

func main() {
	outDir := filepath.Join("..")

	// Load model to get manifest.
	modelPath := filepath.Join(outDir, "bootstrap_mlp.orion")
	data, err := os.ReadFile(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ReadFile(%s): %v\n", modelPath, err)
		os.Exit(1)
	}

	model, err := evaluator.LoadModel(data)
	if err != nil {
		fmt.Fprintf(os.Stderr, "LoadModel: %v\n", err)
		os.Exit(1)
	}

	params, manifest, _ := model.ClientParams()
	ckksParams, err := params.NewCKKSParameters()
	if err != nil {
		fmt.Fprintf(os.Stderr, "NewCKKSParameters: %v\n", err)
		os.Exit(1)
	}

	// Keygen.
	kg := rlwe.NewKeyGenerator(ckksParams)
	sk := kg.GenSecretKeyNew()

	// Evaluation keys.
	var rlk *rlwe.RelinearizationKey
	if manifest.NeedsRLK {
		rlk = kg.GenRelinearizationKeyNew(sk)
	}
	galoisKeys := make([]*rlwe.GaloisKey, 0, len(manifest.GaloisElements))
	for _, ge := range manifest.GaloisElements {
		galoisKeys = append(galoisKeys, kg.GenGaloisKeyNew(ge, sk))
	}
	evk := rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...)

	// Bootstrap keys.
	var btpKeys *bootstrapping.EvaluationKeys
	if len(manifest.BootstrapSlots) > 0 {
		logSlots := int(math.Log2(float64(manifest.BootstrapSlots[0])))
		btpLit := bootstrapping.ParametersLiteral{
			LogN:     utils.Pointy(manifest.BtpLogN),
			LogP:     manifest.BootLogP,
			Xs:       ckksParams.Xs(),
			LogSlots: utils.Pointy(logSlots),
		}
		btpParams, err := bootstrapping.NewParametersFromLiteral(ckksParams, btpLit)
		if err != nil {
			fmt.Fprintf(os.Stderr, "NewParametersFromLiteral: %v\n", err)
			os.Exit(1)
		}
		btpKeys, _, err = btpParams.GenEvaluationKeys(sk)
		if err != nil {
			fmt.Fprintf(os.Stderr, "GenEvaluationKeys: %v\n", err)
			os.Exit(1)
		}
	}

	// Save params JSON (needed by Python evaluator).
	paramsJSON, _ := json.Marshal(params)
	writeFile(filepath.Join(outDir, "bootstrap_mlp.params.json"), paramsJSON)

	// Save SK.
	skData, err := sk.MarshalBinary()
	if err != nil {
		fmt.Fprintf(os.Stderr, "MarshalBinary(sk): %v\n", err)
		os.Exit(1)
	}
	writeFile(filepath.Join(outDir, "bootstrap_mlp.sk"), skData)

	// Save EVK.
	evkData, err := evk.MarshalBinary()
	if err != nil {
		fmt.Fprintf(os.Stderr, "MarshalBinary(evk): %v\n", err)
		os.Exit(1)
	}
	writeFile(filepath.Join(outDir, "bootstrap_mlp.evk"), evkData)

	// Save bootstrap keys.
	if btpKeys != nil {
		btpData, err := btpKeys.MarshalBinary()
		if err != nil {
			fmt.Fprintf(os.Stderr, "MarshalBinary(btpKeys): %v\n", err)
			os.Exit(1)
		}
		writeFile(filepath.Join(outDir, "bootstrap_mlp.btpkeys"), btpData)
		fmt.Printf("Bootstrap keys: %d bytes\n", len(btpData))
	}

	fmt.Println("Done!")
}

func writeFile(path string, data []byte) {
	if err := os.WriteFile(path, data, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "WriteFile(%s): %v\n", path, err)
		os.Exit(1)
	}
	fmt.Printf("Wrote %s (%d bytes)\n", path, len(data))
}
