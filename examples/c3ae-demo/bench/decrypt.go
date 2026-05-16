package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	"github.com/butvinm/orion/v2/evaluator"
)

// runDecrypt implements the `bench decrypt` subcommand.
//
// It loads a compiled .orion model (to recover CKKS params), reads a
// previously-generated secret key and a result ciphertext (the output
// of `bench infer`), decrypts + decodes the ciphertext, takes
// decoded[0] as the binary-classifier logit, and prints a compact JSON
// object {"logit": ..., "prob": ...} to stdout where prob is the
// sigmoid of the logit.
func runDecrypt(args []string) error {
	fs := flag.NewFlagSet("decrypt", flag.ExitOnError)
	modelPath := fs.String("model", "", "path to compiled .orion model (required)")
	skPath := fs.String("sk", "", "path to sk.bin produced by `bench keygen` (required)")
	ctPath := fs.String("ct", "", "path to result ciphertext produced by `bench infer` (required)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" {
		return errors.New("--model is required")
	}
	if *skPath == "" {
		return errors.New("--sk is required")
	}
	if *ctPath == "" {
		return errors.New("--ct is required")
	}

	modelBytes, err := os.ReadFile(*modelPath)
	if err != nil {
		return fmt.Errorf("reading model %q: %w", *modelPath, err)
	}

	orionParams, _, _, err := evaluator.ParseClientParams(modelBytes)
	if err != nil {
		return fmt.Errorf("parsing client params: %w", err)
	}

	params, err := orionParams.NewCKKSParameters()
	if err != nil {
		return fmt.Errorf("constructing CKKS parameters: %w", err)
	}

	skBytes, err := os.ReadFile(*skPath)
	if err != nil {
		return fmt.Errorf("reading sk %q: %w", *skPath, err)
	}
	sk := &rlwe.SecretKey{}
	if err := sk.UnmarshalBinary(skBytes); err != nil {
		return fmt.Errorf("unmarshaling sk: %w", err)
	}

	ctBytes, err := os.ReadFile(*ctPath)
	if err != nil {
		return fmt.Errorf("reading result ct %q: %w", *ctPath, err)
	}
	ct := &rlwe.Ciphertext{}
	if err := ct.UnmarshalBinary(ctBytes); err != nil {
		return fmt.Errorf("unmarshaling result ciphertext: %w", err)
	}

	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	pt := decryptor.DecryptNew(ct)

	values := make([]float64, params.MaxSlots())
	if err := encoder.Decode(pt, values); err != nil {
		return fmt.Errorf("decoding plaintext: %w", err)
	}

	logit := values[0]
	prob := 1.0 / (1.0 + math.Exp(-logit))

	out := struct {
		Logit float64 `json:"logit"`
		Prob  float64 `json:"prob"`
	}{
		Logit: logit,
		Prob:  prob,
	}
	b, err := json.Marshal(out)
	if err != nil {
		return fmt.Errorf("marshaling result: %w", err)
	}
	fmt.Println(string(b))
	return nil
}
