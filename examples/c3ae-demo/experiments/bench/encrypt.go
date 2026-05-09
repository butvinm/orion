package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	"github.com/butvinm/orion/v2/evaluator"
)

// runEncrypt implements the `bench encrypt` subcommand.
//
// It loads a compiled .orion model (to recover CKKS params and the
// model's required input level), reads a previously-generated secret
// key, reads a raw little-endian float64 input blob (12288 values for
// the C3AE 3x64x64 input), encodes + SK-encrypts the values at the
// model's input level, and writes the marshaled ciphertext to --out.
//
// SK-mode encryption is intentional: bench is a single-party local
// simulator, so there is no client/server boundary inside the process
// (see plan, "Why SK-mode encryption?").
func runEncrypt(args []string) error {
	fs := flag.NewFlagSet("encrypt", flag.ExitOnError)
	modelPath := fs.String("model", "", "path to compiled .orion model (required)")
	skPath := fs.String("sk", "", "path to sk.bin produced by `bench keygen` (required)")
	inputPath := fs.String("input", "", "path to raw little-endian float64 input blob (required)")
	outPath := fs.String("out", "", "output ciphertext file path (required)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" {
		return fmt.Errorf("--model is required")
	}
	if *skPath == "" {
		return fmt.Errorf("--sk is required")
	}
	if *inputPath == "" {
		return fmt.Errorf("--input is required")
	}
	if *outPath == "" {
		return fmt.Errorf("--out is required")
	}

	modelBytes, err := os.ReadFile(*modelPath)
	if err != nil {
		return fmt.Errorf("reading model %q: %w", *modelPath, err)
	}

	model, err := evaluator.LoadModel(modelBytes)
	if err != nil {
		return fmt.Errorf("loading model: %w", err)
	}

	orionParams, _, inputLevel := model.ClientParams()

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

	inputBytes, err := os.ReadFile(*inputPath)
	if err != nil {
		return fmt.Errorf("reading input %q: %w", *inputPath, err)
	}
	if len(inputBytes)%8 != 0 {
		return fmt.Errorf(
			"input file size %d is not a multiple of 8 (expected raw little-endian float64)",
			len(inputBytes),
		)
	}
	nVals := len(inputBytes) / 8
	maxSlots := params.MaxSlots()
	if nVals > maxSlots {
		return fmt.Errorf(
			"input has %d float64 values, exceeds params.MaxSlots()=%d",
			nVals, maxSlots,
		)
	}

	values := make([]float64, maxSlots)
	for i := 0; i < nVals; i++ {
		bits := binary.LittleEndian.Uint64(inputBytes[i*8:])
		values[i] = math.Float64frombits(bits)
	}
	// values[nVals:maxSlots] is implicit zero-padding from make().

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, sk)

	pt := ckks.NewPlaintext(params, inputLevel)
	pt.Scale = params.DefaultScale()
	if err := encoder.Encode(values, pt); err != nil {
		return fmt.Errorf("encoding input: %w", err)
	}

	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		return fmt.Errorf("encrypting plaintext: %w", err)
	}

	ctBytes, err := ct.MarshalBinary()
	if err != nil {
		return fmt.Errorf("marshaling ciphertext: %w", err)
	}
	if err := os.WriteFile(*outPath, ctBytes, 0o644); err != nil {
		return fmt.Errorf("writing %q: %w", *outPath, err)
	}

	fmt.Fprintf(os.Stdout,
		"encrypt: %d float64 values (padded to %d slots) at level=%d, ct=%d bytes\n",
		nVals, maxSlots, inputLevel, len(ctBytes),
	)
	return nil
}
