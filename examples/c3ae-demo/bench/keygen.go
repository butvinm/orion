package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"

	"github.com/butvinm/orion/v2/evaluator"
)

// runKeygen implements the `bench keygen` subcommand.
//
// It loads a compiled .orion model, generates a fresh secret key plus an
// evaluation key set (RLK + Galois keys per the model's manifest), and
// writes sk.bin / evk.bin / keygen.json into the --out directory.
//
// This subcommand explicitly does NOT support bootstrap configurations.
// Both supported configs (logn15, logn16) are no-bootstrap. We defensively
// error out if the loaded model's manifest has any BootstrapSlots.
func runKeygen(args []string) error {
	fs := flag.NewFlagSet("keygen", flag.ExitOnError)
	modelPath := fs.String("model", "", "path to compiled .orion model (required)")
	outDir := fs.String("out", "", "output directory for sk.bin/evk.bin/keygen.json (required)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" {
		return errors.New("--model is required")
	}
	if *outDir == "" {
		return errors.New("--out is required")
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		return fmt.Errorf("creating out dir %q: %w", *outDir, err)
	}

	modelBytes, err := os.ReadFile(*modelPath)
	if err != nil {
		return fmt.Errorf("reading model %q: %w", *modelPath, err)
	}

	model, err := evaluator.LoadModel(modelBytes)
	if err != nil {
		return fmt.Errorf("loading model: %w", err)
	}

	orionParams, manifest, _ := model.ClientParams()

	// Defensive: bench is for no-bootstrap configs only.
	if len(manifest.BootstrapSlots) > 0 {
		return fmt.Errorf(
			"bench does not support bootstrap configs; got non-empty BootstrapSlots: %v",
			manifest.BootstrapSlots,
		)
	}

	params, err := orionParams.NewCKKSParameters()
	if err != nil {
		return fmt.Errorf("constructing CKKS parameters: %w", err)
	}

	t0 := time.Now()

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	var rlk *rlwe.RelinearizationKey
	if manifest.NeedsRLK {
		rlk = kgen.GenRelinearizationKeyNew(sk)
	}

	galKeys := make([]*rlwe.GaloisKey, 0, len(manifest.GaloisElements))
	for _, ge := range manifest.GaloisElements {
		galKeys = append(galKeys, kgen.GenGaloisKeyNew(ge, sk))
	}

	evk := rlwe.NewMemEvaluationKeySet(rlk, galKeys...)

	keygenSecs := time.Since(t0).Seconds()

	skBytes, err := sk.MarshalBinary()
	if err != nil {
		return fmt.Errorf("marshaling secret key: %w", err)
	}
	skPath := filepath.Join(*outDir, "sk.bin")
	if err := os.WriteFile(skPath, skBytes, 0o600); err != nil {
		return fmt.Errorf("writing %q: %w", skPath, err)
	}
	// os.WriteFile only applies the permission mode on *creation*; if
	// sk.bin pre-existed (e.g. from an earlier interrupted run) with looser
	// perms, those would persist. Force the permissions to match.
	if err := os.Chmod(skPath, 0o600); err != nil {
		return fmt.Errorf("chmod %q: %w", skPath, err)
	}

	evkBytes, err := evk.MarshalBinary()
	if err != nil {
		return fmt.Errorf("marshaling evaluation key set: %w", err)
	}
	evkPath := filepath.Join(*outDir, "evk.bin")
	if err := os.WriteFile(evkPath, evkBytes, 0o644); err != nil {
		return fmt.Errorf("writing %q: %w", evkPath, err)
	}

	meta := struct {
		KeygenS  float64 `json:"keygen_s"`
		EVKBytes int     `json:"evk_bytes"`
	}{
		KeygenS:  keygenSecs,
		EVKBytes: len(evkBytes),
	}
	metaBytes, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling keygen.json: %w", err)
	}
	metaPath := filepath.Join(*outDir, "keygen.json")
	if err := os.WriteFile(metaPath, metaBytes, 0o644); err != nil {
		return fmt.Errorf("writing %q: %w", metaPath, err)
	}

	fmt.Fprintf(os.Stdout,
		"keygen: %.2fs, sk=%d bytes, evk=%d bytes (galois=%d, rlk=%v)\n",
		keygenSecs, len(skBytes), len(evkBytes),
		len(manifest.GaloisElements), manifest.NeedsRLK,
	)
	return nil
}
