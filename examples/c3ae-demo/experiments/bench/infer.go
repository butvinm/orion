package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"

	"github.com/butvinm/orion/v2/evaluator"
)

// inferMetrics is the JSONL record emitted per `bench infer` invocation.
//
// One line is appended to --metrics per call (open with O_APPEND|O_CREATE).
// The aggregator (build_results.py, Task 15) consumes this file.
type inferMetrics struct {
	SampleIdx     int     `json:"sample_idx"`
	ForwardS      float64 `json:"forward_s"`
	PeakRSSMB     int64   `json:"peak_rss_mb"`
	ResultCTBytes int     `json:"result_ct_bytes"`
}

// runInfer implements the `bench infer` subcommand — the one whose timing
// and peak RSS we actually report. Everything outside the measured section
// is setup/IO; the measured section is just `eval.Forward(...)`.
//
// peak_rss_mb is read from /proc/self/status (VmHWM, kB) and converted to
// MB. On non-Linux it falls back to 0 with a warning.
func runInfer(args []string) error {
	fs := flag.NewFlagSet("infer", flag.ExitOnError)
	modelPath := fs.String("model", "", "path to compiled .orion model (required)")
	evkPath := fs.String("evk", "", "path to evk.bin produced by `bench keygen` (required)")
	ctPath := fs.String("ct", "", "path to input ciphertext produced by `bench encrypt` (required)")
	outPath := fs.String("out", "", "output ciphertext file path (required)")
	metricsPath := fs.String("metrics", "", "JSONL metrics file (append+create) (required)")
	sampleIdx := fs.Int("sample-idx", -1, "sample index for metrics tagging (required)")
	profilePath := fs.String("profile", "", "optional per-op JSONL profile output path; sets ORION_PROFILE_OUT for the evaluator")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" {
		return fmt.Errorf("--model is required")
	}
	if *evkPath == "" {
		return fmt.Errorf("--evk is required")
	}
	if *ctPath == "" {
		return fmt.Errorf("--ct is required")
	}
	if *outPath == "" {
		return fmt.Errorf("--out is required")
	}
	if *metricsPath == "" {
		return fmt.Errorf("--metrics is required")
	}
	if *sampleIdx < 0 {
		return fmt.Errorf("--sample-idx is required (must be >= 0)")
	}

	modelBytes, err := os.ReadFile(*modelPath)
	if err != nil {
		return fmt.Errorf("reading model %q: %w", *modelPath, err)
	}

	model, err := evaluator.LoadModel(modelBytes)
	if err != nil {
		return fmt.Errorf("loading model: %w", err)
	}

	orionParams, _, _ := model.ClientParams()
	params, err := orionParams.NewCKKSParameters()
	if err != nil {
		return fmt.Errorf("constructing CKKS parameters: %w", err)
	}

	evkBytes, err := os.ReadFile(*evkPath)
	if err != nil {
		return fmt.Errorf("reading evk %q: %w", *evkPath, err)
	}
	evk := &rlwe.MemEvaluationKeySet{}
	if err := evk.UnmarshalBinary(evkBytes); err != nil {
		return fmt.Errorf("unmarshaling evk: %w", err)
	}

	ctBytes, err := os.ReadFile(*ctPath)
	if err != nil {
		return fmt.Errorf("reading input ct %q: %w", *ctPath, err)
	}
	ct := &rlwe.Ciphertext{}
	if err := ct.UnmarshalBinary(ctBytes); err != nil {
		return fmt.Errorf("unmarshaling input ciphertext: %w", err)
	}

	// No bootstrap keys: both supported configs are no-bootstrap.
	eval, err := evaluator.NewEvaluatorFromKeySet(params, evk, nil)
	if err != nil {
		return fmt.Errorf("constructing evaluator: %w", err)
	}

	// Activate per-op profiler if requested. The evaluator reads
	// ORION_PROFILE_OUT inside Forward; setting it here scopes profiling to
	// this single bench invocation.
	if *profilePath != "" {
		if err := os.Setenv("ORION_PROFILE_OUT", *profilePath); err != nil {
			return fmt.Errorf("setting ORION_PROFILE_OUT: %w", err)
		}
		fmt.Fprintf(os.Stderr, "infer: per-op profile -> %s\n", *profilePath)
	}

	// --- Measured section ---
	t0 := time.Now()
	result, err := eval.Forward(model, []*rlwe.Ciphertext{ct})
	forwardS := time.Since(t0).Seconds()
	if err != nil {
		return fmt.Errorf("eval.Forward: %w", err)
	}
	peakRSSMB := readVmHWM() / 1024
	// --- End measured section ---

	if len(result) == 0 {
		return fmt.Errorf("eval.Forward returned empty result slice")
	}

	resultBytes, err := result[0].MarshalBinary()
	if err != nil {
		return fmt.Errorf("marshaling result ciphertext: %w", err)
	}
	if err := os.WriteFile(*outPath, resultBytes, 0o644); err != nil {
		return fmt.Errorf("writing %q: %w", *outPath, err)
	}

	metrics := inferMetrics{
		SampleIdx:     *sampleIdx,
		ForwardS:      forwardS,
		PeakRSSMB:     peakRSSMB,
		ResultCTBytes: len(resultBytes),
	}
	metricsLine, err := json.Marshal(metrics)
	if err != nil {
		return fmt.Errorf("marshaling metrics: %w", err)
	}

	mf, err := os.OpenFile(*metricsPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("opening metrics %q: %w", *metricsPath, err)
	}
	// Defer Close so it runs on every exit path. The explicit Sync below
	// flushes data to disk before we return; Close at defer time still
	// reports any error from kernel-side close (e.g. NFS write-back errors).
	defer func() { _ = mf.Close() }()
	if _, err := mf.Write(append(metricsLine, '\n')); err != nil {
		return fmt.Errorf("writing metrics: %w", err)
	}
	if err := mf.Sync(); err != nil {
		return fmt.Errorf("syncing metrics: %w", err)
	}

	fmt.Fprintf(os.Stdout,
		"infer: sample_idx=%d forward_s=%.3f peak_rss_mb=%d result_ct=%d bytes\n",
		*sampleIdx, forwardS, peakRSSMB, len(resultBytes),
	)
	return nil
}
