package main

import (
	"encoding/json"
	"errors"
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
// The aggregator (scripts/build_results.py) consumes this file.
//
// Three RSS sample points are recorded in addition to the end-of-run VmHWM
// peak, so we can attribute RSS deltas to specific lifecycle phases (load,
// first forward, second forward) instead of conflating them into a single
// peak number. The plan at docs/plans/20260516-pre-encode-lintrans.md Task 4
// requires this for the pre-encode-lintrans acceptance gates A/B/C.
//
// LoadS / Forward2S also live here so we can compute the load-time delta
// (eager encoding moves work from Forward to LoadModel) and confirm that a
// second Forward on the same evaluator/model is byte-stable.
type inferMetrics struct {
	SampleIdx        int     `json:"sample_idx"`
	LoadS            float64 `json:"load_s"`
	ForwardS         float64 `json:"forward_s"`
	Forward2S        float64 `json:"forward2_s"`
	RSSPostLoadMB    int64   `json:"rss_post_load_mb"`
	RSSPostForward1MB int64  `json:"rss_post_forward1_mb"`
	RSSPostForward2MB int64  `json:"rss_post_forward2_mb"`
	PeakRSSMB        int64   `json:"peak_rss_mb"`
	ResultCTBytes    int     `json:"result_ct_bytes"`
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
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" {
		return errors.New("--model is required")
	}
	if *evkPath == "" {
		return errors.New("--evk is required")
	}
	if *ctPath == "" {
		return errors.New("--ct is required")
	}
	if *outPath == "" {
		return errors.New("--out is required")
	}
	if *metricsPath == "" {
		return errors.New("--metrics is required")
	}
	if *sampleIdx < 0 {
		return errors.New("--sample-idx is required (must be >= 0)")
	}

	modelBytes, err := os.ReadFile(*modelPath)
	if err != nil {
		return fmt.Errorf("reading model %q: %w", *modelPath, err)
	}

	// --- Measured: model load + evaluator setup ---
	// LoadModel is where eager LT pre-encoding happens on the feature
	// branch. Time it explicitly so we can compute the load-time delta
	// (feature − baseline) called out in the plan's Acceptance gate B.
	loadStart := time.Now()
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
	loadS := time.Since(loadStart).Seconds()
	// Sample RSS after load + evaluator construction, before any Forward.
	// On the feature branch this captures the post-eager-encode resident
	// set; on baseline this captures load-time resident only (no LT
	// encodings yet — those happen lazily inside Forward).
	rssPostLoadMB := readVmRSS() / 1024

	// --- Measured: first Forward ---
	t0 := time.Now()
	result, err := eval.Forward(model, []*rlwe.Ciphertext{ct})
	forwardS := time.Since(t0).Seconds()
	if err != nil {
		return fmt.Errorf("eval.Forward: %w", err)
	}
	rssPostForward1MB := readVmRSS() / 1024

	// --- Measured: second Forward on same model + evaluator ---
	// Re-unmarshal the input ciphertext: Forward consumes/mutates ct
	// internals via the Lattigo evaluator buffers. Using a fresh CT
	// guarantees the second Forward has a clean input identical to the
	// first.
	ct2 := &rlwe.Ciphertext{}
	if err := ct2.UnmarshalBinary(ctBytes); err != nil {
		return fmt.Errorf("unmarshaling input ciphertext for second forward: %w", err)
	}
	t1 := time.Now()
	result2, err := eval.Forward(model, []*rlwe.Ciphertext{ct2})
	forward2S := time.Since(t1).Seconds()
	if err != nil {
		return fmt.Errorf("eval.Forward (second call): %w", err)
	}
	rssPostForward2MB := readVmRSS() / 1024

	// Touch result2 (just enough to keep the compiler from being
	// over-eager about elimination — though the slice escapes through
	// MarshalBinary below in practice).
	_ = result2

	peakRSSMB := readVmHWM() / 1024
	// --- End measured section ---

	if len(result) == 0 {
		return errors.New("eval.Forward returned empty result slice")
	}

	resultBytes, err := result[0].MarshalBinary()
	if err != nil {
		return fmt.Errorf("marshaling result ciphertext: %w", err)
	}
	if err := os.WriteFile(*outPath, resultBytes, 0o644); err != nil {
		return fmt.Errorf("writing %q: %w", *outPath, err)
	}

	metrics := inferMetrics{
		SampleIdx:         *sampleIdx,
		LoadS:             loadS,
		ForwardS:          forwardS,
		Forward2S:         forward2S,
		RSSPostLoadMB:     rssPostLoadMB,
		RSSPostForward1MB: rssPostForward1MB,
		RSSPostForward2MB: rssPostForward2MB,
		PeakRSSMB:         peakRSSMB,
		ResultCTBytes:     len(resultBytes),
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
		"infer: sample_idx=%d load_s=%.3f forward_s=%.3f forward2_s=%.3f "+
			"rss_post_load_mb=%d rss_post_forward1_mb=%d rss_post_forward2_mb=%d "+
			"peak_rss_mb=%d result_ct=%d bytes\n",
		*sampleIdx, loadS, forwardS, forward2S,
		rssPostLoadMB, rssPostForward1MB, rssPostForward2MB,
		peakRSSMB, len(resultBytes),
	)
	return nil
}
