package evaluator

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// opProfiler emits one JSONL record per evaluator op when ORION_PROFILE_OUT
// is set. It is intentionally cheap to construct (single env-var read) and a
// no-op on the nil receiver, so the disabled path adds essentially zero
// overhead to Forward.
//
// Schema is documented inline in opProfileRecord.
type opProfiler struct {
	f            *os.File
	enc          *json.Encoder
	path         string
	forwardStart time.Time
	forwardIdx   int
}

// opProfileRecord is the JSONL line schema. Field tags must stay stable —
// downstream tooling (analysis notebooks, future dashboards) consumes them.
type opProfileRecord struct {
	Idx                int     `json:"idx"`
	Name               string  `json:"name"`
	Op                 string  `json:"op"`
	InCT               int     `json:"in_ct"`
	OutCT              int     `json:"out_ct"`
	OpMS               float64 `json:"op_ms"`
	VmRSSMB            int64   `json:"vmrss_mb"`
	VmHWMMB            int64   `json:"vmhwm_mb"`
	HeapAllocMB        uint64  `json:"heap_alloc_mb"`
	HeapInuseMB        uint64  `json:"heap_inuse_mb"`
	HeapSysMB          uint64  `json:"heap_sys_mb"`
	NumGC              uint32  `json:"num_gc"`
	AliveCTs           int     `json:"alive_cts"`
	AliveCTBytes       int64   `json:"alive_ct_bytes"`
	MaxAliveLevel      int     `json:"max_alive_level"`
	SinceForwardStartS float64 `json:"since_forward_start_s"`
}

// newOpProfilerFromEnv returns nil when ORION_PROFILE_OUT is unset (the
// hot, common path). When set, it opens the file in append mode so that
// per-Forward profiles append into the same file across calls. Failure to
// open is logged once to stderr and yields nil — profiling is best-effort
// and must never break inference.
func newOpProfilerFromEnv() *opProfiler {
	path := os.Getenv("ORION_PROFILE_OUT")
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: ORION_PROFILE_OUT=%q open failed: %v; profiling disabled for this Forward\n", path, err)
		return nil
	}
	return &opProfiler{
		f:    f,
		enc:  json.NewEncoder(f),
		path: path,
	}
}

// startForward resets the per-Forward relative timestamp baseline. Safe on nil.
func (p *opProfiler) startForward() {
	if p == nil {
		return
	}
	p.forwardStart = time.Now()
	p.forwardIdx = 0
}

// recordOp writes one JSONL line describing the just-completed op. Safe on nil.
//
// The mem-stats snapshot (runtime.MemStats + /proc/self/status read) is taken
// AFTER the op completes, so the record reflects post-op memory pressure.
// This matches what we care about: did this op blow up the heap?
func (p *opProfiler) recordOp(
	idx int,
	name, op string,
	inCT, outCT int,
	dur time.Duration,
	results map[string][]*rlwe.Ciphertext,
	params ckks.Parameters,
) {
	if p == nil {
		return
	}

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	// Walk all live result entries. This is a snapshot of what the evaluator
	// is currently holding — it does NOT include Lattigo's internal buffers
	// or the bootstrapper precomputations.
	aliveCount := 0
	var aliveBytes int64
	maxLevel := -1
	n := int64(params.N())
	for _, cts := range results {
		for _, ct := range cts {
			if ct == nil {
				continue
			}
			aliveCount++
			lvl := ct.Level()
			if lvl > maxLevel {
				maxLevel = lvl
			}
			// 2 polynomials * (level+1) limbs * N coefficients * 8 bytes.
			aliveBytes += 2 * int64(lvl+1) * n * 8
		}
	}

	rec := opProfileRecord{
		Idx:                idx,
		Name:               name,
		Op:                 op,
		InCT:               inCT,
		OutCT:              outCT,
		OpMS:               float64(dur.Nanoseconds()) / 1e6,
		VmRSSMB:            readMemKB("VmRSS:") / 1024,
		VmHWMMB:            readMemKB("VmHWM:") / 1024,
		HeapAllocMB:        ms.HeapAlloc / (1024 * 1024),
		HeapInuseMB:        ms.HeapInuse / (1024 * 1024),
		HeapSysMB:          ms.HeapSys / (1024 * 1024),
		NumGC:              ms.NumGC,
		AliveCTs:           aliveCount,
		AliveCTBytes:       aliveBytes,
		MaxAliveLevel:      maxLevel,
		SinceForwardStartS: time.Since(p.forwardStart).Seconds(),
	}

	if err := p.enc.Encode(&rec); err != nil {
		fmt.Fprintf(os.Stderr, "warning: ORION_PROFILE_OUT write failed at idx=%d: %v\n", idx, err)
		return
	}
	// Sync after each line so a crashed Forward (e.g. OOM kill) still leaves
	// a usable trace on disk. The cost is small relative to op_ms (FHE ops
	// dwarf one fsync), and crash-resilience is the whole point of this file.
	if err := p.f.Sync(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: ORION_PROFILE_OUT fsync failed at idx=%d: %v\n", idx, err)
	}
}

// close flushes and closes the underlying file. Safe on nil.
func (p *opProfiler) close() {
	if p == nil {
		return
	}
	if p.f != nil {
		_ = p.f.Sync()
		_ = p.f.Close()
		p.f = nil
	}
}

// readMemKB scans /proc/self/status for a field like "VmRSS:" or "VmHWM:"
// and returns its value in kB. Returns 0 on non-Linux or any read error.
//
// The field argument MUST include the trailing colon to avoid false matches
// (e.g. "Vm" would match VmPeak, VmSize, etc.).
func readMemKB(field string) int64 {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return 0
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, field) {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		v, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			return 0
		}
		return v
	}
	return 0
}
