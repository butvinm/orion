package evaluator

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// opProfiler emits one JSONL record per evaluator op, optionally augmented
// with a Go pprof heap snapshot per op. It has two independently-gated
// outputs:
//
//  1. JSONL line per op — enabled when ORION_PROFILE_OUT is set.
//  2. pprof heap snapshot per op — enabled when ORION_HEAP_PROFILE_DIR is set.
//
// The profiler is constructed when EITHER env var is set; the disabled
// path (both unset) returns nil from newOpProfilerFromEnv so all hooks
// are no-ops.
//
// WARNING: enabling ORION_HEAP_PROFILE_DIR forces a runtime.GC() before
// every per-op snapshot to make the dump cleaner. This adds significant
// per-op latency (tens to hundreds of ms depending on heap size). It is
// intended strictly for debugging; do not leave it on in normal runs.
//
// Schema is documented inline in opProfileRecord.
type opProfiler struct {
	// JSONL output. f and enc are nil when only ORION_HEAP_PROFILE_DIR is set.
	f    *os.File
	enc  *json.Encoder
	path string

	// Per-op pprof heap snapshot output. Empty when disabled.
	heapDir string

	forwardStart time.Time
	forwardIdx   int

	// Snapshot of MemStats + VmRSS from the previous recordOp call (or
	// from startForward for the first op of a Forward), used to compute
	// per-op deltas. haveBaseline is set after startForward records the
	// pre-Forward baseline so the first recordOp produces deltas vs the
	// state just before any op ran.
	prevMS       runtime.MemStats
	prevVmRSSMB  int64
	haveBaseline bool
}

// liveObject is one entry of the per-named-output breakdown. Each entry
// corresponds to one key in the evaluator's `results` map after the op
// just completed — i.e. one named output the evaluator is currently
// holding alive. Sorted by Bytes desc in the emitted JSONL line so the
// largest holders surface first when reading.
type liveObject struct {
	Node     string `json:"node"`
	NCTs     int    `json:"n_cts"`
	MaxLevel int    `json:"max_level"`
	Bytes    int64  `json:"bytes"`
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

	// Per-named-output breakdown. One entry per key in the evaluator's
	// `results` map, sorted by Bytes desc.
	LiveObjects []liveObject `json:"live_objects"`

	// Counts of live ciphertexts grouped by level. Map keys are
	// stringified ints (Go encodes int-keyed maps as strings for JSON
	// anyway; we do it explicitly to make the type honest).
	LevelHistogram map[string]int `json:"level_histogram"`

	// Per-op deltas vs the previous recordOp call (or the startForward
	// baseline for the first op).
	//
	// HeapAllocDeltaMB: bytes allocated by Go since the previous snapshot,
	// derived from runtime.MemStats.TotalAlloc (a monotonic counter, so
	// always >= 0). Reflects allocator pressure introduced by this op.
	//
	// HeapFreedDeltaMB: bytes the Go runtime returned to the OS since the
	// previous snapshot, derived from runtime.MemStats.HeapReleased. Note
	// this is the OS-level release counter, not the in-process free
	// counter — releases happen lazily and only after GC, so this can be
	// 0 for many ops in a row and then jump.
	//
	// NumGCDelta: number of completed GC cycles since the previous
	// snapshot.
	//
	// VmRSSDeltaMB: signed change in /proc/self/status VmRSS. Can be
	// negative if the kernel reclaimed pages between snapshots.
	HeapAllocDeltaMB int64  `json:"heap_alloc_delta_mb"`
	HeapFreedDeltaMB int64  `json:"heap_freed_delta_mb"`
	NumGCDelta       uint32 `json:"num_gc_delta"`
	VmRSSDeltaMB     int64  `json:"vmrss_delta_mb"`

	// Path to the per-op pprof heap snapshot, present only when
	// ORION_HEAP_PROFILE_DIR was set AND the snapshot was written
	// successfully. Use `go tool pprof <path>` to inspect.
	HeapPprofPath string `json:"heap_pprof_path,omitempty"`
}

// newOpProfilerFromEnv returns nil when BOTH ORION_PROFILE_OUT and
// ORION_HEAP_PROFILE_DIR are unset (the hot, common path). When at least
// one is set, the profiler is constructed; the disabled output (file or
// dir) is left empty.
//
// JSONL file open failure logs once to stderr and disables the JSONL
// output but keeps heap profiling alive (and vice versa). If both
// outputs fail to initialize the profiler returns nil — profiling is
// best-effort and must never break inference.
func newOpProfilerFromEnv() *opProfiler {
	jsonlPath := os.Getenv("ORION_PROFILE_OUT")
	heapDir := os.Getenv("ORION_HEAP_PROFILE_DIR")
	if jsonlPath == "" && heapDir == "" {
		return nil
	}

	p := &opProfiler{}

	if jsonlPath != "" {
		f, err := os.OpenFile(jsonlPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: ORION_PROFILE_OUT=%q open failed: %v; JSONL profiling disabled for this Forward\n", jsonlPath, err)
		} else {
			p.f = f
			p.enc = json.NewEncoder(f)
			p.path = jsonlPath
		}
	}

	if heapDir != "" {
		if err := os.MkdirAll(heapDir, 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "warning: ORION_HEAP_PROFILE_DIR=%q mkdir failed: %v; heap profiling disabled for this Forward\n", heapDir, err)
		} else {
			p.heapDir = heapDir
		}
	}

	if p.f == nil && p.heapDir == "" {
		return nil
	}
	return p
}

// startForward resets the per-Forward relative timestamp baseline and
// records the initial MemStats + VmRSS so the first recordOp's deltas
// are computed against the state just before any op ran. Safe on nil.
func (p *opProfiler) startForward() {
	if p == nil {
		return
	}
	p.forwardStart = time.Now()
	p.forwardIdx = 0

	runtime.ReadMemStats(&p.prevMS)
	p.prevVmRSSMB = readMemKB("VmRSS:") / 1024
	p.haveBaseline = true
}

// filenameSanitizer scrubs characters that are awkward in filesystem
// paths. Used for the heap-snapshot filename so node names like
// "block1/conv:0" don't leak as literal directory separators.
var filenameSanitizer = strings.NewReplacer(
	"/", "_",
	":", "_",
	" ", "_",
	"\\", "_",
)

// recordOp writes one JSONL line describing the just-completed op and,
// when ORION_HEAP_PROFILE_DIR is set, also writes a per-op pprof heap
// snapshot. Safe on nil.
//
// The mem-stats snapshot (runtime.MemStats + /proc/self/status read) is
// taken AFTER the op completes, so the record reflects post-op memory
// pressure. This matches what we care about: did this op blow up the
// heap?
//
// WARNING: when heap profiling is enabled, this calls runtime.GC()
// before each snapshot. That adds significant per-op latency and is for
// debugging only.
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

	// If heap profiling is enabled, run GC before the snapshot so the
	// dump reflects the steady-state heap rather than transient garbage.
	// This is the latency-adding step we warn about in the docstring.
	if p.heapDir != "" {
		runtime.GC()
	}

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	curVmRSSMB := readMemKB("VmRSS:") / 1024

	// Walk all live result entries.
	//
	// First pass: per-named-output aggregation. The same loop also
	// produces the totals (aliveCount, aliveBytes, maxLevel) and the
	// level histogram so we walk the map exactly once. This is a
	// hundred-ish entries at most; cost is negligible vs an FHE op.
	n := int64(params.N())
	aliveCount := 0
	var aliveBytes int64
	maxLevel := -1
	liveObjects := make([]liveObject, 0, len(results))
	levelHist := make(map[string]int)

	for nodeName, cts := range results {
		entryCount := 0
		var entryBytes int64
		entryMaxLevel := -1
		for _, ct := range cts {
			if ct == nil {
				continue
			}
			entryCount++
			lvl := ct.Level()
			if lvl > entryMaxLevel {
				entryMaxLevel = lvl
			}
			// 2 polynomials * (level+1) limbs * N coefficients * 8 bytes.
			entryBytes += 2 * int64(lvl+1) * n * 8
			levelHist[strconv.Itoa(lvl)]++
		}
		if entryCount == 0 {
			// Skip empty entries — they tell us nothing and would clutter
			// the JSONL line.
			continue
		}
		liveObjects = append(liveObjects, liveObject{
			Node:     nodeName,
			NCTs:     entryCount,
			MaxLevel: entryMaxLevel,
			Bytes:    entryBytes,
		})
		aliveCount += entryCount
		aliveBytes += entryBytes
		if entryMaxLevel > maxLevel {
			maxLevel = entryMaxLevel
		}
	}
	sort.Slice(liveObjects, func(i, j int) bool {
		return liveObjects[i].Bytes > liveObjects[j].Bytes
	})

	// Compute deltas vs the previous snapshot. If for some reason
	// startForward wasn't called (shouldn't happen — Forward always
	// calls it), fall back to zero deltas rather than computing against
	// a zero MemStats which would produce huge garbage values.
	var (
		heapAllocDeltaMB int64
		heapFreedDeltaMB int64
		numGCDelta       uint32
		vmRSSDeltaMB     int64
	)
	if p.haveBaseline {
		// TotalAlloc is a monotonic counter so this subtraction is always
		// >= 0 in practice (and uint64 underflow would only happen if we
		// ran for ~years).
		heapAllocDeltaMB = int64((ms.TotalAlloc - p.prevMS.TotalAlloc) / (1024 * 1024))
		// HeapReleased is also monotonic. This reflects bytes returned
		// to the OS, not bytes "freed" inside the Go heap (those go back
		// into the free pool and may be re-used without an OS-level
		// release). Documented in the field's comment above.
		heapFreedDeltaMB = int64((ms.HeapReleased - p.prevMS.HeapReleased) / (1024 * 1024))
		numGCDelta = ms.NumGC - p.prevMS.NumGC
		vmRSSDeltaMB = curVmRSSMB - p.prevVmRSSMB
	}

	// Write the per-op pprof heap snapshot before encoding the JSONL
	// line so the snapshot path can be embedded in the record.
	heapPprofPath := ""
	if p.heapDir != "" {
		safeName := filenameSanitizer.Replace(name)
		fname := fmt.Sprintf("op_%04d_%s.pprof", idx, safeName)
		fpath := filepath.Join(p.heapDir, fname)
		if err := writeHeapProfile(fpath); err != nil {
			fmt.Fprintf(os.Stderr, "warning: heap pprof write failed at idx=%d path=%q: %v\n", idx, fpath, err)
		} else {
			heapPprofPath = fpath
		}
	}

	// Only encode JSONL if that output is enabled. The two outputs are
	// independently gated; either may be on alone.
	if p.enc != nil {
		rec := opProfileRecord{
			Idx:                idx,
			Name:               name,
			Op:                 op,
			InCT:               inCT,
			OutCT:              outCT,
			OpMS:               float64(dur.Nanoseconds()) / 1e6,
			VmRSSMB:            curVmRSSMB,
			VmHWMMB:            readMemKB("VmHWM:") / 1024,
			HeapAllocMB:        ms.HeapAlloc / (1024 * 1024),
			HeapInuseMB:        ms.HeapInuse / (1024 * 1024),
			HeapSysMB:          ms.HeapSys / (1024 * 1024),
			NumGC:              ms.NumGC,
			AliveCTs:           aliveCount,
			AliveCTBytes:       aliveBytes,
			MaxAliveLevel:      maxLevel,
			SinceForwardStartS: time.Since(p.forwardStart).Seconds(),
			LiveObjects:        liveObjects,
			LevelHistogram:     levelHist,
			HeapAllocDeltaMB:   heapAllocDeltaMB,
			HeapFreedDeltaMB:   heapFreedDeltaMB,
			NumGCDelta:         numGCDelta,
			VmRSSDeltaMB:       vmRSSDeltaMB,
			HeapPprofPath:      heapPprofPath,
		}

		if err := p.enc.Encode(&rec); err != nil {
			fmt.Fprintf(os.Stderr, "warning: ORION_PROFILE_OUT write failed at idx=%d: %v\n", idx, err)
		} else if err := p.f.Sync(); err != nil {
			// Sync after each line so a crashed Forward (e.g. OOM kill)
			// still leaves a usable trace on disk. The cost is small
			// relative to op_ms (FHE ops dwarf one fsync) and
			// crash-resilience is the whole point of this file.
			fmt.Fprintf(os.Stderr, "warning: ORION_PROFILE_OUT fsync failed at idx=%d: %v\n", idx, err)
		}
	}

	// Update baseline for the next op's deltas.
	p.prevMS = ms
	p.prevVmRSSMB = curVmRSSMB
}

// writeHeapProfile dumps the current heap profile to path. Returns the
// first error encountered (open / write / close). The caller logs and
// continues on error — pprof failures must never break inference.
func writeHeapProfile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create: %w", err)
	}
	if err := pprof.Lookup("heap").WriteTo(f, 0); err != nil {
		_ = f.Close()
		return fmt.Errorf("write: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close: %w", err)
	}
	return nil
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
