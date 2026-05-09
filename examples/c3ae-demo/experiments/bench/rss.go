package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
)

// vmHWMWarnOnce ensures we only emit the "/proc/self/status unavailable"
// warning a single time per process.
var vmHWMWarnOnce sync.Once

// readVmHWM parses /proc/self/status and returns the VmHWM value in kB.
//
// VmHWM is the peak resident set size ("high water mark") of the process,
// reported by the Linux kernel. The line looks like:
//
//	VmHWM:	   12345 kB
//
// Returns 0 on platforms where /proc/self/status does not exist (e.g. macOS)
// or when the file cannot be parsed. A warning is emitted at most once.
func readVmHWM() int64 {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		vmHWMWarnOnce.Do(func() {
			fmt.Fprintf(os.Stderr, "warning: /proc/self/status unavailable (%v); peak_rss_mb will be 0\n", err)
		})
		return 0
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "VmHWM:") {
			continue
		}
		fields := strings.Fields(line)
		// Expected: ["VmHWM:", "<number>", "kB"]
		if len(fields) < 2 {
			vmHWMWarnOnce.Do(func() {
				fmt.Fprintf(os.Stderr, "warning: malformed VmHWM line %q; peak_rss_mb will be 0\n", line)
			})
			return 0
		}
		v, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			vmHWMWarnOnce.Do(func() {
				fmt.Fprintf(os.Stderr, "warning: cannot parse VmHWM value %q: %v; peak_rss_mb will be 0\n", fields[1], err)
			})
			return 0
		}
		return v
	}
	if err := scanner.Err(); err != nil {
		vmHWMWarnOnce.Do(func() {
			fmt.Fprintf(os.Stderr, "warning: scanning /proc/self/status: %v; peak_rss_mb will be 0\n", err)
		})
	}
	return 0
}
