package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// readVmHWM parses /proc/self/status and returns the VmHWM value in kB.
//
// VmHWM is the peak resident set size ("high water mark") of the process,
// reported by the Linux kernel. The line looks like:
//
//	VmHWM:	   12345 kB
//
// This experiment harness targets Linux only (the dependent CGO Lattigo
// build is Linux-only at logn=15+ scales); /proc/self/status is therefore
// expected to exist. If it doesn't, we return 0 and emit a single warning
// to stderr — non-Linux callers get a sentinel rather than a panic so the
// rest of the bench remains usable for unit-style sanity checks.
func readVmHWM() int64 {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: /proc/self/status unavailable (%v); peak_rss_mb will be 0\n", err)
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
			fmt.Fprintf(os.Stderr, "warning: malformed VmHWM line %q; peak_rss_mb will be 0\n", line)
			return 0
		}
		v, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: cannot parse VmHWM value %q: %v; peak_rss_mb will be 0\n", fields[1], err)
			return 0
		}
		return v
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: scanning /proc/self/status: %v; peak_rss_mb will be 0\n", err)
	}
	return 0
}
