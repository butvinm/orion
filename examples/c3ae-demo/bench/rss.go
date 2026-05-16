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
	return readProcStatusField("VmHWM:")
}

// readVmRSS parses /proc/self/status and returns the VmRSS value in kB.
//
// VmRSS is the current resident set size of the process (live, not peak).
// Sample this between lifecycle phases to attribute RSS deltas to specific
// operations (e.g. post-load vs post-forward), which VmHWM alone can't
// distinguish.
func readVmRSS() int64 {
	return readProcStatusField("VmRSS:")
}

// readProcStatusField is the shared parser for kB-valued lines in
// /proc/self/status. The format is `<prefix>\t<int> kB`. Returns 0 with a
// stderr warning on any IO/parse failure (consistent with readVmHWM's
// "sentinel rather than panic" policy).
func readProcStatusField(prefix string) int64 {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: /proc/self/status unavailable (%v); %s value will be 0\n", err, prefix)
		return 0
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		fields := strings.Fields(line)
		// Expected: ["<prefix>", "<number>", "kB"]
		if len(fields) < 2 {
			fmt.Fprintf(os.Stderr, "warning: malformed %s line %q; value will be 0\n", prefix, line)
			return 0
		}
		v, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: cannot parse %s value %q: %v; value will be 0\n", prefix, fields[1], err)
			return 0
		}
		return v
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: scanning /proc/self/status: %v; %s value will be 0\n", err, prefix)
	}
	return 0
}
