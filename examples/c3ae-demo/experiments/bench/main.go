// Package main is the C3AE FHE bench binary.
//
// It dispatches to one of four subcommands (keygen, encrypt, infer, decrypt)
// to perform the parts of the FHE pipeline that we want to measure end-to-end
// in a single Go process — eliminating Python wrapper overhead from RSS
// accounting (see docs/plans/2026-05-08-c3ae-experiments.md, Task 8 ff.).
package main

import (
	"fmt"
	"os"
)

func usage(w *os.File) {
	fmt.Fprintf(w, `usage: bench <subcommand> [flags]

Subcommands:
  keygen    Generate secret key + evaluation key set for a compiled .orion model.
  encrypt   Encrypt a raw float64 input blob into a ciphertext.
  infer     Run encrypted forward pass; record forward time and peak RSS.
  decrypt   Decrypt a result ciphertext and print the logit/prob as JSON.

Run 'bench <subcommand> --help' for per-subcommand flags.
`)
}

// subcommands dispatches a subcommand name to its implementation. Each
// implementation lives in a per-file ``run*`` function; this map exists
// solely to remove a redundant cmd*-wrapper layer that simply called
// run* and printed an error prefix.
var subcommands = map[string]func([]string) error{
	"keygen":  runKeygen,
	"encrypt": runEncrypt,
	"infer":   runInfer,
	"decrypt": runDecrypt,
}

func main() {
	if len(os.Args) < 2 {
		usage(os.Stderr)
		os.Exit(2)
	}

	sub := os.Args[1]
	args := os.Args[2:]

	if sub == "-h" || sub == "--help" || sub == "help" {
		usage(os.Stdout)
		return
	}

	fn, ok := subcommands[sub]
	if !ok {
		fmt.Fprintf(os.Stderr, "bench: unknown subcommand %q\n\n", sub)
		usage(os.Stderr)
		os.Exit(2)
	}
	if err := fn(args); err != nil {
		fmt.Fprintf(os.Stderr, "bench %s: %v\n", sub, err)
		os.Exit(1)
	}
}
