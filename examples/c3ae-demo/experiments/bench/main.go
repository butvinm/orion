// Package main is the C3AE FHE bench binary.
//
// It dispatches to one of four subcommands (keygen, encrypt, infer, decrypt)
// to perform the parts of the FHE pipeline that we want to measure end-to-end
// in a single Go process — eliminating Python wrapper overhead from RSS
// accounting (see docs/plans/2026-05-08-c3ae-experiments.md, Task 8 ff.).
//
// The handlers are stubbed in Task 8 and implemented in Tasks 9–12.
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

func cmdKeygen(args []string) {
	if err := runKeygen(args); err != nil {
		fmt.Fprintf(os.Stderr, "bench keygen: %v\n", err)
		os.Exit(1)
	}
}

func cmdEncrypt(args []string) {
	_ = args
	panic("not implemented")
}

func cmdInfer(args []string) {
	_ = args
	panic("not implemented")
}

func cmdDecrypt(args []string) {
	_ = args
	panic("not implemented")
}

func main() {
	if len(os.Args) < 2 {
		usage(os.Stderr)
		os.Exit(2)
	}

	sub := os.Args[1]
	args := os.Args[2:]

	switch sub {
	case "keygen":
		cmdKeygen(args)
	case "encrypt":
		cmdEncrypt(args)
	case "infer":
		cmdInfer(args)
	case "decrypt":
		cmdDecrypt(args)
	case "-h", "--help", "help":
		usage(os.Stdout)
	default:
		fmt.Fprintf(os.Stderr, "bench: unknown subcommand %q\n\n", sub)
		usage(os.Stderr)
		os.Exit(2)
	}
}
