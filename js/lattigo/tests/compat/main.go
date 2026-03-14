// Cross-platform serialization test: verifies that bytes marshaled by the WASM
// bridge can be unmarshaled by native Go (same Lattigo version).
//
// Usage: go run . <params.json> <sk.bin> <ct.bin>
//
// Reads CKKS params JSON, secret key bytes, and ciphertext bytes.
// Unmarshal SK and CT using native Lattigo, decrypt, decode, print values.
// Exits 0 on success, 1 on any error.
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

type paramsJSON struct {
	LogN     int    `json:"LogN"`
	LogQ     []int  `json:"LogQ"`
	LogP     []int  `json:"LogP"`
	LogScale int    `json:"LogDefaultScale"`
	H        int    `json:"H"`
	RingType string `json:"RingType"`
}

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintf(os.Stderr, "Usage: %s <params.json> <sk.bin> <ct.bin>\n", os.Args[0])
		os.Exit(1)
	}

	// Read params JSON
	pBytes, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "read params: %v\n", err)
		os.Exit(1)
	}
	var pj paramsJSON
	if err := json.Unmarshal(pBytes, &pj); err != nil {
		fmt.Fprintf(os.Stderr, "parse params: %v\n", err)
		os.Exit(1)
	}

	// Build Lattigo params (same logic as WASM bridge params.go)
	rt := ring.ConjugateInvariant
	if pj.RingType == "Standard" || pj.RingType == "standard" {
		rt = ring.Standard
	}
	lit := ckks.ParametersLiteral{
		LogN:            pj.LogN,
		LogQ:            pj.LogQ,
		LogP:            pj.LogP,
		LogDefaultScale: pj.LogScale,
		RingType:        rt,
	}
	if pj.H > 0 {
		lit.Xs = ring.Ternary{H: pj.H}
	}
	params, err := ckks.NewParametersFromLiteral(lit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "create params: %v\n", err)
		os.Exit(1)
	}

	// Unmarshal secret key
	skBytes, err := os.ReadFile(os.Args[2])
	if err != nil {
		fmt.Fprintf(os.Stderr, "read sk: %v\n", err)
		os.Exit(1)
	}
	sk := rlwe.NewSecretKey(params.Parameters)
	if err := sk.UnmarshalBinary(skBytes); err != nil {
		fmt.Fprintf(os.Stderr, "unmarshal sk: %v\n", err)
		os.Exit(1)
	}

	// Unmarshal ciphertext
	ctBytes, err := os.ReadFile(os.Args[3])
	if err != nil {
		fmt.Fprintf(os.Stderr, "read ct: %v\n", err)
		os.Exit(1)
	}
	ct := rlwe.NewCiphertext(params.Parameters, 1, params.MaxLevel())
	if err := ct.UnmarshalBinary(ctBytes); err != nil {
		fmt.Fprintf(os.Stderr, "unmarshal ct: %v\n", err)
		os.Exit(1)
	}

	// Decrypt
	dec := ckks.NewDecryptor(params, sk)
	pt := dec.DecryptNew(ct)

	// Decode
	enc := ckks.NewEncoder(params)
	values := make([]float64, params.MaxSlots())
	if err := enc.Decode(pt, values); err != nil {
		fmt.Fprintf(os.Stderr, "decode: %v\n", err)
		os.Exit(1)
	}

	// Print first 8 values as JSON array
	n := 8
	if len(values) < n {
		n = len(values)
	}
	output := values[:n]
	jsonOut, _ := json.Marshal(output)
	fmt.Println(string(jsonOut))
}
