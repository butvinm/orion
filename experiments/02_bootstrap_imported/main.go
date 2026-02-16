// Experiment 02: Go Bootstrap with Imported Keys
//
// Hypothesis: A bootstrapping.Evaluator can be constructed from deserialized
// bootstrap evaluation keys, without the secret key present on the server side.
//
// What this proves: That the most complex key type (bootstrap keys — generated
// via btpParams.GenEvaluationKeys(sk)) can be serialized, transferred, and used
// independently. This is critical because bootstrap is the operation that
// refreshes ciphertext noise budget.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"os"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
	"github.com/baahl-nyu/lattigo/v6/utils"
	"github.com/baahl-nyu/lattigo/v6/utils/sampling"
)

// Small CKKS parameters for fast testing (insecure, but sufficient to prove the hypothesis).
// LogN=10 keeps bootstrap keygen + eval under a few seconds.
var residualParamsLit = ckks.ParametersLiteral{
	LogN:            10,
	LogQ:            []int{60, 40},
	LogP:            []int{61},
	LogDefaultScale: 40,
}

// btpParamsLit: empty literal means all defaults. LogN is set programmatically.
var btpParamsLit = bootstrapping.ParametersLiteral{}

// buildParams creates the residual and bootstrapping parameter objects.
// Returns both, along with a flag indicating whether corrections for small LogN are applied.
func buildParams() (ckks.Parameters, bootstrapping.Parameters) {
	params, err := ckks.NewParametersFromLiteral(residualParamsLit)
	if err != nil {
		panic(fmt.Sprintf("failed to create residual params: %v", err))
	}

	btpParamsLit.LogN = utils.Pointy(params.LogN())

	btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParamsLit)
	if err != nil {
		panic(fmt.Sprintf("failed to create bootstrap params: %v", err))
	}

	// Corrections for small LogN (from Lattigo test suite):
	// Adjust LogSlots and message ratio for adequate precision at LogN=10.
	btpParams.SlotsToCoeffsParameters.LogSlots = btpParams.BootstrappingParameters.LogN() - 1
	btpParams.CoeffsToSlotsParameters.LogSlots = btpParams.BootstrappingParameters.LogN() - 1
	btpParams.Mod1ParametersLiteral.LogMessageRatio += 16 - params.LogN()

	fmt.Printf("Residual params: LogN=%d, MaxLevel=%d, MaxSlots=%d, LogQP=%.1f\n",
		params.LogN(), params.MaxLevel(), params.MaxSlots(), params.LogQP())
	fmt.Printf("Bootstrap params: LogN=%d, LogSlots=%d\n",
		btpParams.BootstrappingParameters.LogN(),
		btpParams.SlotsToCoeffsParameters.LogSlots)

	return params, btpParams
}

// clientGenKeysAndEncrypt simulates the client:
//   - Generates sk from the bootstrapping parameters
//   - Generates bootstrap evaluation keys
//   - Serializes the bootstrap eval keys
//   - Encrypts a test vector at level 0 (ready for bootstrap)
//
// Returns serialized bootstrap keys, serialized ciphertext, expected values, sk.
func clientGenKeysAndEncrypt(
	params ckks.Parameters,
	btpParams bootstrapping.Parameters,
) (
	btpKeysBytes []byte,
	ctBytes []byte,
	expectedValues []complex128,
	sk *rlwe.SecretKey,
) {
	fmt.Println("\n--- Client: key generation ---")

	// Generate secret key from the bootstrapping parameters (since no ring degree switch, this uses residual LogN)
	kgen := rlwe.NewKeyGenerator(btpParams.BootstrappingParameters)
	sk = kgen.GenSecretKeyNew()
	fmt.Println("[CLIENT] Generated secret key")

	// Generate bootstrap evaluation keys
	btpKeys, _, err := btpParams.GenEvaluationKeys(sk)
	if err != nil {
		panic(fmt.Sprintf("failed to generate bootstrap eval keys: %v", err))
	}
	fmt.Println("[CLIENT] Generated bootstrap evaluation keys")

	// Serialize bootstrap eval keys
	btpKeysBytes, err = btpKeys.MarshalBinary()
	if err != nil {
		panic(fmt.Sprintf("failed to serialize bootstrap keys: %v", err))
	}
	fmt.Printf("[CLIENT] Serialized bootstrap keys: %d bytes (%.2f MB)\n",
		len(btpKeysBytes), float64(len(btpKeysBytes))/(1024*1024))

	// Encode and encrypt test data at level 0 (the level bootstrap expects as input)
	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, sk)

	slots := params.MaxSlots()
	expectedValues = make([]complex128, slots)
	for i := range expectedValues {
		expectedValues[i] = sampling.RandComplex128(-1, 1)
	}

	plaintext := ckks.NewPlaintext(params, 0) // level 0: bootstrap input
	if err := encoder.Encode(expectedValues, plaintext); err != nil {
		panic(fmt.Sprintf("failed to encode: %v", err))
	}

	ct, err := encryptor.EncryptNew(plaintext)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt: %v", err))
	}
	fmt.Printf("[CLIENT] Encrypted test vector at level %d (%d slots)\n", ct.Level(), slots)

	ctBytes, err = ct.MarshalBinary()
	if err != nil {
		panic(fmt.Sprintf("failed to serialize ciphertext: %v", err))
	}
	fmt.Printf("[CLIENT] Serialized ciphertext: %d bytes\n", len(ctBytes))

	return
}

// serverBootstrap simulates the server:
//   - Deserializes bootstrap evaluation keys (NO sk, NO keygen)
//   - Constructs bootstrapping.Evaluator from deserialized keys
//   - Bootstraps the ciphertext (level 0 -> max level)
//   - Returns serialized result
func serverBootstrap(
	params ckks.Parameters,
	btpParams bootstrapping.Parameters,
	btpKeysBytes []byte,
	ctBytes []byte,
) []byte {
	fmt.Println("\n--- Server: bootstrap (NO sk, NO keygen) ---")

	// Deserialize bootstrap evaluation keys
	btpKeys := &bootstrapping.EvaluationKeys{}
	if err := btpKeys.UnmarshalBinary(btpKeysBytes); err != nil {
		panic(fmt.Sprintf("failed to deserialize bootstrap keys: %v", err))
	}
	fmt.Println("[SERVER] Deserialized bootstrap evaluation keys")

	// Construct bootstrapping evaluator from deserialized keys
	btpEval, err := bootstrapping.NewEvaluator(btpParams, btpKeys)
	if err != nil {
		panic(fmt.Sprintf("failed to create bootstrap evaluator: %v", err))
	}
	fmt.Println("[SERVER] Created bootstrapping.Evaluator from deserialized keys (NO sk, NO keygen)")

	// Deserialize ciphertext
	ct := rlwe.NewCiphertext(params, 1, 0) // degree=1, level=0
	if err := ct.UnmarshalBinary(ctBytes); err != nil {
		panic(fmt.Sprintf("failed to deserialize ciphertext: %v", err))
	}
	fmt.Printf("[SERVER] Deserialized ciphertext at level %d\n", ct.Level())

	// Perform bootstrap: level 0 -> max level
	ctBoot, err := btpEval.Bootstrap(ct)
	if err != nil {
		panic(fmt.Sprintf("bootstrap failed: %v", err))
	}
	fmt.Printf("[SERVER] Bootstrapped ciphertext: level %d -> %d\n", ct.Level(), ctBoot.Level())

	// Verify output level is max level
	if ctBoot.Level() != params.MaxLevel() {
		fmt.Printf("[WARN] Expected output level %d, got %d\n", params.MaxLevel(), ctBoot.Level())
	}

	// Serialize result
	resultBytes, err := ctBoot.MarshalBinary()
	if err != nil {
		panic(fmt.Sprintf("failed to serialize bootstrapped ciphertext: %v", err))
	}
	fmt.Printf("[SERVER] Serialized bootstrapped ciphertext: %d bytes\n", len(resultBytes))

	return resultBytes
}

// clientDecryptAndVerify decrypts the bootstrapped result and checks accuracy.
func clientDecryptAndVerify(
	params ckks.Parameters,
	sk *rlwe.SecretKey,
	resultBytes []byte,
	expectedValues []complex128,
) {
	fmt.Println("\n--- Client: decrypt and verify ---")

	// Deserialize result
	ctResult := rlwe.NewCiphertext(params, 1, params.MaxLevel())
	if err := ctResult.UnmarshalBinary(resultBytes); err != nil {
		panic(fmt.Sprintf("failed to deserialize result: %v", err))
	}
	fmt.Printf("[CLIENT] Deserialized result at level %d\n", ctResult.Level())

	// Decrypt
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	ptResult := decryptor.DecryptNew(ctResult)

	slots := params.MaxSlots()
	result := make([]complex128, slots)
	if err := encoder.Decode(ptResult, result); err != nil {
		panic(fmt.Sprintf("failed to decode: %v", err))
	}

	// Compute errors
	maxErr := 0.0
	meanErr := 0.0
	for i := 0; i < slots; i++ {
		err := cmplx.Abs(result[i] - expectedValues[i])
		meanErr += err
		if err > maxErr {
			maxErr = err
		}
	}
	meanErr /= float64(slots)

	// Print first few values
	for i := 0; i < min(8, slots); i++ {
		err := cmplx.Abs(result[i] - expectedValues[i])
		fmt.Printf("  slot[%d]: got=(%.6f, %.6f), want=(%.6f, %.6f), err=%.2e\n",
			i,
			real(result[i]), imag(result[i]),
			real(expectedValues[i]), imag(expectedValues[i]),
			err)
	}

	fmt.Printf("\nMax error across %d slots: %.2e\n", slots, maxErr)
	fmt.Printf("Mean error: %.2e\n", meanErr)
	fmt.Printf("Precision: %.1f bits\n", math.Log2(1/maxErr))

	// Bootstrap at LogN=10 with adjusted message ratio should give reasonable precision.
	// The threshold is generous — we're proving the key import works, not optimizing precision.
	threshold := 1.0 // very generous for small params
	if maxErr < threshold {
		fmt.Printf("[PASS] Max error %.2e < threshold %.2e\n", maxErr, threshold)
	} else {
		fmt.Printf("[FAIL] Max error %.2e >= threshold %.2e\n", maxErr, threshold)
		os.Exit(1)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	fmt.Println("=== Experiment 02: Go Bootstrap with Imported Keys ===")

	// Build shared parameters
	params, btpParams := buildParams()

	// Phase 1: Client generates keys, serializes bootstrap eval keys, encrypts data
	btpKeysBytes, ctBytes, expectedValues, sk := clientGenKeysAndEncrypt(params, btpParams)

	// Phase 2: Server deserializes bootstrap keys, bootstraps ciphertext (NO sk)
	resultBytes := serverBootstrap(params, btpParams, btpKeysBytes, ctBytes)

	// Phase 3: Client decrypts and verifies
	clientDecryptAndVerify(params, sk, resultBytes, expectedValues)

	fmt.Println("\n=== Experiment 02 PASSED ===")

	// Document findings
	fmt.Println("\n=== API Documentation ===")
	fmt.Println("Bootstrap key serialization (client):")
	fmt.Println("  btpKeys, _, err := btpParams.GenEvaluationKeys(sk)")
	fmt.Println("  data, err := btpKeys.MarshalBinary()")
	fmt.Println()
	fmt.Println("Bootstrap key deserialization (server):")
	fmt.Println("  btpKeys := &bootstrapping.EvaluationKeys{}")
	fmt.Println("  err := btpKeys.UnmarshalBinary(data)")
	fmt.Println()
	fmt.Println("Bootstrap evaluator construction (server, no sk):")
	fmt.Println("  eval, err := bootstrapping.NewEvaluator(btpParams, btpKeys)")
	fmt.Println()
	fmt.Println("bootstrapping.EvaluationKeys struct contains:")
	fmt.Println("  - EvkN1ToN2, EvkN2ToN1: ring degree switch keys (nil if N1==N2)")
	fmt.Println("  - EvkRealToCmplx, EvkCmplxToReal: ring type switch keys (nil for Standard)")
	fmt.Println("  - EvkDenseToSparse, EvkSparseToDense: encapsulation keys (nil if EphemeralSecretWeight==0)")
	fmt.Println("  - *rlwe.MemEvaluationKeySet: embedded rlk + all bootstrap galois keys")
	fmt.Println()
	fmt.Println("Note: EvaluationKeys implements MarshalBinary/UnmarshalBinary directly.")
	fmt.Println("      Component keys do NOT need to be serialized individually.")
	fmt.Println("      bootstrapping.NewEvaluator accepts deserialized keys without needing sk.")
}
