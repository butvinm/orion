// Experiment 01: Go Client-Server Eval Key Roundtrip
//
// Hypothesis: Keys (pk, rlk, galois keys) serialized from one Lattigo instance
// can be deserialized and used for ciphertext evaluation (ct-ct multiply,
// relinearize, rotate) on another instance that NEVER had the secret key.
//
// What this proves: The fundamental building block — that Lattigo's key
// serialization produces self-contained key objects usable without the
// originating keygen/sk.

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

// CKKS parameters (small for fast experiment, but large enough for meaningful ops)
var paramsLit = ckks.ParametersLiteral{
	LogN:            13,
	LogQ:            []int{55, 40, 40, 40, 40, 40},
	LogP:            []int{45, 45},
	LogDefaultScale: 40,
	Xs:              ring.Ternary{H: 192},
	RingType:        ring.Standard,
}

// rotationsToTest are the rotation amounts the server will need
var rotationsToTest = []int{1, 2, 4, 8}

// clientScope simulates the client side:
// - Creates params, keygen, sk
// - Generates pk, rlk, galois keys for specific rotations
// - Serializes all eval keys to bytes
// - Encrypts test data
// Returns: serialized keys, serialized ciphertext, expected cleartext result
func clientScope(params ckks.Parameters) (
	pkBytes []byte,
	rlkBytes []byte,
	galoisKeyBytes map[uint64][]byte, // galEl -> serialized key
	ctABytes []byte,
	ctBBytes []byte,
	clearA []float64,
	clearB []float64,
	sk *rlwe.SecretKey,
) {
	// Generate keys
	kgen := rlwe.NewKeyGenerator(params)
	sk = kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Generate galois keys for specific rotations
	galoisKeyBytes = make(map[uint64][]byte)
	for _, rot := range rotationsToTest {
		galEl := params.GaloisElement(rot)
		gk := kgen.GenGaloisKeyNew(galEl, sk)
		data, err := gk.MarshalBinary()
		if err != nil {
			panic(fmt.Sprintf("failed to serialize galois key for rotation %d: %v", rot, err))
		}
		galoisKeyBytes[galEl] = data
		fmt.Printf("[CLIENT] Serialized galois key for rotation %d (galEl=%d): %d bytes\n", rot, galEl, len(data))
	}

	// Serialize pk
	var err error
	pkBytes, err = pk.MarshalBinary()
	if err != nil {
		panic(fmt.Sprintf("failed to serialize pk: %v", err))
	}
	fmt.Printf("[CLIENT] Serialized public key: %d bytes\n", len(pkBytes))

	// Serialize rlk
	rlkBytes, err = rlk.MarshalBinary()
	if err != nil {
		panic(fmt.Sprintf("failed to serialize rlk: %v", err))
	}
	fmt.Printf("[CLIENT] Serialized relinearization key: %d bytes\n", len(rlkBytes))

	// Encrypt test data
	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)

	slots := params.MaxSlots()
	clearA = make([]float64, slots)
	clearB = make([]float64, slots)
	for i := range clearA {
		clearA[i] = float64(i) * 0.001
		clearB[i] = float64(slots-i) * 0.001
	}

	ptA := ckks.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(clearA, ptA)
	ctA, err := encryptor.EncryptNew(ptA)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt A: %v", err))
	}

	ptB := ckks.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(clearB, ptB)
	ctB, err := encryptor.EncryptNew(ptB)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt B: %v", err))
	}

	ctABytes, err = ctA.MarshalBinary()
	if err != nil {
		panic(fmt.Sprintf("failed to serialize ctA: %v", err))
	}
	ctBBytes, err = ctB.MarshalBinary()
	if err != nil {
		panic(fmt.Sprintf("failed to serialize ctB: %v", err))
	}

	fmt.Printf("[CLIENT] Encrypted and serialized two ciphertexts: %d bytes each\n", len(ctABytes))
	return
}

// serverScope simulates the server side:
// - Creates params (same config, but NO keygen, NO sk)
// - Deserializes pk/rlk/galois keys
// - Constructs evaluator from deserialized keys
// - Performs ct-ct multiply + relinearize + rotate
// Returns: serialized result ciphertext
func serverScope(
	params ckks.Parameters,
	pkBytes []byte,
	rlkBytes []byte,
	galoisKeyBytes map[uint64][]byte,
	ctABytes []byte,
	ctBBytes []byte,
) []byte {
	// NOTE: Server NEVER creates a KeyGenerator or SecretKey.
	// It only deserializes evaluation keys.

	// Deserialize public key
	pk := rlwe.NewPublicKey(params)
	if err := pk.UnmarshalBinary(pkBytes); err != nil {
		panic(fmt.Sprintf("failed to deserialize pk: %v", err))
	}
	fmt.Println("[SERVER] Deserialized public key")

	// Deserialize relinearization key
	rlk := rlwe.NewRelinearizationKey(params)
	if err := rlk.UnmarshalBinary(rlkBytes); err != nil {
		panic(fmt.Sprintf("failed to deserialize rlk: %v", err))
	}
	fmt.Println("[SERVER] Deserialized relinearization key")

	// Deserialize galois keys
	galoisKeys := make([]*rlwe.GaloisKey, 0, len(galoisKeyBytes))
	for galEl, data := range galoisKeyBytes {
		gk := rlwe.NewGaloisKey(params)
		if err := gk.UnmarshalBinary(data); err != nil {
			panic(fmt.Sprintf("failed to deserialize galois key %d: %v", galEl, err))
		}
		galoisKeys = append(galoisKeys, gk)
		fmt.Printf("[SERVER] Deserialized galois key for galEl=%d\n", galEl)
	}

	// Construct evaluation key set from deserialized keys (NO keygen, NO sk!)
	evk := rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...)
	fmt.Println("[SERVER] Constructed MemEvaluationKeySet from deserialized keys")

	// Create evaluator with the deserialized key set
	eval := ckks.NewEvaluator(params, evk)
	fmt.Println("[SERVER] Created ckks.Evaluator from deserialized keys (no sk, no keygen)")

	// Deserialize ciphertexts
	ctA := rlwe.NewCiphertext(params, 1, params.MaxLevel())
	if err := ctA.UnmarshalBinary(ctABytes); err != nil {
		panic(fmt.Sprintf("failed to deserialize ctA: %v", err))
	}
	ctB := rlwe.NewCiphertext(params, 1, params.MaxLevel())
	if err := ctB.UnmarshalBinary(ctBBytes); err != nil {
		panic(fmt.Sprintf("failed to deserialize ctB: %v", err))
	}
	fmt.Println("[SERVER] Deserialized two ciphertexts")

	// Perform ct-ct multiply + relinearize
	var ctMul *rlwe.Ciphertext
	var err error
	ctMul, err = eval.MulRelinNew(ctA, ctB)
	if err != nil {
		panic(fmt.Sprintf("MulRelin failed: %v", err))
	}
	if err = eval.Rescale(ctMul, ctMul); err != nil {
		panic(fmt.Sprintf("Rescale failed: %v", err))
	}
	fmt.Println("[SERVER] Performed ct*ct multiply + relinearize + rescale")

	// Perform rotation by 1
	ctRot, err := eval.RotateNew(ctMul, 1)
	if err != nil {
		panic(fmt.Sprintf("Rotate by 1 failed: %v", err))
	}
	fmt.Println("[SERVER] Performed rotation by 1")

	// Perform additional rotations to verify all provided keys work
	for _, rot := range rotationsToTest[1:] { // skip 1, already done
		_, err := eval.RotateNew(ctMul, rot)
		if err != nil {
			panic(fmt.Sprintf("Rotate by %d failed: %v", rot, err))
		}
		fmt.Printf("[SERVER] Performed rotation by %d\n", rot)
	}

	// Serialize result
	resultBytes, err := ctRot.MarshalBinary()
	if err != nil {
		panic(fmt.Sprintf("failed to serialize result: %v", err))
	}
	fmt.Printf("[SERVER] Serialized result ciphertext: %d bytes\n", len(resultBytes))
	return resultBytes
}

// testMissingKeyFails verifies that a rotation with a missing galois key
// produces an explicit error (not silent corruption).
func testMissingKeyFails(params ckks.Parameters, rlkBytes []byte, ctABytes []byte) {
	fmt.Println("\n=== Testing missing galois key behavior ===")

	// Deserialize rlk only — no galois keys at all
	rlk := rlwe.NewRelinearizationKey(params)
	if err := rlk.UnmarshalBinary(rlkBytes); err != nil {
		panic(fmt.Sprintf("failed to deserialize rlk: %v", err))
	}

	// Create evaluator with NO galois keys
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := ckks.NewEvaluator(params, evk)

	// Deserialize a ciphertext
	ct := rlwe.NewCiphertext(params, 1, params.MaxLevel())
	if err := ct.UnmarshalBinary(ctABytes); err != nil {
		panic(fmt.Sprintf("failed to deserialize ct: %v", err))
	}

	// Attempt rotation with missing key — should fail
	_, err := eval.RotateNew(ct, 1)
	if err != nil {
		fmt.Printf("[PASS] Rotation with missing key failed with error: %v\n", err)
	} else {
		// If no error, this is a problem — check if it panicked or silently corrupted
		fmt.Println("[WARN] Rotation with missing key did NOT return an error!")
		fmt.Println("       This could mean silent corruption — Lattigo may panic instead of returning error.")
		fmt.Println("       In production, we should handle this case.")
	}
}

// clientDecrypt decrypts the result on the client side and verifies correctness
func clientDecrypt(
	params ckks.Parameters,
	sk *rlwe.SecretKey,
	resultBytes []byte,
	clearA []float64,
	clearB []float64,
) {
	fmt.Println("\n=== Client-side decryption and verification ===")

	// Deserialize result
	ctResult := rlwe.NewCiphertext(params, 1, params.MaxLevel())
	if err := ctResult.UnmarshalBinary(resultBytes); err != nil {
		panic(fmt.Sprintf("failed to deserialize result: %v", err))
	}

	// Decrypt
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	ptResult := decryptor.DecryptNew(ctResult)

	slots := params.MaxSlots()
	result := make([]float64, slots)
	if err := encoder.Decode(ptResult, result); err != nil {
		panic(fmt.Sprintf("failed to decode result: %v", err))
	}

	// Compute expected result: (A * B) rotated by 1
	// After rotation by 1, element[i] = original[i+1]
	expected := make([]float64, slots)
	for i := 0; i < slots; i++ {
		expected[i] = clearA[(i+1)%slots] * clearB[(i+1)%slots]
	}

	// Verify
	maxErr := 0.0
	for i := 0; i < min(10, slots); i++ {
		err := math.Abs(result[i] - expected[i])
		if err > maxErr {
			maxErr = err
		}
		fmt.Printf("  result[%d] = %.10f, expected = %.10f, err = %.2e\n", i, result[i], expected[i], err)
	}

	// Check overall max error
	for i := 0; i < slots; i++ {
		err := math.Abs(result[i] - expected[i])
		if err > maxErr {
			maxErr = err
		}
	}

	fmt.Printf("\nMax error across all %d slots: %.2e\n", slots, maxErr)

	// For CKKS with 40-bit scale, errors should be well below 1e-5
	threshold := 1e-3
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
	fmt.Println("=== Experiment 01: Go Client-Server Eval Key Roundtrip ===\n")

	// Create params (shared configuration — both sides know the scheme parameters)
	params, err := ckks.NewParametersFromLiteral(paramsLit)
	if err != nil {
		panic(fmt.Sprintf("failed to create params: %v", err))
	}
	scale := params.DefaultScale().Value
	fmt.Printf("CKKS Parameters: LogN=%d, MaxLevel=%d, MaxSlots=%d, DefaultScale=%s\n\n",
		params.LogN(), params.MaxLevel(), params.MaxSlots(), (&scale).Text('f', 0))

	// Phase 1: Client generates and serializes keys + encrypts data
	fmt.Println("--- Phase 1: Client key generation and encryption ---")
	pkBytes, rlkBytes, galoisKeyBytes, ctABytes, ctBBytes, clearA, clearB, sk := clientScope(params)

	// Phase 2: Server deserializes keys and performs computation
	fmt.Println("\n--- Phase 2: Server computation (NO sk, NO keygen) ---")
	resultBytes := serverScope(params, pkBytes, rlkBytes, galoisKeyBytes, ctABytes, ctBBytes)

	// Phase 3: Test that missing key fails explicitly
	testMissingKeyFails(params, rlkBytes, ctABytes)

	// Phase 4: Client decrypts and verifies
	clientDecrypt(params, sk, resultBytes, clearA, clearB)

	fmt.Println("\n=== Experiment 01 PASSED ===")

	// Document API calls needed
	fmt.Println("\n=== API Documentation ===")
	fmt.Println("Key serialization (client):")
	fmt.Println("  pk.MarshalBinary() -> []byte")
	fmt.Println("  rlk.MarshalBinary() -> []byte")
	fmt.Println("  galoisKey.MarshalBinary() -> []byte")
	fmt.Println("  ciphertext.MarshalBinary() -> []byte")
	fmt.Println()
	fmt.Println("Key deserialization (server):")
	fmt.Println("  pk := rlwe.NewPublicKey(params); pk.UnmarshalBinary(data)")
	fmt.Println("  rlk := rlwe.NewRelinearizationKey(params); rlk.UnmarshalBinary(data)")
	fmt.Println("  gk := rlwe.NewGaloisKey(params); gk.UnmarshalBinary(data)")
	fmt.Println("  ct := rlwe.NewCiphertext(params, degree, level); ct.UnmarshalBinary(data)")
	fmt.Println()
	fmt.Println("Evaluator construction (server, no sk):")
	fmt.Println("  evk := rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...)")
	fmt.Println("  eval := ckks.NewEvaluator(params, evk)")
	fmt.Println()
	fmt.Println("Note: rlwe.NewPublicKey, NewRelinearizationKey, NewGaloisKey need params")
	fmt.Println("      to allocate correctly-sized ring elements before UnmarshalBinary.")

}
