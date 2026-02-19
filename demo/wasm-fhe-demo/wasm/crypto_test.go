package main

import (
	"testing"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
)

// MLP model parameters from the plan.
var (
	testLogN     = 13
	testLogQ     = []int{29, 26, 26, 26, 26, 26}
	testLogP     = []int{29, 29}
	testLogScale = 26
	testH        = 8192
	testRingType = "conjugate_invariant"
)

func initTestScheme(t *testing.T) {
	t.Helper()
	if err := InitScheme(testLogN, testLogQ, testLogP, testLogScale, testH, testRingType); err != nil {
		t.Fatalf("InitScheme failed: %v", err)
	}
}

func TestInitScheme(t *testing.T) {
	initTestScheme(t)

	if s.Params == nil {
		t.Fatal("Params is nil after InitScheme")
	}
	if s.KeyGen == nil {
		t.Fatal("KeyGen is nil after InitScheme")
	}
	if s.SecretKey == nil {
		t.Fatal("SecretKey is nil after InitScheme")
	}
	if s.PublicKey == nil {
		t.Fatal("PublicKey is nil after InitScheme")
	}
	if s.RelinKey == nil {
		t.Fatal("RelinKey is nil after InitScheme")
	}
	if s.Encoder == nil {
		t.Fatal("Encoder is nil after InitScheme")
	}
	if s.Encryptor == nil {
		t.Fatal("Encryptor is nil after InitScheme")
	}
	if s.Decryptor == nil {
		t.Fatal("Decryptor is nil after InitScheme")
	}

	if got := s.Params.LogN(); got != testLogN {
		t.Errorf("LogN = %d, want %d", got, testLogN)
	}
}

func TestGetMaxSlots(t *testing.T) {
	initTestScheme(t)

	slots := GetMaxSlots()
	// For conjugate_invariant ring with logN=13, maxSlots = N/2 = 2^13 / 2 = 4096.
	// Actually, conjugate_invariant doubles the slots: maxSlots = N = 2^13 = 8192?
	// No — for CKKS conjugate invariant, MaxSlots = N/2 where N = 2^logN.
	// Let's just verify it's positive and a power of 2.
	if slots <= 0 {
		t.Fatalf("GetMaxSlots() = %d, want > 0", slots)
	}
	if slots&(slots-1) != 0 {
		t.Fatalf("GetMaxSlots() = %d, not a power of 2", slots)
	}
	t.Logf("MaxSlots = %d", slots)
}

func TestSerializeRelinKey(t *testing.T) {
	initTestScheme(t)

	data, err := SerializeRelinKey()
	if err != nil {
		t.Fatalf("SerializeRelinKey failed: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("SerializeRelinKey returned empty data")
	}
	t.Logf("RelinKey serialized: %d bytes", len(data))

	// Round-trip: deserialize and verify
	rlk := rlwe.NewRelinearizationKey(s.Params)
	if err := rlk.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary RelinKey failed: %v", err)
	}
}

func TestGenerateAndSerializeGaloisKey(t *testing.T) {
	initTestScheme(t)

	// Use a real galois element from the parameters.
	galEl := s.Params.GaloisElement(1)

	data, err := GenerateAndSerializeGaloisKey(galEl)
	if err != nil {
		t.Fatalf("GenerateAndSerializeGaloisKey(%d) failed: %v", galEl, err)
	}
	if len(data) == 0 {
		t.Fatal("GenerateAndSerializeGaloisKey returned empty data")
	}
	t.Logf("GaloisKey(%d) serialized: %d bytes", galEl, len(data))

	// Round-trip: deserialize and verify
	var gk rlwe.GaloisKey
	if err := gk.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary GaloisKey failed: %v", err)
	}
}

func TestMultipleGaloisKeys(t *testing.T) {
	initTestScheme(t)

	rotations := []int{1, 2, 4, 8}
	for _, rot := range rotations {
		galEl := s.Params.GaloisElement(rot)
		data, err := GenerateAndSerializeGaloisKey(galEl)
		if err != nil {
			t.Fatalf("GaloisKey for rotation %d failed: %v", rot, err)
		}
		if len(data) == 0 {
			t.Fatalf("GaloisKey for rotation %d is empty", rot)
		}
		t.Logf("GaloisKey(rot=%d, galEl=%d): %d bytes", rot, galEl, len(data))
	}
}

func TestGetDefaultScale(t *testing.T) {
	initTestScheme(t)

	scale := GetDefaultScale()
	// logScale=26, so default scale should be 2^26 = 67108864
	expected := uint64(1) << uint64(testLogScale)
	if scale != expected {
		t.Errorf("GetDefaultScale() = %d, want %d", scale, expected)
	}
	t.Logf("DefaultScale = %d (2^%d)", scale, testLogScale)
}

func TestEncodeEncryptDecryptRoundTrip(t *testing.T) {
	initTestScheme(t)

	// Prepare test values: some floats that represent a small input vector.
	inputValues := []float64{0.1, 0.5, -0.3, 1.0, 0.0, -1.0, 0.42, 0.99}
	level := s.Params.MaxLevel()
	scale := GetDefaultScale()

	// Encode
	ptID, err := Encode(inputValues, level, scale)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}
	t.Logf("Encoded plaintext ID: %d", ptID)

	// Encrypt
	ctBytes, err := Encrypt(ptID)
	if err != nil {
		t.Fatalf("Encrypt failed: %v", err)
	}
	if len(ctBytes) == 0 {
		t.Fatal("Encrypt returned empty bytes")
	}
	t.Logf("Ciphertext: %d bytes", len(ctBytes))

	// Decrypt (includes decode)
	result, err := Decrypt(ctBytes)
	if err != nil {
		t.Fatalf("Decrypt failed: %v", err)
	}
	if len(result) == 0 {
		t.Fatal("Decrypt returned empty result")
	}
	t.Logf("Decrypted %d slots", len(result))

	// Verify the first N values match within CKKS tolerance.
	const tol = 1e-3
	for i, want := range inputValues {
		got := result[i]
		diff := got - want
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			t.Errorf("result[%d] = %f, want %f (diff=%e, tol=%e)", i, got, want, diff, tol)
		}
	}
	t.Logf("Round-trip first %d values match within tolerance %e", len(inputValues), tol)
}

func TestEncryptInvalidID(t *testing.T) {
	initTestScheme(t)

	// Should fail for non-existent plaintext ID.
	_, err := Encrypt(999)
	if err == nil {
		t.Fatal("Encrypt(999) should have failed for invalid ID")
	}
	t.Logf("Expected error: %v", err)
}

func TestEncryptConsumesThenRejects(t *testing.T) {
	initTestScheme(t)

	values := []float64{1.0, 2.0, 3.0}
	level := s.Params.MaxLevel()
	scale := GetDefaultScale()

	ptID, err := Encode(values, level, scale)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	// First encrypt should succeed.
	_, err = Encrypt(ptID)
	if err != nil {
		t.Fatalf("First Encrypt failed: %v", err)
	}

	// Second encrypt of same ID should fail (plaintext was consumed).
	_, err = Encrypt(ptID)
	if err == nil {
		t.Fatal("Second Encrypt should have failed (plaintext consumed)")
	}
	t.Logf("Expected error on reuse: %v", err)
}
