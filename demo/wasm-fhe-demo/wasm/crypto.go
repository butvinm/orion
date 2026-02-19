package main

import (
	"fmt"
	"math"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
	"github.com/baahl-nyu/lattigo/v6/utils"
)

// scheme holds all client-side cryptographic state.
// Only one set of parameters can be active at a time (singleton).
type scheme struct {
	Params    *ckks.Parameters
	KeyGen    *rlwe.KeyGenerator
	SecretKey *rlwe.SecretKey
	PublicKey *rlwe.PublicKey
	RelinKey  *rlwe.RelinearizationKey
	Encoder   *ckks.Encoder
	Encryptor *rlwe.Encryptor
	Decryptor *rlwe.Decryptor
}

var s scheme

// InitScheme initializes CKKS parameters, generates all keys, and sets up
// encoder/encryptor/decryptor. This is the only initialization function
// the client needs to call.
func InitScheme(logN int, logQ, logP []int, logScale, h int, ringType string) error {
	ringT := ring.Standard
	if ringType != "standard" {
		ringT = ring.ConjugateInvariant
	}

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            logQ,
		LogP:            logP,
		LogDefaultScale: logScale,
		Xs:              ring.Ternary{H: h},
		RingType:        ringT,
	})
	if err != nil {
		return fmt.Errorf("NewParametersFromLiteral: %w", err)
	}

	keyGen := rlwe.NewKeyGenerator(params)
	sk := keyGen.GenSecretKeyNew()
	pk := keyGen.GenPublicKeyNew(sk)
	rlk := keyGen.GenRelinearizationKeyNew(sk)

	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, pk)
	decryptor := ckks.NewDecryptor(params, sk)

	s = scheme{
		Params:    &params,
		KeyGen:    keyGen,
		SecretKey: sk,
		PublicKey: pk,
		RelinKey:  rlk,
		Encoder:   encoder,
		Encryptor: encryptor,
		Decryptor: decryptor,
	}

	return nil
}

// GetMaxSlots returns the maximum number of plaintext slots for the current parameters.
func GetMaxSlots() int {
	return s.Params.MaxSlots()
}

// SerializeRelinKey serializes the relinearization key to binary.
func SerializeRelinKey() ([]byte, error) {
	if s.RelinKey == nil {
		return nil, fmt.Errorf("relinearization key not generated")
	}
	data, err := s.RelinKey.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("MarshalBinary RelinKey: %w", err)
	}
	return data, nil
}

// GenerateAndSerializeGaloisKey generates a Galois key for the given element
// and returns its binary serialization.
func GenerateAndSerializeGaloisKey(galEl uint64) ([]byte, error) {
	if s.KeyGen == nil || s.SecretKey == nil {
		return nil, fmt.Errorf("scheme not initialized")
	}
	rotKey := s.KeyGen.GenGaloisKeyNew(galEl, s.SecretKey)
	data, err := rotKey.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("MarshalBinary GaloisKey(%d): %w", galEl, err)
	}
	return data, nil
}

// SerializeBootstrapKeys generates bootstrap evaluation keys for the given
// slot count and auxiliary modulus chain, then serializes them.
func SerializeBootstrapKeys(numSlots int, logP []int) ([]byte, error) {
	if s.Params == nil || s.SecretKey == nil {
		return nil, fmt.Errorf("scheme not initialized")
	}

	btpLit := bootstrapping.ParametersLiteral{
		LogN:     utils.Pointy(s.Params.LogN()),
		LogP:     logP,
		Xs:       s.Params.Xs(),
		LogSlots: utils.Pointy(int(math.Log2(float64(numSlots)))),
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(*s.Params, btpLit)
	if err != nil {
		return nil, fmt.Errorf("bootstrap NewParametersFromLiteral: %w", err)
	}

	btpKeys, _, err := btpParams.GenEvaluationKeys(s.SecretKey)
	if err != nil {
		return nil, fmt.Errorf("bootstrap GenEvaluationKeys: %w", err)
	}

	data, err := btpKeys.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("bootstrap MarshalBinary: %w", err)
	}

	return data, nil
}
