package main

import (
	"C"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/polynomial"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)
import (
	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/lintrans"
)

type Scheme struct {
	Params        *ckks.Parameters
	KeyGen        *rlwe.KeyGenerator
	SecretKey     *rlwe.SecretKey
	PublicKey     *rlwe.PublicKey
	RelinKey      *rlwe.RelinearizationKey
	EvalKeys      *rlwe.MemEvaluationKeySet
	Encoder       *ckks.Encoder
	Encryptor     *rlwe.Encryptor
	Decryptor     *rlwe.Decryptor
	Evaluator     *ckks.Evaluator
	PolyEvaluator *polynomial.Evaluator
	LinEvaluator  *lintrans.Evaluator
	Bootstrapper  *bootstrapping.Evaluator
}

var scheme Scheme

//export NewScheme
func NewScheme(
	logN C.int,
	logQPtr *C.int, lenQ C.int,
	logPPtr *C.int, lenP C.int,
	logScale C.int,
	h C.int,
	ringType *C.char,
	keysPath *C.char,
	ioMode *C.char,
) {
	// Convert LogQ and LogP to Go slices
	logQ := CArrayToSlice(logQPtr, lenQ, convertCIntToInt)
	logP := CArrayToSlice(logPPtr, lenP, convertCIntToInt)

	ringT := ring.Standard
	if C.GoString(ringType) != "standard" {
		ringT = ring.ConjugateInvariant
	}

	var err error
	var params ckks.Parameters

	if params, err = ckks.NewParametersFromLiteral(
		ckks.ParametersLiteral{
			LogN:            int(logN),
			LogQ:            logQ,
			LogP:            logP,
			LogDefaultScale: int(logScale),
			Xs:              ring.Ternary{H: int(h)},
			RingType:        ringT,
		}); err != nil {
		panic(err)
	}

	keyGen := rlwe.NewKeyGenerator(params)
	secretKey := GenSecretKeyNew(keyGen, keysPath, ioMode)
	publicKey := keyGen.GenPublicKeyNew(secretKey)

	relinKey := keyGen.GenRelinearizationKeyNew(secretKey)
	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, publicKey)
	decryptor := ckks.NewDecryptor(params, secretKey)
	evaluator := ckks.NewEvaluator(params, rlwe.NewMemEvaluationKeySet(relinKey))
	polyeval := polynomial.NewEvaluator(params, evaluator)

	// We'll instantiate a different evaluator for linear transforms so that
	// they can freely manipulate rotation keys
	evalKeys := rlwe.NewMemEvaluationKeySet(relinKey)
	lineval := lintrans.NewEvaluator(ckks.NewEvaluator(params, evalKeys))

	scheme = Scheme{
		Params:        &params,
		KeyGen:        keyGen,
		SecretKey:     secretKey,
		PublicKey:     publicKey,
		RelinKey:      relinKey,
		EvalKeys:      evalKeys,
		Encoder:       encoder,
		Encryptor:     encryptor,
		Decryptor:     decryptor,
		Evaluator:     evaluator,
		PolyEvaluator: polyeval,
		LinEvaluator:  lineval,
		Bootstrapper:  nil,
	}

	// We'll add the power-of-two rotation keys to our evaluator,
	// and they'll be kept in RAM for the duration of the program.
	AddPo2RotationKeys()
}

//export DeleteScheme
func DeleteScheme() {
	scheme = Scheme{}

	DeleteRotationKeys()
	DeleteBootstrappers()
	DeleteMinimaxSignMap()

	ltHeap.Reset()
	polyHeap.Reset()
	ptHeap.Reset()
	ctHeap.Reset()
}

func AddPo2RotationKeys() {
	maxSlots := scheme.Params.MaxSlots()

	// Generate all positive power-of-two rotation keys
	for i := 1; i < maxSlots; i *= 2 {
		AddRotationKey(i)
	}
}
