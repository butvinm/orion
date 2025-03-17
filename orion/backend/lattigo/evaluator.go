package main

import (
	"C"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

var liveRotKeys = make(map[uint64]*rlwe.GaloisKey)
var savedRotKeys = []uint64{}

//export NewEvaluator
func NewEvaluator() {
	scheme.Evaluator = ckks.NewEvaluator(
		*scheme.Params, rlwe.NewMemEvaluationKeySet(scheme.RelinKey))

	// After declaring the evaluator, we'll also just generate and
	// store in memory all power of two rotation keys. This will ensure
	// all keys needed for the rotations and summations in the hyrid
	// method remain alive.
	AddPo2RotationKeys()
}

func AddPo2RotationKeys() {
	maxSlots := scheme.Params.MaxSlots()
	// Generate all positive power-of-two rotation keys
	for i := 1; i < maxSlots; i *= 2 {
		AddRotationKey(C.int(i))
	}
}

//export AddRotationKey
func AddRotationKey(rotation C.int) {
	galEl := scheme.Params.GaloisElement(int(rotation))

	// Generate the required rotation key if it doesn't exist
	if _, exists := liveRotKeys[galEl]; !exists {
		rotKey := scheme.KeyGen.GenGaloisKeyNew(galEl, scheme.SecretKey)
		liveRotKeys[galEl] = rotKey

		allKeysList := GetValuesFromMap(liveRotKeys)
		keys := rlwe.NewMemEvaluationKeySet(scheme.RelinKey, allKeysList...)
		scheme.Evaluator = scheme.Evaluator.WithKey(keys)
	}
}

//export Negate
func Negate(ciphertextID C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	ctOut, err := scheme.Evaluator.MulNew(ctIn, -1.0)
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export Rotate
func Rotate(ciphertextID, amount C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	AddRotationKey(amount)
	scheme.Evaluator.Rotate(ctIn, int(amount), ctIn)

	return ciphertextID
}

//export RotateNew
func RotateNew(ciphertextID, amount C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	AddRotationKey(amount)

	ctOut, err := scheme.Evaluator.RotateNew(ctIn, int(amount))
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export Rescale
func Rescale(ciphertextID C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	scheme.Evaluator.Rescale(ctIn, ctIn)

	return ciphertextID
}

//export RescaleNew
func RescaleNew(ciphertextID C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	scheme.Evaluator.Rescale(ctIn, ctIn)
	ctOut := ctIn.CopyNew()

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export AddScalar
func AddScalar(ciphertextID C.int, scalar C.float) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	scheme.Evaluator.Add(ctIn, float64(scalar), ctIn)

	return ciphertextID
}

//export AddScalarNew
func AddScalarNew(ciphertextID C.int, scalar C.float) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	ctOut, err := scheme.Evaluator.AddNew(ctIn, float64(scalar))
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export SubScalar
func SubScalar(ciphertextID C.int, scalar C.float) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	scheme.Evaluator.Sub(ctIn, float64(scalar), ctIn)

	return ciphertextID
}

//export SubScalarNew
func SubScalarNew(ciphertextID C.int, scalar C.float) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	ctOut, err := scheme.Evaluator.SubNew(ctIn, float64(scalar))
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export MulScalarInt
func MulScalarInt(ciphertextID C.int, scalar C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	scheme.Evaluator.Mul(ctIn, int(scalar), ctIn)

	return ciphertextID
}

//export MulScalarIntNew
func MulScalarIntNew(ciphertextID C.int, scalar C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	ctOut, err := scheme.Evaluator.MulNew(ctIn, int(scalar))
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export MulScalarFloat
func MulScalarFloat(ciphertextID C.int, scalar C.float) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	scheme.Evaluator.Mul(ctIn, float64(scalar), ctIn)

	return ciphertextID
}

//export MulScalarFloatNew
func MulScalarFloatNew(ciphertextID C.int, scalar C.float) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	ctOut, err := scheme.Evaluator.MulNew(ctIn, float64(scalar))
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export AddPlaintext
func AddPlaintext(ciphertextID, plaintextID C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	ptIn := RetrievePlaintext(int(plaintextID))
	scheme.Evaluator.Add(ctIn, ptIn, ctIn)

	return ciphertextID
}

//export AddPlaintextNew
func AddPlaintextNew(ciphertextID, plaintextID C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	ptIn := RetrievePlaintext(int(plaintextID))

	ctOut, err := scheme.Evaluator.AddNew(ctIn, ptIn)
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export SubPlaintext
func SubPlaintext(ciphertextID, plaintextID C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	ptIn := RetrievePlaintext(int(plaintextID))
	scheme.Evaluator.Sub(ctIn, ptIn, ctIn)

	return ciphertextID
}

//export SubPlaintextNew
func SubPlaintextNew(ciphertextID, plaintextID C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	ptIn := RetrievePlaintext(int(plaintextID))

	ctOut, err := scheme.Evaluator.SubNew(ctIn, ptIn)
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export MulPlaintext
func MulPlaintext(ciphertextID, plaintextID C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	ptIn := RetrievePlaintext(int(plaintextID))
	scheme.Evaluator.Mul(ctIn, ptIn, ctIn)

	return ciphertextID
}

//export MulPlaintextNew
func MulPlaintextNew(ciphertextID, plaintextID C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	ptIn := RetrievePlaintext(int(plaintextID))

	ctOut, err := scheme.Evaluator.MulNew(ctIn, ptIn)
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export AddCiphertext
func AddCiphertext(ctID0, ctID1 C.int) C.int {
	ctIn0 := RetrieveCiphertext(int(ctID0))
	ctIn1 := RetrieveCiphertext((int(ctID1)))
	scheme.Evaluator.Add(ctIn0, ctIn1, ctIn0)

	return ctID0
}

//export AddCiphertextNew
func AddCiphertextNew(ctID0, ctID1 C.int) C.int {
	ctIn0 := RetrieveCiphertext(int(ctID0))
	ctIn1 := RetrieveCiphertext((int(ctID1)))

	ctOut, err := scheme.Evaluator.AddNew(ctIn0, ctIn1)
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export SubCiphertext
func SubCiphertext(ctID0, ctID1 C.int) C.int {
	ctIn0 := RetrieveCiphertext(int(ctID0))
	ctIn1 := RetrieveCiphertext((int(ctID1)))
	scheme.Evaluator.Sub(ctIn0, ctIn1, ctIn0)

	return ctID0
}

//export SubCiphertextNew
func SubCiphertextNew(ctID0, ctID1 C.int) C.int {
	ctIn0 := RetrieveCiphertext(int(ctID0))
	ctIn1 := RetrieveCiphertext((int(ctID1)))

	ctOut, err := scheme.Evaluator.SubNew(ctIn0, ctIn1)
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export MulRelinCiphertext
func MulRelinCiphertext(ctID0, ctID1 C.int) C.int {
	ctIn0 := RetrieveCiphertext(int(ctID0))
	ctIn1 := RetrieveCiphertext((int(ctID1)))
	scheme.Evaluator.MulRelin(ctIn0, ctIn1, ctIn0)

	return ctID0
}

//export MulRelinCiphertextNew
func MulRelinCiphertextNew(ctID0, ctID1 C.int) C.int {
	ctIn0 := RetrieveCiphertext(int(ctID0))
	ctIn1 := RetrieveCiphertext((int(ctID1)))

	ctOut, err := scheme.Evaluator.MulRelinNew(ctIn0, ctIn1)
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

func DeleteRotationKeys() {
	liveRotKeys = make(map[uint64]*rlwe.GaloisKey)
	savedRotKeys = []uint64{}
}
