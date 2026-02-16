package main

import (
	"C"
	"unsafe"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
)

//export NewKeyGenerator
func NewKeyGenerator() {
	scheme.KeyGen = rlwe.NewKeyGenerator(scheme.Params)
}

//export GenerateSecretKey
func GenerateSecretKey() {
	scheme.SecretKey = scheme.KeyGen.GenSecretKeyNew()
}

//export GeneratePublicKey
func GeneratePublicKey() {
	scheme.PublicKey = scheme.KeyGen.GenPublicKeyNew(scheme.SecretKey)
}

//export GenerateRelinearizationKey
func GenerateRelinearizationKey() {
	scheme.RelinKey = scheme.KeyGen.GenRelinearizationKeyNew(scheme.SecretKey)
}

//export GenerateEvaluationKeys
func GenerateEvaluationKeys() {
	scheme.EvalKeys = rlwe.NewMemEvaluationKeySet(scheme.RelinKey)
}

//export SerializeSecretKey
func SerializeSecretKey() (*C.char, C.ulong) {
	data, err := scheme.SecretKey.MarshalBinary()
	if err != nil {
		panic(err)
	}

	arrPtr, length := SliceToCArray(data, convertByteToCChar)
	return arrPtr, length
}

//export LoadSecretKey
func LoadSecretKey(dataPtr *C.char, lenData C.ulong) {
	skSerial := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))

	sk := &rlwe.SecretKey{}
	if err := sk.UnmarshalBinary(skSerial); err != nil {
		panic(err)
	}

	scheme.SecretKey = sk
}

//export SerializePublicKey
func SerializePublicKey() (*C.char, C.ulong) {
	data, err := scheme.PublicKey.MarshalBinary()
	if err != nil {
		panic(err)
	}

	arrPtr, length := SliceToCArray(data, convertByteToCChar)
	return arrPtr, length
}

//export LoadPublicKey
func LoadPublicKey(dataPtr *C.char, lenData C.ulong) {
	pkSerial := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))

	pk := rlwe.NewPublicKey(scheme.Params)
	if err := pk.UnmarshalBinary(pkSerial); err != nil {
		panic(err)
	}

	scheme.PublicKey = pk
}

//export SerializeRelinKey
func SerializeRelinKey() (*C.char, C.ulong) {
	data, err := scheme.RelinKey.MarshalBinary()
	if err != nil {
		panic(err)
	}

	arrPtr, length := SliceToCArray(data, convertByteToCChar)
	return arrPtr, length
}

//export LoadRelinKey
func LoadRelinKey(dataPtr *C.char, lenData C.ulong) {
	rlkSerial := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))

	rlk := rlwe.NewRelinearizationKey(scheme.Params)
	if err := rlk.UnmarshalBinary(rlkSerial); err != nil {
		panic(err)
	}

	scheme.RelinKey = rlk
}
