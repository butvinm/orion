package main

import (
	"C"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
)
import (
	"unsafe"
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
