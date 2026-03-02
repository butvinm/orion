package main

// Bridge exports for Lattigo primitive types.
// These work with raw Lattigo types (*rlwe.SecretKey, *rlwe.Ciphertext, etc.),
// NOT the Orion wrapper types (*orion.Ciphertext, *orion.Plaintext).

//#include <stdlib.h>
//#include <stdint.h>
import "C"

import (
	"encoding/json"
	"runtime/cgo"
	"unsafe"

	orion "github.com/baahl-nyu/orion"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

// =====================================================================
// CKKS Parameters
// =====================================================================

//export NewCKKSParams
func NewCKKSParams(paramsJSON *C.char, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	var p orion.Params
	if err := json.Unmarshal([]byte(C.GoString(paramsJSON)), &p); err != nil {
		setErr(errOut, "parsing params: "+err.Error())
		return 0
	}
	ckksParams, err := p.NewCKKSParameters()
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(&ckksParams))
}

//export CKKSParamsMaxSlots
func CKKSParamsMaxSlots(paramsH C.uintptr_t) C.int {
	defer logPanic()
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)
	return C.int(p.MaxSlots())
}

//export CKKSParamsMaxLevel
func CKKSParamsMaxLevel(paramsH C.uintptr_t) C.int {
	defer logPanic()
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)
	return C.int(p.MaxLevel())
}

//export CKKSParamsDefaultScale
func CKKSParamsDefaultScale(paramsH C.uintptr_t) C.ulonglong {
	defer logPanic()
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)
	scale := p.DefaultScale()
	val, _ := scale.Value.Uint64()
	return C.ulonglong(val)
}

//export CKKSParamsGaloisElement
func CKKSParamsGaloisElement(paramsH C.uintptr_t, rotation C.int) C.ulonglong {
	defer logPanic()
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)
	return C.ulonglong(p.GaloisElement(int(rotation)))
}

//export CKKSParamsModuliChain
func CKKSParamsModuliChain(paramsH C.uintptr_t, outLen *C.int, errOut **C.char) *C.ulonglong {
	defer catchPanic(errOut)
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)
	qi := p.Q()
	n := len(qi)
	if n == 0 {
		*outLen = 0
		return nil
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.ulonglong(0)))
	ptr := (*C.ulonglong)(C.malloc(size))
	slice := unsafe.Slice(ptr, n)
	for i, v := range qi {
		slice[i] = C.ulonglong(v)
	}
	*outLen = C.int(n)
	return ptr
}

//export CKKSParamsAuxModuliChain
func CKKSParamsAuxModuliChain(paramsH C.uintptr_t, outLen *C.int, errOut **C.char) *C.ulonglong {
	defer catchPanic(errOut)
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)
	pi := p.P()
	n := len(pi)
	if n == 0 {
		*outLen = 0
		return nil
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.ulonglong(0)))
	ptr := (*C.ulonglong)(C.malloc(size))
	slice := unsafe.Slice(ptr, n)
	for i, v := range pi {
		slice[i] = C.ulonglong(v)
	}
	*outLen = C.int(n)
	return ptr
}

// =====================================================================
// KeyGenerator
// =====================================================================

//export NewKeyGenerator
func NewKeyGenerator(paramsH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)
	kg := rlwe.NewKeyGenerator(*p)
	return C.uintptr_t(cgo.NewHandle(kg))
}

//export KeyGenGenSecretKey
func KeyGenGenSecretKey(kgH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	kg := cgo.Handle(kgH).Value().(*rlwe.KeyGenerator)
	sk := kg.GenSecretKeyNew()
	return C.uintptr_t(cgo.NewHandle(sk))
}

//export KeyGenGenPublicKey
func KeyGenGenPublicKey(kgH C.uintptr_t, skH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	kg := cgo.Handle(kgH).Value().(*rlwe.KeyGenerator)
	sk := cgo.Handle(skH).Value().(*rlwe.SecretKey)
	pk := kg.GenPublicKeyNew(sk)
	return C.uintptr_t(cgo.NewHandle(pk))
}

//export KeyGenGenRelinearizationKey
func KeyGenGenRelinearizationKey(kgH C.uintptr_t, skH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	kg := cgo.Handle(kgH).Value().(*rlwe.KeyGenerator)
	sk := cgo.Handle(skH).Value().(*rlwe.SecretKey)
	rlk := kg.GenRelinearizationKeyNew(sk)
	return C.uintptr_t(cgo.NewHandle(rlk))
}

//export KeyGenGenGaloisKey
func KeyGenGenGaloisKey(kgH C.uintptr_t, skH C.uintptr_t, galEl C.ulonglong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	kg := cgo.Handle(kgH).Value().(*rlwe.KeyGenerator)
	sk := cgo.Handle(skH).Value().(*rlwe.SecretKey)
	gk := kg.GenGaloisKeyNew(uint64(galEl), sk)
	return C.uintptr_t(cgo.NewHandle(gk))
}

// =====================================================================
// CKKS Encoder
// =====================================================================

//export NewCKKSEncoder
func NewCKKSEncoder(paramsH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)
	enc := ckks.NewEncoder(*p)
	return C.uintptr_t(cgo.NewHandle(enc))
}

//export CKKSEncoderEncode
func CKKSEncoderEncode(encH C.uintptr_t, paramsH C.uintptr_t, values *C.double, numValues C.int, level C.int, scale C.ulonglong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	enc := cgo.Handle(encH).Value().(*ckks.Encoder)
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)

	goValues := cDoublesToGoFloat64s(values, numValues)
	pt := ckks.NewPlaintext(*p, int(level))
	pt.Scale = rlwe.NewScale(uint64(scale))

	if err := enc.Encode(goValues, pt); err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(pt))
}

//export CKKSEncoderDecode
func CKKSEncoderDecode(encH C.uintptr_t, ptH C.uintptr_t, numSlots C.int, outLen *C.int, errOut **C.char) *C.double {
	defer catchPanic(errOut)
	enc := cgo.Handle(encH).Value().(*ckks.Encoder)
	pt := cgo.Handle(ptH).Value().(*rlwe.Plaintext)

	result := make([]float64, int(numSlots))
	if err := enc.Decode(pt, result); err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goFloat64sToCDoubles(result)
	*outLen = length
	return ptr
}

// =====================================================================
// Encryptor
// =====================================================================

//export NewCKKSEncryptor
func NewCKKSEncryptor(paramsH C.uintptr_t, pkH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)
	pk := cgo.Handle(pkH).Value().(*rlwe.PublicKey)
	encryptor := ckks.NewEncryptor(*p, pk)
	return C.uintptr_t(cgo.NewHandle(encryptor))
}

//export EncryptorEncryptNew
func EncryptorEncryptNew(encryptorH C.uintptr_t, ptH C.uintptr_t, paramsH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	encryptor := cgo.Handle(encryptorH).Value().(*rlwe.Encryptor)
	pt := cgo.Handle(ptH).Value().(*rlwe.Plaintext)
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)

	ct := ckks.NewCiphertext(*p, 1, pt.Level())
	if err := encryptor.Encrypt(pt, ct); err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(ct))
}

// =====================================================================
// Decryptor
// =====================================================================

//export NewCKKSDecryptor
func NewCKKSDecryptor(paramsH C.uintptr_t, skH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)
	sk := cgo.Handle(skH).Value().(*rlwe.SecretKey)
	decryptor := ckks.NewDecryptor(*p, sk)
	return C.uintptr_t(cgo.NewHandle(decryptor))
}

//export DecryptorDecryptNew
func DecryptorDecryptNew(decryptorH C.uintptr_t, ctH C.uintptr_t, paramsH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	decryptor := cgo.Handle(decryptorH).Value().(*rlwe.Decryptor)
	ct := cgo.Handle(ctH).Value().(*rlwe.Ciphertext)
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)

	pt := ckks.NewPlaintext(*p, ct.Level())
	decryptor.Decrypt(ct, pt)
	return C.uintptr_t(cgo.NewHandle(pt))
}

// =====================================================================
// Serialization — SecretKey
// =====================================================================

//export SecretKeyMarshal
func SecretKeyMarshal(skH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
	defer catchPanic(errOut)
	sk := cgo.Handle(skH).Value().(*rlwe.SecretKey)
	data, err := sk.MarshalBinary()
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export SecretKeyUnmarshal
func SecretKeyUnmarshal(data *C.char, dataLen C.ulong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goData := cBytesToGoSlice(data, dataLen)
	sk := new(rlwe.SecretKey)
	if err := sk.UnmarshalBinary(goData); err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(sk))
}

// =====================================================================
// Serialization — PublicKey
// =====================================================================

//export PublicKeyMarshal
func PublicKeyMarshal(pkH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
	defer catchPanic(errOut)
	pk := cgo.Handle(pkH).Value().(*rlwe.PublicKey)
	data, err := pk.MarshalBinary()
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export PublicKeyUnmarshal
func PublicKeyUnmarshal(data *C.char, dataLen C.ulong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goData := cBytesToGoSlice(data, dataLen)
	pk := new(rlwe.PublicKey)
	if err := pk.UnmarshalBinary(goData); err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(pk))
}

// =====================================================================
// Serialization — RelinearizationKey
// =====================================================================

//export RelinearizationKeyMarshal
func RelinearizationKeyMarshal(rlkH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
	defer catchPanic(errOut)
	rlk := cgo.Handle(rlkH).Value().(*rlwe.RelinearizationKey)
	data, err := rlk.MarshalBinary()
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export RelinearizationKeyUnmarshal
func RelinearizationKeyUnmarshal(data *C.char, dataLen C.ulong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goData := cBytesToGoSlice(data, dataLen)
	rlk := new(rlwe.RelinearizationKey)
	if err := rlk.UnmarshalBinary(goData); err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(rlk))
}

// =====================================================================
// Serialization — GaloisKey
// =====================================================================

//export GaloisKeyMarshal
func GaloisKeyMarshal(gkH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
	defer catchPanic(errOut)
	gk := cgo.Handle(gkH).Value().(*rlwe.GaloisKey)
	data, err := gk.MarshalBinary()
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export GaloisKeyUnmarshal
func GaloisKeyUnmarshal(data *C.char, dataLen C.ulong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goData := cBytesToGoSlice(data, dataLen)
	gk := new(rlwe.GaloisKey)
	if err := gk.UnmarshalBinary(goData); err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(gk))
}

// =====================================================================
// Serialization — raw rlwe.Ciphertext
// =====================================================================

//export RLWECiphertextMarshal
func RLWECiphertextMarshal(ctH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
	defer catchPanic(errOut)
	ct := cgo.Handle(ctH).Value().(*rlwe.Ciphertext)
	data, err := ct.MarshalBinary()
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export RLWECiphertextUnmarshal
func RLWECiphertextUnmarshal(data *C.char, dataLen C.ulong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goData := cBytesToGoSlice(data, dataLen)
	ct := new(rlwe.Ciphertext)
	if err := ct.UnmarshalBinary(goData); err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(ct))
}

//export RLWECiphertextLevel
func RLWECiphertextLevel(ctH C.uintptr_t) C.int {
	defer logPanic()
	ct := cgo.Handle(ctH).Value().(*rlwe.Ciphertext)
	return C.int(ct.Level())
}

// =====================================================================
// Serialization — raw rlwe.Plaintext
// =====================================================================

//export RLWEPlaintextMarshal
func RLWEPlaintextMarshal(ptH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
	defer catchPanic(errOut)
	pt := cgo.Handle(ptH).Value().(*rlwe.Plaintext)
	data, err := pt.MarshalBinary()
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export RLWEPlaintextUnmarshal
func RLWEPlaintextUnmarshal(data *C.char, dataLen C.ulong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goData := cBytesToGoSlice(data, dataLen)
	pt := new(rlwe.Plaintext)
	if err := pt.UnmarshalBinary(goData); err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(pt))
}

//export RLWEPlaintextLevel
func RLWEPlaintextLevel(ptH C.uintptr_t) C.int {
	defer logPanic()
	pt := cgo.Handle(ptH).Value().(*rlwe.Plaintext)
	return C.int(pt.Level())
}

// =====================================================================
// MemEvaluationKeySet
// =====================================================================

//export NewMemEvalKeySet
func NewMemEvalKeySet(rlkH C.uintptr_t, gkHs *C.uintptr_t, numGks C.int, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)

	var rlk *rlwe.RelinearizationKey
	if uint64(rlkH) != 0 {
		rlk = cgo.Handle(rlkH).Value().(*rlwe.RelinearizationKey)
	}

	n := int(numGks)
	galKeys := make([]*rlwe.GaloisKey, n)
	if n > 0 {
		gkSlice := unsafe.Slice(gkHs, n)
		for i := 0; i < n; i++ {
			galKeys[i] = cgo.Handle(gkSlice[i]).Value().(*rlwe.GaloisKey)
		}
	}

	evk := rlwe.NewMemEvaluationKeySet(rlk, galKeys...)
	return C.uintptr_t(cgo.NewHandle(evk))
}

//export MemEvalKeySetMarshal
func MemEvalKeySetMarshal(evkH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
	defer catchPanic(errOut)
	evk := cgo.Handle(evkH).Value().(*rlwe.MemEvaluationKeySet)
	data, err := evk.MarshalBinary()
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export MemEvalKeySetUnmarshal
func MemEvalKeySetUnmarshal(data *C.char, dataLen C.ulong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	goData := cBytesToGoSlice(data, dataLen)
	evk := new(rlwe.MemEvaluationKeySet)
	if err := evk.UnmarshalBinary(goData); err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(evk))
}
