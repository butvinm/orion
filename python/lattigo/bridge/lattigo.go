package main

// Bridge exports for Lattigo primitive types.
// Pure Lattigo bindings — no Orion imports.

//#include <stdlib.h>
//#include <stdint.h>
import "C"

import (
	"fmt"
	"runtime/cgo"
	"unsafe"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Wrapper structs store params alongside the object so callers
// don't need to pass params redundantly on every operation.

type encoderHandle struct {
	enc    *ckks.Encoder
	params *ckks.Parameters
}

type encryptorHandle struct {
	enc    *rlwe.Encryptor
	params *ckks.Parameters
}

type decryptorHandle struct {
	dec    *rlwe.Decryptor
	params *ckks.Parameters
}

// =====================================================================
// CKKS Parameters
// =====================================================================

//export NewCKKSParams
func NewCKKSParams(
	logn C.int,
	logqPtr *C.int, logqLen C.int,
	logpPtr *C.int, logpLen C.int,
	logDefaultScale C.int,
	h C.int,
	ringType *C.char,
	logNthRoot C.int,
	errOut **C.char,
) C.uintptr_t {
	defer catchPanic(errOut)

	logq := cIntsToGoInts(logqPtr, logqLen)
	logp := cIntsToGoInts(logpPtr, logpLen)

	rt := ring.ConjugateInvariant
	rtStr := C.GoString(ringType)
	switch rtStr {
	case "standard":
		rt = ring.Standard
	case "conjugate_invariant", "":
		rt = ring.ConjugateInvariant
	default:
		setErr(errOut, fmt.Sprintf("unknown ring type: %q", rtStr))
		return 0
	}

	lit := ckks.ParametersLiteral{
		LogN:            int(logn),
		LogQ:            logq,
		LogP:            logp,
		LogDefaultScale: int(logDefaultScale),
		RingType:        rt,
	}

	if int(h) > 0 {
		lit.Xs = ring.Ternary{H: int(h)}
	}

	if int(logNthRoot) > 0 {
		lit.LogNthRoot = int(logNthRoot)
	}

	params, err := ckks.NewParametersFromLiteral(lit)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(&params))
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

//export KeyGenGenRelinKey
func KeyGenGenRelinKey(kgH C.uintptr_t, skH C.uintptr_t, errOut **C.char) C.uintptr_t {
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
// Encoder
// =====================================================================

//export NewEncoder
func NewEncoder(paramsH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)
	enc := ckks.NewEncoder(*p)
	h := &encoderHandle{enc: enc, params: p}
	return C.uintptr_t(cgo.NewHandle(h))
}

//export EncoderEncode
func EncoderEncode(encH C.uintptr_t, values *C.double, numValues C.int, level C.int, scale C.ulonglong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	h := cgo.Handle(encH).Value().(*encoderHandle)

	goValues := cDoublesToGoFloat64s(values, numValues)
	pt := ckks.NewPlaintext(*h.params, int(level))
	pt.Scale = rlwe.NewScale(uint64(scale))

	if err := h.enc.Encode(goValues, pt); err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(pt))
}

//export EncoderDecode
func EncoderDecode(encH C.uintptr_t, ptH C.uintptr_t, numSlots C.int, outLen *C.int, errOut **C.char) *C.double {
	defer catchPanic(errOut)
	h := cgo.Handle(encH).Value().(*encoderHandle)
	pt := cgo.Handle(ptH).Value().(*rlwe.Plaintext)

	result := make([]float64, int(numSlots))
	if err := h.enc.Decode(pt, result); err != nil {
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

//export NewEncryptor
func NewEncryptor(paramsH C.uintptr_t, pkH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)
	pk := cgo.Handle(pkH).Value().(*rlwe.PublicKey)
	enc := ckks.NewEncryptor(*p, pk)
	h := &encryptorHandle{enc: enc, params: p}
	return C.uintptr_t(cgo.NewHandle(h))
}

//export EncryptorEncryptNew
func EncryptorEncryptNew(encryptorH C.uintptr_t, ptH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	h := cgo.Handle(encryptorH).Value().(*encryptorHandle)
	pt := cgo.Handle(ptH).Value().(*rlwe.Plaintext)

	ct := ckks.NewCiphertext(*h.params, 1, pt.Level())
	if err := h.enc.Encrypt(pt, ct); err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(ct))
}

// =====================================================================
// Decryptor
// =====================================================================

//export NewDecryptor
func NewDecryptor(paramsH C.uintptr_t, skH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)
	sk := cgo.Handle(skH).Value().(*rlwe.SecretKey)
	dec := ckks.NewDecryptor(*p, sk)
	h := &decryptorHandle{dec: dec, params: p}
	return C.uintptr_t(cgo.NewHandle(h))
}

//export DecryptorDecryptNew
func DecryptorDecryptNew(decryptorH C.uintptr_t, ctH C.uintptr_t, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	h := cgo.Handle(decryptorH).Value().(*decryptorHandle)
	ct := cgo.Handle(ctH).Value().(*rlwe.Ciphertext)

	pt := ckks.NewPlaintext(*h.params, ct.Level())
	h.dec.Decrypt(ct, pt)
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
// Serialization — RelinKey
// =====================================================================

//export RelinKeyMarshal
func RelinKeyMarshal(rlkH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
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

//export RelinKeyUnmarshal
func RelinKeyUnmarshal(data *C.char, dataLen C.ulong, errOut **C.char) C.uintptr_t {
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

// =====================================================================
// Bootstrap
// =====================================================================

//export NewBootstrapParams
func NewBootstrapParams(
	paramsH C.uintptr_t,
	logn C.int,
	logpPtr *C.int, logpLen C.int,
	h C.int,
	logSlots C.int,
	errOut **C.char,
) C.uintptr_t {
	defer catchPanic(errOut)
	p := cgo.Handle(paramsH).Value().(*ckks.Parameters)

	btpLit := bootstrapping.ParametersLiteral{}
	if int(logn) > 0 {
		v := int(logn)
		btpLit.LogN = &v
	}
	if int(logpLen) > 0 {
		btpLit.LogP = cIntsToGoInts(logpPtr, logpLen)
	}
	if int(h) > 0 {
		btpLit.Xs = ring.Ternary{H: int(h)}
	}
	if int(logSlots) > 0 {
		v := int(logSlots)
		btpLit.LogSlots = &v
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(*p, btpLit)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(&btpParams))
}

//export BootstrapParamsGenEvalKeys
func BootstrapParamsGenEvalKeys(
	btpParamsH C.uintptr_t,
	skH C.uintptr_t,
	outEvkH *C.uintptr_t,
	errOut **C.char,
) C.uintptr_t {
	defer catchPanic(errOut)
	btpParams := cgo.Handle(btpParamsH).Value().(*bootstrapping.Parameters)
	sk := cgo.Handle(skH).Value().(*rlwe.SecretKey)

	btpKeys, _, err := btpParams.GenEvaluationKeys(sk)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}

	evkHandle := C.uintptr_t(cgo.NewHandle(btpKeys.MemEvaluationKeySet))
	*outEvkH = evkHandle

	return C.uintptr_t(cgo.NewHandle(btpKeys))
}

//export BootstrapEvalKeysMarshal
func BootstrapEvalKeysMarshal(btpEvkH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
	defer catchPanic(errOut)
	btpKeys := cgo.Handle(btpEvkH).Value().(*bootstrapping.EvaluationKeys)
	data, err := btpKeys.MarshalBinary()
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}
