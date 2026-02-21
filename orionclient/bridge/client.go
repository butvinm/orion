package main

//#include <stdlib.h>
//#include <stdint.h>
import "C"

import (
	"encoding/json"
	"runtime/cgo"
	"unsafe"

	orionclient "github.com/baahl-nyu/orion/orionclient"
)

// parseParams parses a JSON string into orionclient.Params.
func parseParams(jsonStr string) (orionclient.Params, error) {
	var p orionclient.Params
	if err := json.Unmarshal([]byte(jsonStr), &p); err != nil {
		return p, err
	}
	return p, nil
}

// parseManifest parses a JSON string into orionclient.Manifest.
func parseManifest(jsonStr string) (orionclient.Manifest, error) {
	var m orionclient.Manifest
	if err := json.Unmarshal([]byte(jsonStr), &m); err != nil {
		return m, err
	}
	return m, nil
}

//export NewClient
func NewClient(paramsJSON *C.char, errOut **C.char) C.uintptr_t {
	params, err := parseParams(C.GoString(paramsJSON))
	if err != nil {
		setErr(errOut, "parsing params: "+err.Error())
		return 0
	}
	client, err := orionclient.New(params)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(client))
}

//export NewClientFromSecretKey
func NewClientFromSecretKey(paramsJSON *C.char, skData *C.char, skLen C.ulong, errOut **C.char) C.uintptr_t {
	params, err := parseParams(C.GoString(paramsJSON))
	if err != nil {
		setErr(errOut, "parsing params: "+err.Error())
		return 0
	}
	sk := cBytesToGoSlice(skData, skLen)
	client, err := orionclient.FromSecretKey(params, sk)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(client))
}

//export ClientClose
func ClientClose(clientH C.uintptr_t) {
	h := cgo.Handle(clientH)
	client := h.Value().(*orionclient.Client)
	client.Close()
	h.Delete()
}

//export ClientSecretKey
func ClientSecretKey(clientH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	data, err := client.SecretKey()
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export ClientEncode
func ClientEncode(clientH C.uintptr_t, values *C.double, numValues C.int, level C.int, scale C.ulonglong, errOut **C.char) C.uintptr_t {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	goValues := cDoublesToGoFloat64s(values, numValues)
	pt, err := client.Encode(goValues, int(level), uint64(scale))
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(pt))
}

//export ClientDecode
func ClientDecode(clientH C.uintptr_t, ptH C.uintptr_t, outLen *C.int, errOut **C.char) *C.double {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	vals, err := client.Decode(pt)
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goFloat64sToCDoubles(vals)
	*outLen = length
	return ptr
}

//export ClientEncrypt
func ClientEncrypt(clientH C.uintptr_t, ptH C.uintptr_t, errOut **C.char) C.uintptr_t {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	pt := cgo.Handle(ptH).Value().(*orionclient.Plaintext)
	ct, err := client.Encrypt(pt)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(ct))
}

// ClientDecrypt returns a C array of plaintext handles (one per underlying ciphertext).
// The caller must free the returned array with FreeCArray and each handle with DeleteHandle.
//
//export ClientDecrypt
func ClientDecrypt(clientH C.uintptr_t, ctH C.uintptr_t, numOut *C.int, errOut **C.char) *C.uintptr_t {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	ct := cgo.Handle(ctH).Value().(*orionclient.Ciphertext)
	pts, err := client.Decrypt(ct)
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	n := len(pts)
	*numOut = C.int(n)
	if n == 0 {
		return nil
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.uintptr_t(0)))
	ptr := (*C.uintptr_t)(C.malloc(size))
	handles := unsafe.Slice(ptr, n)
	for i, pt := range pts {
		handles[i] = C.uintptr_t(cgo.NewHandle(pt))
	}
	return ptr
}

//export ClientGenerateRLK
func ClientGenerateRLK(clientH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	data, err := client.GenerateRLK()
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export ClientGenerateGaloisKey
func ClientGenerateGaloisKey(clientH C.uintptr_t, galEl C.ulonglong, outLen *C.ulong, errOut **C.char) *C.char {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	data, err := client.GenerateGaloisKey(uint64(galEl))
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

//export ClientGenerateBootstrapKeys
func ClientGenerateBootstrapKeys(clientH C.uintptr_t, slots C.int, logP *C.int, logPLen C.int, outLen *C.ulong, errOut **C.char) *C.char {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	goLogP := cIntsToGoInts(logP, logPLen)
	data, err := client.GenerateBootstrapKeys(int(slots), goLogP)
	if err != nil {
		setErr(errOut, err.Error())
		return nil
	}
	ptr, length := goSliceToCBytes(data)
	*outLen = length
	return ptr
}

// ClientGenerateKeys generates all evaluation keys from a JSON-encoded Manifest.
// Returns a handle to an EvalKeyBundle.
//
//export ClientGenerateKeys
func ClientGenerateKeys(clientH C.uintptr_t, manifestJSON *C.char, errOut **C.char) C.uintptr_t {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	manifest, err := parseManifest(C.GoString(manifestJSON))
	if err != nil {
		setErr(errOut, "parsing manifest: "+err.Error())
		return 0
	}
	bundle, err := client.GenerateKeys(manifest)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(bundle))
}

//export ClientMaxSlots
func ClientMaxSlots(clientH C.uintptr_t) C.int {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	return C.int(client.MaxSlots())
}

//export ClientDefaultScale
func ClientDefaultScale(clientH C.uintptr_t) C.ulonglong {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	return C.ulonglong(client.DefaultScale())
}

//export ClientGaloisElement
func ClientGaloisElement(clientH C.uintptr_t, rotation C.int) C.ulonglong {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	return C.ulonglong(client.GaloisElement(int(rotation)))
}

//export ClientModuliChain
func ClientModuliChain(clientH C.uintptr_t, outLen *C.int) *C.ulonglong {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	chain := client.ModuliChain()
	n := len(chain)
	if n == 0 {
		*outLen = 0
		return nil
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.ulonglong(0)))
	ptr := (*C.ulonglong)(C.malloc(size))
	slice := unsafe.Slice(ptr, n)
	for i, v := range chain {
		slice[i] = C.ulonglong(v)
	}
	*outLen = C.int(n)
	return ptr
}

//export ClientAuxModuliChain
func ClientAuxModuliChain(clientH C.uintptr_t, outLen *C.int) *C.ulonglong {
	client := cgo.Handle(clientH).Value().(*orionclient.Client)
	chain := client.AuxModuliChain()
	n := len(chain)
	if n == 0 {
		*outLen = 0
		return nil
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.ulonglong(0)))
	ptr := (*C.ulonglong)(C.malloc(size))
	slice := unsafe.Slice(ptr, n)
	for i, v := range chain {
		slice[i] = C.ulonglong(v)
	}
	*outLen = C.int(n)
	return ptr
}
