package main

//#include <stdlib.h>
//#include <stdint.h>
import "C"

import (
	"encoding/json"
	"runtime/cgo"
	"unsafe"

	orion "github.com/baahl-nyu/orion"
	"github.com/baahl-nyu/orion/client"
)

// parseParams parses a JSON string into orion.Params.
func parseParams(jsonStr string) (orion.Params, error) {
	var p orion.Params
	if err := json.Unmarshal([]byte(jsonStr), &p); err != nil {
		return p, err
	}
	return p, nil
}

// parseManifest parses a JSON string into orion.Manifest.
func parseManifest(jsonStr string) (orion.Manifest, error) {
	var m orion.Manifest
	if err := json.Unmarshal([]byte(jsonStr), &m); err != nil {
		return m, err
	}
	return m, nil
}

//export NewClient
func NewClient(paramsJSON *C.char, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	params, err := parseParams(C.GoString(paramsJSON))
	if err != nil {
		setErr(errOut, "parsing params: "+err.Error())
		return 0
	}
	c, err := client.New(params)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(c))
}

//export NewClientFromSecretKey
func NewClientFromSecretKey(paramsJSON *C.char, skData *C.char, skLen C.ulong, errOut **C.char) C.uintptr_t {
	defer catchPanic(errOut)
	params, err := parseParams(C.GoString(paramsJSON))
	if err != nil {
		setErr(errOut, "parsing params: "+err.Error())
		return 0
	}
	sk := cBytesToGoSlice(skData, skLen)
	c, err := client.FromSecretKey(params, sk)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(c))
}

//export ClientClose
func ClientClose(clientH C.uintptr_t) {
	defer logPanic()
	h := cgo.Handle(clientH)
	c := h.Value().(*client.Client)
	c.Close()
	// NOTE: Do NOT call h.Delete() here. ClientClose only does resource cleanup
	// (zeros the secret key). The Python GoHandle.close() calls DeleteHandle
	// separately to free the cgo handle slot (two-step close pattern).
}

//export ClientSecretKey
func ClientSecretKey(clientH C.uintptr_t, outLen *C.ulong, errOut **C.char) *C.char {
	defer catchPanic(errOut)
	c := cgo.Handle(clientH).Value().(*client.Client)
	data, err := c.SecretKey()
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
	defer catchPanic(errOut)
	c := cgo.Handle(clientH).Value().(*client.Client)
	goValues := cDoublesToGoFloat64s(values, numValues)
	pt, err := c.Encode(goValues, int(level), uint64(scale))
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(pt))
}

//export ClientDecode
func ClientDecode(clientH C.uintptr_t, ptH C.uintptr_t, outLen *C.int, errOut **C.char) *C.double {
	defer catchPanic(errOut)
	c := cgo.Handle(clientH).Value().(*client.Client)
	pt := cgo.Handle(ptH).Value().(*orion.Plaintext)
	vals, err := c.Decode(pt)
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
	defer catchPanic(errOut)
	c := cgo.Handle(clientH).Value().(*client.Client)
	pt := cgo.Handle(ptH).Value().(*orion.Plaintext)
	ct, err := c.Encrypt(pt)
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
	defer catchPanic(errOut)
	c := cgo.Handle(clientH).Value().(*client.Client)
	ct := cgo.Handle(ctH).Value().(*orion.Ciphertext)
	pts, err := c.Decrypt(ct)
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
	defer catchPanic(errOut)
	c := cgo.Handle(clientH).Value().(*client.Client)
	data, err := c.GenerateRLK()
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
	defer catchPanic(errOut)
	c := cgo.Handle(clientH).Value().(*client.Client)
	data, err := c.GenerateGaloisKey(uint64(galEl))
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
	defer catchPanic(errOut)
	c := cgo.Handle(clientH).Value().(*client.Client)
	goLogP := cIntsToGoInts(logP, logPLen)
	data, err := c.GenerateBootstrapKeys(int(slots), goLogP)
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
	defer catchPanic(errOut)
	c := cgo.Handle(clientH).Value().(*client.Client)
	manifest, err := parseManifest(C.GoString(manifestJSON))
	if err != nil {
		setErr(errOut, "parsing manifest: "+err.Error())
		return 0
	}
	bundle, err := c.GenerateKeys(manifest)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	return C.uintptr_t(cgo.NewHandle(bundle))
}

//export ClientMaxSlots
func ClientMaxSlots(clientH C.uintptr_t) C.int {
	defer logPanic()
	c := cgo.Handle(clientH).Value().(*client.Client)
	return C.int(c.MaxSlots())
}

//export ClientDefaultScale
func ClientDefaultScale(clientH C.uintptr_t) C.ulonglong {
	defer logPanic()
	c := cgo.Handle(clientH).Value().(*client.Client)
	return C.ulonglong(c.DefaultScale())
}

//export ClientGaloisElement
func ClientGaloisElement(clientH C.uintptr_t, rotation C.int) C.ulonglong {
	defer logPanic()
	c := cgo.Handle(clientH).Value().(*client.Client)
	return C.ulonglong(c.GaloisElement(int(rotation)))
}

//export ClientModuliChain
func ClientModuliChain(clientH C.uintptr_t, outLen *C.int, errOut **C.char) *C.ulonglong {
	defer catchPanic(errOut)
	c := cgo.Handle(clientH).Value().(*client.Client)
	chain := c.ModuliChain()
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
func ClientAuxModuliChain(clientH C.uintptr_t, outLen *C.int, errOut **C.char) *C.ulonglong {
	defer catchPanic(errOut)
	c := cgo.Handle(clientH).Value().(*client.Client)
	chain := c.AuxModuliChain()
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
