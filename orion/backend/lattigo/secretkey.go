package main

import (
	"C"
	"fmt"
	"log"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"gonum.org/v1/hdf5"
)

func GenSecretKeyNew(
	keyGen *rlwe.KeyGenerator,
	keysPath *C.char,
	ioMode *C.char,
) *rlwe.SecretKey {
	mode := C.GoString(ioMode)

	// Either generate or load the secret key.
	var sk *rlwe.SecretKey
	if mode != "load" {
		sk = keyGen.GenSecretKeyNew()
	} else {
		sk = LoadSecretKey(keysPath)
	}

	// We'll also serialize and save the secret key if needed.
	if mode == "save" {
		SaveSecretKey(sk, keysPath)
	}

	return sk
}

func LoadSecretKey(
	keysPath *C.char,
) *rlwe.SecretKey {
	filename := C.GoString(keysPath)
	fmt.Println("Loading existing secret key from", filename)

	// Open the HDF5 file in read-only mode.
	file, err := hdf5.OpenFile(filename, hdf5.F_ACC_RDONLY)
	if err != nil {
		log.Fatal("error opening hdf5 file:", err)
	}
	defer file.Close()

	// Open the dataset containing the secret key.
	dset, err := file.OpenDataset("sk")
	if err != nil {
		log.Fatal("error opening dataset 'sk':", err)
	}
	defer dset.Close()

	// Get the dataspace and its size.
	space := dset.Space()
	if space == nil {
		log.Fatal("error retrieving dataspace for 'sk'")
	}
	defer space.Close()

	// Allocate a buffer for the binary data.
	nElems := space.SimpleExtentNPoints()
	data := make([]byte, nElems)

	// Read the binary data into the buffer.
	if err := dset.Read(&data); err != nil {
		log.Fatal("error reading dataset 'sk':", err)
	}

	// Unmarshal the binary data into a secret key.
	sk := &rlwe.SecretKey{}
	if err := sk.UnmarshalBinary(data); err != nil {
		log.Fatal("error unmarshaling secret key:", err)
	}

	return sk
}

func SaveSecretKey(
	sk *rlwe.SecretKey,
	keysPath *C.char,
) {
	filename := C.GoString(keysPath)
	fmt.Println("Saving secret key to", filename)

	// Create a new HDF5 file, overwriting if it exists.
	file, err := hdf5.CreateFile(filename, hdf5.F_ACC_TRUNC)
	if err != nil {
		log.Fatal("error creating hdf5 file:", err)
	}
	defer file.Close()

	// Serialize secret key.
	skBinary, err := sk.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}

	dataspace, err := hdf5.CreateSimpleDataspace([]uint{uint(len(skBinary))}, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer dataspace.Close()

	// Create the dataset for the secret key.
	dset, err := file.CreateDataset("sk", hdf5.T_NATIVE_UINT8, dataspace)
	if err != nil {
		log.Fatal(err)
	}
	defer dset.Close()

	// Write the binary data to the dataset.
	if err := dset.Write(&skBinary); err != nil {
		log.Fatal(err)
	}
}
