package main

import (
	"C"
	"math"

	"fmt"
	"os"
	"strconv"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/lintrans"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"

	"github.com/baahl-nyu/lattigo/v6/ring/ringqp"
	"gonum.org/v1/hdf5"
)
import "slices"

var ltHeap = NewHeapAllocator()

func AddLinearTransform(lt lintrans.LinearTransformation) int {
	return ltHeap.Add(lt)
}

func RetrieveLinearTransform(id int) lintrans.LinearTransformation {
	return ltHeap.Retrieve(id).(lintrans.LinearTransformation)
}

//export DeleteLinearTransform
func DeleteLinearTransform(id C.int) {
	ltHeap.Delete(int(id))
}

//export GenerateLinearTransform
func GenerateLinearTransform(
	diagIdxsC *C.int, diagIdxsLen C.int,
	diagDataC *C.float, diagDataLen C.int,
	level C.int,
	bsgsRatio C.float,
	blockRow C.int,
	blockCol C.int,
	moduleNameC *C.char,
	diagsPathC *C.char,
	keysPathC *C.char,
	ioModeC *C.char,
) C.int {

	moduleName := C.GoString(moduleNameC)
	diagsPath := C.GoString(diagsPathC)
	keysPath := C.GoString(keysPathC)
	ioMode := C.GoString(ioModeC)

	// Unload diags data
	diagIdxs := CArrayToSlice(diagIdxsC, diagIdxsLen, convertCIntToInt)
	diagDataFlat := CArrayToSlice(diagDataC, diagDataLen, convertCFloatToFloat)

	// diagDataFlat is a flattened array of length len(diagIdxs) * slots.
	// The first element in diagIdxs corresponds to the first [0, slots]
	// values in diagsDataFlat, and so on. We'll extract these into a
	// dictionary that can be passed to Lattigo's LinearTransform evaluator.
	slots := scheme.Params.MaxSlots()
	diagonals := make(lintrans.Diagonals[float64])

	for i, key := range diagIdxs {
		diagonals[key] = diagDataFlat[i*slots : (i+1)*slots]
	}

	ltparams := lintrans.Parameters{
		DiagonalsIndexList:        diagonals.DiagonalsIndexList(),
		LevelQ:                    int(level),
		LevelP:                    scheme.Params.MaxLevelP(),
		Scale:                     rlwe.NewScale(scheme.Params.Q()[int(level)]),
		LogDimensions:             ring.Dimensions{Rows: 0, Cols: scheme.Params.LogMaxSlots()},
		LogBabyStepGiantStepRatio: int(math.Log(float64(bsgsRatio))),
	}

	lt := lintrans.NewTransformation(scheme.Params, ltparams)

	// ---------------------------- //
	//  Diagonal Generation/Saving  //
	// ---------------------------- //

	// If ioMode is "load", then we expect the diagonals to have already been
	// generated and serialized, so there's no need to regenerate them here.
	// We do, however, still need to instantiate empty plaintext diagonals.
	if ioMode == "load" {
		lt.Vec = make(map[int]ringqp.Poly)
		for _, diag := range diagIdxs {
			lt.Vec[diag] = ringqp.Poly{}
		}
	} else { // otherwise, generate diagonals here.
		if err := lintrans.Encode(scheme.Encoder, diagonals, lt); err != nil {
			panic(err)
		}
	}

	if ioMode == "save" {
		SaveDiagonals(lt, diagsPath, moduleName, int(blockCol), int(blockRow))

		// Force delete as we'll be loading them in during inference.
		for i := range lt.Vec {
			lt.Vec[i] = ringqp.Poly{}
		}
	}

	// ----------------------- //
	//  Key Generation/Saving  //
	// ----------------------- //

	if ioMode != "load" {
		// Get required rotations by the current linear transform
		ltRots := lintrans.GaloisElements(scheme.Params, ltparams)

		if ioMode == "save" {
			// When IO mode is `save`, then we'll need to generate and save
			// rotation keys. To avoid re-generating previous keys, we'll
			// instead track already saved rotation keys and only generate
			// new ones for this linear transform.
			rotsToSave := []uint64{}
			for _, rot := range ltRots {
				if !slices.Contains(savedRotKeys, rot) {
					rotsToSave = append(rotsToSave, rot)
				}
			}

			rotKeys := scheme.KeyGen.GenGaloisKeysNew(rotsToSave, scheme.SecretKey)
			SaveRotationKeys(rotsToSave, rotKeys, keysPath)

			// Now we'll just add our saved rotation keys to the list of all
			// saved rotation keys to avoid generating them in the future.
			savedRotKeys = append(savedRotKeys, rotsToSave...)
		} else {
			// Otherwise, IO mode is `none` here. In this case, we won't be
			// saving rotation keys. Instead, we'll be using a list of live
			// rotation keys currently in memory to determine if we need to
			// generate a key. If it's already in mem, then we skip.
			rotsToAdd := []uint64{}
			for _, rot := range ltRots {
				if _, exists := liveRotKeys[rot]; !exists {
					rotsToAdd = append(rotsToAdd, rot)
				}
			}

			// Generate only the new rotation keys not in mem
			rotKeys := scheme.KeyGen.GenGaloisKeysNew(rotsToAdd, scheme.SecretKey)
			for i, rot := range rotsToAdd {
				liveRotKeys[rot] = rotKeys[i]
			}

			// Now our linear transform evaluator will have access to all
			// generated keys, including from previous linear transforms.
			allRotKeys := GetValuesFromMap(liveRotKeys)
			scheme.LinEvaluator = lintrans.NewEvaluator(scheme.Evaluator.WithKey(
				rlwe.NewMemEvaluationKeySet(scheme.RelinKey, allRotKeys...),
			))
		}
	}

	// Return reference to linear transform object we just created
	ltID := ltHeap.Add(lt)
	return C.int(ltID)
}

//export EvaluateLinearTransforms
func EvaluateLinearTransforms(
	transformIDs *C.int, lenTransformIDs C.int,
	ctIDs *C.int, lenCtIDs C.int,
	moduleNameC *C.char,
	diagsPathC *C.char,
	keysPathC *C.char,
	ioModeC *C.char,
) (*C.int, C.int) {

	moduleName := C.GoString(moduleNameC)
	diagsPath := C.GoString(diagsPathC)
	keysPath := C.GoString(keysPathC)
	ioMode := C.GoString(ioModeC)

	// First we'll reconstruct our (row, col) format of blocked transforms.
	// For matrix dimensions, we know there must be the same number of column
	// blocks ase there are input ciphertexts. Then we can evenly divide the
	// number of transforms to get the number of blocked rows.
	cols := int(lenCtIDs)
	rows := int(lenTransformIDs) / cols

	// Create a 2d grid of linear transforms
	transformsFlat := CArrayToSlice(transformIDs, lenTransformIDs, convertCIntToInt)
	transforms := make([][]lintrans.LinearTransformation, rows)
	for i := range transforms {
		row := make([]lintrans.LinearTransformation, cols)
		for j := range cols {
			row[j] = RetrieveLinearTransform(transformsFlat[i*cols+j])
		}
		transforms[i] = row
	}

	// Then we'll get our input ciphertexts
	ctInIDs := CArrayToSlice(ctIDs, lenCtIDs, convertCIntToInt)
	ctsIn := make([]*rlwe.Ciphertext, len(ctInIDs))
	for i, id := range ctInIDs {
		ctsIn[i] = RetrieveCiphertext(id)
	}

	// -------------------- //
	// Main evaluation loop //
	// -------------------- //

	ctsOut := make([]*rlwe.Ciphertext, rows)

	// Here, we'll iterate row by row over the blocked matrix. The result of
	// each block's matrix-vector product will be accumulated to produce the
	// final output row's vector.
	for i := range rows {
		for j := range cols {

			currLT := transforms[i][j]

			// If the IO mode isn't `none`, then our diagonals and rotation keys
			// for this column of blocks have been saved to disk. We'll need to
			// load them in before evaluating this column of transforms.
			if ioMode != "none" {
				LoadDiagonals(&currLT, i, j, moduleName, diagsPath)
				keys := LoadRotationKeys(&currLT, keysPath)

				// In this case, we'll also need to update our linear transform
				// evaluator to have the required rotation keys. If ioMode == `none`,
				// then
				scheme.LinEvaluator = lintrans.NewEvaluator(scheme.Evaluator.WithKey(
					rlwe.NewMemEvaluationKeySet(scheme.RelinKey, keys...),
				))
			}

			// Now we can perform the linear transform.
			ctPartial, err := scheme.LinEvaluator.EvaluateNew(ctsIn[j], currLT)
			if err != nil {
				panic(err)
			}

			// And accumulate the partials.
			if j == 0 {
				ctsOut[i] = ctPartial
			} else {
				ctsOut[i], err = scheme.Evaluator.AddNew(ctsOut[i], ctPartial)
				if err != nil {
					panic(err)
				}
			}

			// Finally, we'll do some clean up. If the IO mode isn't `None`, then
			// we don't want to keep these rotation keys/diagonals in memory.
			// They'll still be in disk though. This is "potentially" non-optimal
			// since keys may be reused across columns, but it reduces overall
			// memory consumption, which I feel is preferable here.
			if ioMode != "none" {
				RemoveDiagonals(&currLT)
				RemoveRotationKeys()
			}
		}
	}

	// We've delayed rescaling until now for efficiency. This means an entire
	// (potentially blocked) linear transform requires only one `Rescale`
	// operation. Kinda neat.
	for r := range rows {
		if err := scheme.Evaluator.Rescale(ctsOut[r], ctsOut[r]); err != nil {
			panic(err)
		}
	}

	ctOutIDs := make([]int, rows)
	for i := range ctOutIDs {
		id := PushCiphertext(ctsOut[i])
		ctOutIDs[i] = id
	}

	arrPtr, length := SliceToCArray(ctOutIDs, convertIntToCInt)
	return arrPtr, length
}

func SaveDiagonals(
	linTransf lintrans.LinearTransformation,
	diagsPath string,
	moduleName string,
	blockRow int,
	blockCol int,
) {
	// Open HDF5 file
	file, err := hdf5.OpenFile(diagsPath, hdf5.F_ACC_RDWR)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Open module's group (conv1, etc.)
	moduleGroup, err := file.OpenGroup(moduleName)
	if err != nil {
		panic(err)
	}
	defer moduleGroup.Close()

	// Open plaintext's group (used to store plaintext diagonals)
	plaintextsGroup, err := moduleGroup.OpenGroup("plaintexts")
	if err != nil {
		panic(err)
	}
	defer plaintextsGroup.Close()

	// Create a similar row/col block group as diagonals in Python
	blockIdx := fmt.Sprintf("%d_%d", blockRow, blockCol)
	blockGroup, err := plaintextsGroup.CreateGroup(blockIdx)
	if err != nil {
		panic(err)
	}
	defer blockGroup.Close()

	// Store each diagonal's serialized plaintext
	for diag, vec := range linTransf.Vec {
		data, err := vec.MarshalBinary()
		if err != nil {
			panic(err)
		}
		datasetName := strconv.Itoa(diag)

		// Create dataspace
		dataspace, err := hdf5.CreateSimpleDataspace([]uint{uint(len(data))}, nil)
		if err != nil {
			panic(err)
		}
		defer dataspace.Close()

		// Create dataset
		dset, err := blockGroup.CreateDataset(
			datasetName, hdf5.T_NATIVE_UINT8, dataspace)
		if err != nil {
			panic(err)
		}
		defer dset.Close()

		// Write the binary data to the dataset
		if err := dset.Write(&data); err != nil {
			panic(err)
		}
	}
}

func SaveRotationKeys(
	galEls []uint64,
	galElKeys []*rlwe.GaloisKey,
	keysPath string,
) {
	// Check if the file exists
	var file *hdf5.File

	if _, err := os.Stat(keysPath); os.IsNotExist(err) {
		// If the file does not exist, create it
		file, err = hdf5.CreateFile(keysPath, hdf5.F_ACC_TRUNC)
		if err != nil {
			panic(err)
		}
	} else {
		// If the file exists, open it in read-write mode
		file, err = hdf5.OpenFile(keysPath, hdf5.F_ACC_RDWR)
		if err != nil {
			panic(err)
		}
	}

	defer file.Close()

	// Iterate over the Galois keys and store them
	for i, key := range galElKeys {
		// Convert uint64 to string for the dataset name
		datasetName := strconv.FormatUint(galEls[i], 10)

		// Skip if the dataset already exists
		if file.LinkExists(datasetName) {
			continue
		}

		// Marshal the key to binary
		data, err := key.MarshalBinary()
		if err != nil {
			panic(err)
		}

		// Create a dataspace for the binary data
		dataspace, err := hdf5.CreateSimpleDataspace([]uint{uint(len(data))}, nil)
		if err != nil {
			panic(err)
		}

		// Create a dataset for the key
		dset, err := file.CreateDataset(datasetName, hdf5.T_NATIVE_UINT8, dataspace)
		if err != nil {
			dataspace.Close()
			panic(err)
		}

		// Write the binary data to the dataset
		if err := dset.Write(&data); err != nil {
			dset.Close()
			dataspace.Close()
			panic(err)
		}

		dset.Close()
		dataspace.Close()
	}
}

func LoadDiagonals(
	linTransf *lintrans.LinearTransformation,
	rowIdx int,
	colIdx int,
	moduleName string,
	diagsPath string,
) {

	// Open the hdf5 file
	file, err := hdf5.OpenFile(diagsPath, hdf5.F_ACC_RDONLY)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Open module's group (conv1, etc.)
	moduleGroup, err := file.OpenGroup(moduleName)
	if err != nil {
		panic(err)
	}
	defer moduleGroup.Close()

	// Open plaintext's group
	plaintextsGroup, err := moduleGroup.OpenGroup("plaintexts")
	if err != nil {
		panic(err)
	}
	defer plaintextsGroup.Close()

	blockIdx := fmt.Sprintf("%d_%d", rowIdx, colIdx)

	blockGroup, err := plaintextsGroup.OpenGroup(blockIdx)
	if err != nil {
		panic(err)
	}
	defer blockGroup.Close()

	// Iterate over linTransf.Vec to load corresponding diagonals
	for diag := range (*linTransf).Vec {

		// Try to open the dataset for this transform
		datasetName := strconv.Itoa(diag)
		dset, err := blockGroup.OpenDataset(datasetName)
		if err != nil {
			panic(err)
		}
		defer dset.Close()

		// Get the dataspace and its size
		space := dset.Space()
		defer space.Close()

		// Get the total number of elements in the dataset
		nElems := space.SimpleExtentNPoints()

		// Allocate a buffer with the correct size
		data := make([]byte, nElems)

		// Read the binary data from the dataset
		if err := dset.Read(&data); err != nil {
			panic(err)
		}

		// Unmarshal the binary data back into its plaintext
		var poly ringqp.Poly
		if err := poly.UnmarshalBinary(data); err != nil {
			panic(err)
		}

		// Store the plaintext back in the LT
		(*linTransf).Vec[diag] = poly
	}
}

func LoadRotationKeys(
	linTransf *lintrans.LinearTransformation,
	keysPath string,
) []*rlwe.GaloisKey {

	// Open the hdf5 file
	file, err := hdf5.OpenFile(keysPath, hdf5.F_ACC_RDONLY)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	var allKeys []*rlwe.GaloisKey
	currRots := (*linTransf).GaloisElements(scheme.Params)

	for _, rot := range currRots {
		datasetName := strconv.FormatUint(rot, 10)

		// Try to open the dataset with name matching the key
		dset, err := file.OpenDataset(datasetName)
		if err != nil {
			panic(err)
		}
		defer dset.Close()

		// Get the dataspace and its size
		space := dset.Space()
		defer space.Close()

		// Get the total number of elements in the dataset
		nElems := space.SimpleExtentNPoints()

		// Allocate a buffer with the correct size
		data := make([]byte, nElems)

		// Read the binary data from the dataset
		if err := dset.Read(&data); err != nil {
			panic(err)
		}

		// Unmarshal the binary data back into the key struct
		var key rlwe.GaloisKey
		if err := key.UnmarshalBinary(data); err != nil {
			panic(err)
		}
		allKeys = append(allKeys, &key)
	}
	return allKeys
}

func RemoveDiagonals(linTransf *lintrans.LinearTransformation) {
	for diag := range (*linTransf).Vec {
		(*linTransf).Vec[diag] = ringqp.Poly{}
	}
}

func RemoveRotationKeys() {
	// We'll just update the linear transform evaluator to no loner have
	// access to the Galois keys it had before. GC should do the rest.
	scheme.LinEvaluator = lintrans.NewEvaluator(scheme.Evaluator.WithKey(
		rlwe.NewMemEvaluationKeySet(scheme.RelinKey),
	))
}
