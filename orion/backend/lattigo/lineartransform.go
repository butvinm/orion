package main

import (
	"C"
	"encoding/binary"
	"math"
	"unsafe"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/lintrans"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/ring/ringqp"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

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

//export NewLinearTransformEvaluator
func NewLinearTransformEvaluator() {
	scheme.LinEvaluator = lintrans.NewEvaluator(
		ckks.NewEvaluator(*scheme.Params, scheme.EvalKeys))
}

//export GenerateLinearTransform
func GenerateLinearTransform(
	diagIdxsC *C.int, diagIdxsLen C.int,
	diagDataC *C.float, diagDataLen C.int,
	level C.int,
	bsgsRatio C.float,
) C.int {
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

	if err := lintrans.Encode(scheme.Encoder, diagonals, lt); err != nil {
		panic(err)
	}

	// Return reference to linear transform object we just created
	ltID := ltHeap.Add(lt)
	return C.int(ltID)
}

//export EvaluateLinearTransform
func EvaluateLinearTransform(transformID, ctxtID C.int) C.int {
	transform := RetrieveLinearTransform(int(transformID))
	ctIn := RetrieveCiphertext(int(ctxtID))

	// Update the linear transform evaluator to have the most
	// recent set of rotation keys.
	scheme.LinEvaluator = lintrans.NewEvaluator(
		scheme.Evaluator.WithKey(scheme.EvalKeys),
	)

	ctOut, err := scheme.LinEvaluator.EvaluateNew(ctIn, transform)
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export GetLinearTransformRotationKeys
func GetLinearTransformRotationKeys(transformID C.int) (*C.int, C.ulong) {
	transform := RetrieveLinearTransform(int(transformID))
	galEls := transform.GaloisElements(scheme.Params)

	arrPtr, length := SliceToCArray(galEls, convertULongtoInt)
	return arrPtr, length
}

//export GenerateLinearTransformRotationKey
func GenerateLinearTransformRotationKey(galEl C.ulong) {
	rotKey := scheme.KeyGen.GenGaloisKeyNew(uint64(galEl), scheme.SecretKey)
	scheme.EvalKeys.GaloisKeys[uint64(galEl)] = rotKey
}

//export GenerateAndSerializeRotationKey
func GenerateAndSerializeRotationKey(galEl C.ulong) (*C.char, C.ulong) {
	rotKey := scheme.KeyGen.GenGaloisKeyNew(uint64(galEl), scheme.SecretKey)
	data, err := rotKey.MarshalBinary() // Marshal the key to binary
	if err != nil {
		panic(err)
	}

	arrPtr, length := SliceToCArray(data, convertByteToCChar)
	return arrPtr, length
}

//export LoadRotationKey
func LoadRotationKey(
	dataPtr *C.char, lenData C.ulong,
	galEl C.ulong,
) {
	rotKeySerial := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))

	// Unmarshal the binary data into a GaloisKey
	var rotKey rlwe.GaloisKey
	if err := rotKey.UnmarshalBinary(rotKeySerial); err != nil {
		panic(err)
	}

	// Update our global map of evaluation keys to include what
	// we just loaded. This will eventually get used by the
	// current linear transform and then deleted from RAM.
	scheme.EvalKeys.GaloisKeys[uint64(galEl)] = &rotKey
}

//export SerializeDiagonal
func SerializeDiagonal(transformID, diagIdx C.int) (*C.char, C.ulong) {
	transform := RetrieveLinearTransform(int(transformID))
	diag := transform.Vec[int(diagIdx)]

	data, err := diag.MarshalBinary() // Marshal the diag to binary
	if err != nil {
		panic(err)
	}

	// Since it will be saved to disk, we can delete it from our
	// linear transform object and load it in dynamically at runtime
	transform.Vec[int(diagIdx)] = ringqp.Poly{}

	arrPtr, length := SliceToCArray(data, convertByteToCChar)
	return arrPtr, length
}

//export LoadPlaintextDiagonal
func LoadPlaintextDiagonal(
	dataPtr *C.char, lenData C.ulong,
	transformID C.int,
	diagIdx C.ulong,
) {
	transform := RetrieveLinearTransform(int(transformID))
	diagSerial := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))

	var poly ringqp.Poly
	if err := poly.UnmarshalBinary(diagSerial); err != nil {
		panic(err)
	}
	transform.Vec[int(diagIdx)] = poly
}

//export SerializeLinearTransform
func SerializeLinearTransform(transformID C.int) (*C.char, C.ulong) {
	lt := RetrieveLinearTransform(int(transformID))

	// Serialize MetaData
	metaBytes, err := lt.MetaData.MarshalBinary()
	if err != nil {
		panic(err)
	}

	// Estimate buffer size: metadata + 4 int64s + vec entries
	buf := make([]byte, 0, len(metaBytes)+256)

	// Write MetaData: [8 bytes length][data]
	tmp := make([]byte, 8)
	binary.LittleEndian.PutUint64(tmp, uint64(len(metaBytes)))
	buf = append(buf, tmp...)
	buf = append(buf, metaBytes...)

	// Write int fields as int64
	binary.LittleEndian.PutUint64(tmp, uint64(lt.LogBabyStepGiantStepRatio))
	buf = append(buf, tmp...)
	binary.LittleEndian.PutUint64(tmp, uint64(lt.N1))
	buf = append(buf, tmp...)
	binary.LittleEndian.PutUint64(tmp, uint64(lt.LevelQ))
	buf = append(buf, tmp...)
	binary.LittleEndian.PutUint64(tmp, uint64(lt.LevelP))
	buf = append(buf, tmp...)

	// Write Vec map: [8 bytes count] then for each entry [8 bytes diagIndex][8 bytes polyLen][polyData]
	binary.LittleEndian.PutUint64(tmp, uint64(len(lt.Vec)))
	buf = append(buf, tmp...)

	for diagIdx, poly := range lt.Vec {
		// Diagonal index (signed -> uint64 via two's complement)
		binary.LittleEndian.PutUint64(tmp, uint64(int64(diagIdx)))
		buf = append(buf, tmp...)

		polyBytes, err := poly.MarshalBinary()
		if err != nil {
			panic(err)
		}
		binary.LittleEndian.PutUint64(tmp, uint64(len(polyBytes)))
		buf = append(buf, tmp...)
		buf = append(buf, polyBytes...)
	}

	arrPtr, length := SliceToCArray(buf, convertByteToCChar)
	return arrPtr, length
}

//export LoadLinearTransform
func LoadLinearTransform(dataPtr *C.char, lenData C.ulong) C.int {
	data := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))
	off := 0

	// Read MetaData
	metaLen := int(binary.LittleEndian.Uint64(data[off : off+8]))
	off += 8
	var meta rlwe.MetaData
	if err := meta.UnmarshalBinary(data[off : off+metaLen]); err != nil {
		panic(err)
	}
	off += metaLen

	// Read int fields
	logBSGS := int(int64(binary.LittleEndian.Uint64(data[off : off+8])))
	off += 8
	n1 := int(int64(binary.LittleEndian.Uint64(data[off : off+8])))
	off += 8
	levelQ := int(int64(binary.LittleEndian.Uint64(data[off : off+8])))
	off += 8
	levelP := int(int64(binary.LittleEndian.Uint64(data[off : off+8])))
	off += 8

	// Read Vec map
	vecLen := int(binary.LittleEndian.Uint64(data[off : off+8]))
	off += 8

	vec := make(map[int]ringqp.Poly, vecLen)
	for i := 0; i < vecLen; i++ {
		diagIdx := int(int64(binary.LittleEndian.Uint64(data[off : off+8])))
		off += 8

		polyLen := int(binary.LittleEndian.Uint64(data[off : off+8]))
		off += 8

		var poly ringqp.Poly
		if err := poly.UnmarshalBinary(data[off : off+polyLen]); err != nil {
			panic(err)
		}
		off += polyLen

		vec[diagIdx] = poly
	}

	lt := lintrans.LinearTransformation{
		MetaData:                  &meta,
		LogBabyStepGiantStepRatio: logBSGS,
		N1:                        n1,
		LevelQ:                    levelQ,
		LevelP:                    levelP,
		Vec:                       vec,
	}

	ltID := ltHeap.Add(lt)
	return C.int(ltID)
}

//export RemovePlaintextDiagonals
func RemovePlaintextDiagonals(transformID C.int) {
	linTransf := RetrieveLinearTransform(int(transformID))
	for diag := range linTransf.Vec {
		linTransf.Vec[diag] = ringqp.Poly{}
	}
}

//export RemoveRotationKeys
func RemoveRotationKeys() {
	// We'll just update the linear transform evaluator to no longer have
	// access to the Galois keys it had before. GC should do the rest.
	scheme.EvalKeys = rlwe.NewMemEvaluationKeySet(scheme.RelinKey)
	scheme.LinEvaluator = lintrans.NewEvaluator(scheme.Evaluator.WithKey(
		scheme.EvalKeys,
	))
}
