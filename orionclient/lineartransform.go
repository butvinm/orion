package orionclient

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/lintrans"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/ring/ringqp"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

// LinearTransform wraps a Lattigo lintrans.LinearTransformation.
type LinearTransform struct {
	raw lintrans.LinearTransformation
}

// Raw returns the underlying Lattigo linear transformation.
func (lt *LinearTransform) Raw() lintrans.LinearTransformation {
	return lt.raw
}

// GenerateLinearTransform creates a linear transform from diagonal data.
// diagIndices maps diagonal index to float64 values (one per slot).
// level is the multiplicative level; bsgsRatio controls baby-step-giant-step.
func GenerateLinearTransform(p Params, diagIndices map[int][]float64, level int, bsgsRatio float64) (*LinearTransform, error) {
	ckksParams, err := p.NewCKKSParameters()
	if err != nil {
		return nil, fmt.Errorf("creating CKKS parameters: %w", err)
	}

	diagonals := make(lintrans.Diagonals[float64])
	for idx, vals := range diagIndices {
		diagonals[idx] = vals
	}

	ltparams := lintrans.Parameters{
		DiagonalsIndexList:        diagonals.DiagonalsIndexList(),
		LevelQ:                    level,
		LevelP:                    ckksParams.MaxLevelP(),
		Scale:                     rlwe.NewScale(ckksParams.Q()[level]),
		LogDimensions:             ring.Dimensions{Rows: 0, Cols: ckksParams.LogMaxSlots()},
		LogBabyStepGiantStepRatio: int(math.Log(bsgsRatio)),
	}

	enc := ckks.NewEncoder(ckksParams)

	lt := lintrans.NewTransformation(ckksParams, ltparams)
	if err := lintrans.Encode(enc, diagonals, lt); err != nil {
		return nil, fmt.Errorf("encoding linear transform: %w", err)
	}

	return &LinearTransform{raw: lt}, nil
}

// LoadLinearTransform deserializes a linear transform from a blob.
// This is the inverse of Marshal().
func LoadLinearTransform(data []byte) (*LinearTransform, error) {
	if len(data) < 48 { // minimum: 8 (metaLen) + some meta + 4*8 (fields) + 8 (vecLen)
		return nil, fmt.Errorf("linear transform data too short: %d bytes", len(data))
	}

	off := 0

	// Read MetaData
	metaLen := int(binary.LittleEndian.Uint64(data[off : off+8]))
	off += 8
	if off+metaLen > len(data) {
		return nil, fmt.Errorf("unexpected end of data reading metadata")
	}
	var meta rlwe.MetaData
	if err := meta.UnmarshalBinary(data[off : off+metaLen]); err != nil {
		return nil, fmt.Errorf("unmarshalling metadata: %w", err)
	}
	off += metaLen

	if off+40 > len(data) {
		return nil, fmt.Errorf("unexpected end of data reading fields")
	}

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
		if off+16 > len(data) {
			return nil, fmt.Errorf("unexpected end of data reading diagonal %d", i)
		}
		diagIdx := int(int64(binary.LittleEndian.Uint64(data[off : off+8])))
		off += 8
		polyLen := int(binary.LittleEndian.Uint64(data[off : off+8]))
		off += 8

		if off+polyLen > len(data) {
			return nil, fmt.Errorf("unexpected end of data reading diagonal %d poly", i)
		}
		var poly ringqp.Poly
		if err := poly.UnmarshalBinary(data[off : off+polyLen]); err != nil {
			return nil, fmt.Errorf("unmarshalling diagonal %d: %w", i, err)
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

	return &LinearTransform{raw: lt}, nil
}

// Marshal serializes the linear transform to bytes.
// Format matches the backend's SerializeLinearTransform.
func (lt *LinearTransform) Marshal() ([]byte, error) {
	raw := lt.raw

	metaBytes, err := raw.MetaData.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("marshalling metadata: %w", err)
	}

	buf := make([]byte, 0, len(metaBytes)+256)
	tmp := make([]byte, 8)

	// MetaData: [8 bytes length][data]
	binary.LittleEndian.PutUint64(tmp, uint64(len(metaBytes)))
	buf = append(buf, tmp...)
	buf = append(buf, metaBytes...)

	// Int fields
	binary.LittleEndian.PutUint64(tmp, uint64(raw.LogBabyStepGiantStepRatio))
	buf = append(buf, tmp...)
	binary.LittleEndian.PutUint64(tmp, uint64(raw.N1))
	buf = append(buf, tmp...)
	binary.LittleEndian.PutUint64(tmp, uint64(raw.LevelQ))
	buf = append(buf, tmp...)
	binary.LittleEndian.PutUint64(tmp, uint64(raw.LevelP))
	buf = append(buf, tmp...)

	// Vec map
	binary.LittleEndian.PutUint64(tmp, uint64(len(raw.Vec)))
	buf = append(buf, tmp...)

	for diagIdx, poly := range raw.Vec {
		binary.LittleEndian.PutUint64(tmp, uint64(int64(diagIdx)))
		buf = append(buf, tmp...)

		polyBytes, err := poly.MarshalBinary()
		if err != nil {
			return nil, fmt.Errorf("marshalling diagonal %d: %w", diagIdx, err)
		}
		binary.LittleEndian.PutUint64(tmp, uint64(len(polyBytes)))
		buf = append(buf, tmp...)
		buf = append(buf, polyBytes...)
	}

	return buf, nil
}

// RequiredGaloisElements returns the Galois elements needed for evaluation.
func (lt *LinearTransform) RequiredGaloisElements(p Params) ([]uint64, error) {
	ckksParams, err := p.NewCKKSParameters()
	if err != nil {
		return nil, fmt.Errorf("creating CKKS parameters: %w", err)
	}
	return lt.raw.GaloisElements(ckksParams), nil
}
