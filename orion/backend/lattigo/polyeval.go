package main

import (
	"C"

	"fmt"
	"math/big"
	"strings"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/minimax"
	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/polynomial"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
	"github.com/baahl-nyu/lattigo/v6/utils/bignum"
)

var polyHeap = NewHeapAllocator()
var minimaxSignMap = make(map[string][][]float64)

func AddPoly(poly bignum.Polynomial) int {
	return polyHeap.Add(poly)
}

func RetrievePoly(polyID int) bignum.Polynomial {
	return polyHeap.Retrieve(polyID).(bignum.Polynomial)
}

func DeletePoly(polyID int) {
	polyHeap.Delete(polyID)
}

//export NewPolynomialEvaluator
func NewPolynomialEvaluator() {
	scheme.PolyEvaluator = polynomial.NewEvaluator(*scheme.Params, scheme.Evaluator)
}

//export GenerateMonomial
func GenerateMonomial(
	coeffsPtr *C.float,
	lenCoeffs C.int,
) C.int {
	coeffs := CArrayToSlice(coeffsPtr, lenCoeffs, convertCFloatToFloat)
	poly := bignum.NewPolynomial(bignum.Monomial, coeffs, nil)

	idx := AddPoly(poly)
	return C.int(idx)
}

//export GenerateChebyshev
func GenerateChebyshev(
	coeffsPtr *C.float,
	lenCoeffs C.int,
) C.int {
	coeffs := CArrayToSlice(coeffsPtr, lenCoeffs, convertCFloatToFloat)
	poly := bignum.NewPolynomial(
		bignum.Chebyshev, coeffs, [2]float64{-1.0, 1.0})

	idx := AddPoly(poly)
	return C.int(idx)
}

//export EvaluatePolynomial
func EvaluatePolynomial(
	ctInID C.int,
	polyID C.int,
	outScale C.ulong,
) C.int {
	poly := RetrievePoly(int(polyID))
	ctIn := RetrieveCiphertext(int(ctInID))

	// Often times we'll want to keep the original input ciphertext unchanged.
	ctTmp := ckks.NewCiphertext(*scheme.Params, 1, ctIn.Level())
	ctTmp.Copy(ctIn)

	res, err := scheme.PolyEvaluator.Evaluate(
		ctTmp, poly, rlwe.NewScale(uint64(outScale)),
	)
	if err != nil {
		panic(err)
	}

	ctOutID := PushCiphertext(res)
	return C.int(ctOutID)
}

// ------------------------------ //
//  Minimax Sign Helper Functions //
// ------------------------------ //

//export GenerateMinimaxSignCoeffs
func GenerateMinimaxSignCoeffs(
	degreesPtr *C.int, lenDegrees C.int,
	prec C.int,
	logalpha C.int,
	logerr C.int,
	debug C.int,
) (*C.double, C.ulong) {
	degrees := CArrayToSlice(degreesPtr, lenDegrees, convertCIntToInt)

	// We'll eventually return this flattened list of coefficients
	sumDegrees := 0
	for _, d := range degrees {
		sumDegrees += d + 1
	}
	flatCoeffs := make([]float64, sumDegrees)

	// Generate key for given minimax sign parameters
	key := GenerateUniqueKey(
		degrees,
		uint(prec),
		int(logalpha),
		int(logerr),
	)

	// Check if coefficients already exist in the map
	if existingCoeffs, exists := minimaxSignMap[key]; exists {
		// If so, avoid generating them and instead return these
		idx := 0
		for _, poly := range existingCoeffs {
			for _, coeff := range poly {
				flatCoeffs[idx] = coeff
				idx++
			}
		}
	} else {
		// Otherwise, generate new coefficients
		coeffs := minimax.GenMinimaxCompositePolynomial(
			uint(prec),
			int(logalpha),
			int(logerr),
			degrees,
			bignum.Sign,
			int(debug) != 0,
		)

		// Divide last poly by 2 to scale from [-1,1] -> [-0.5, 0.5]
		for i := range coeffs[len(degrees)-1] {
			coeffs[len(degrees)-1][i].Quo(coeffs[len(degrees)-1][i], big.NewFloat(2))
		}

		// Add 0.5 to last polynomial so sign outputs in range [0, 1]
		coeffs[len(degrees)-1][0] = coeffs[len(degrees)-1][0].Add(
			coeffs[len(degrees)-1][0], big.NewFloat(0.5))

		// Create 2D array of float64 to store in map
		float64Coeffs := make([][]float64, len(coeffs))
		for i := range coeffs {
			float64Coeffs[i] = make([]float64, len(coeffs[i]))
		}

		idx := 0
		for i, poly := range coeffs {
			for j, coeff := range poly {
				f64, _ := coeff.Float64()
				flatCoeffs[idx] = f64
				float64Coeffs[i][j] = f64
				idx++
			}
		}

		// Store coefficients in the map for future use
		minimaxSignMap[key] = float64Coeffs
	}

	arrPtr, arrLen := SliceToCArray(flatCoeffs, convertFloat64ToCDouble)
	return arrPtr, arrLen
}

// Create a unique string from the minimax parameters to use as an
// index for the sign map.
func GenerateUniqueKey(
	degrees []int,
	prec uint,
	logAlpha int,
	logErr int,
) string {
	degreesStr := make([]string, len(degrees))
	for i, deg := range degrees {
		degreesStr[i] = fmt.Sprintf("%d", deg)
	}

	// Create a composite key
	return fmt.Sprintf("%s|%d|%d|%d",
		strings.Join(degreesStr, ","),
		prec,
		logAlpha,
		logErr)
}

func DeleteMinimaxSignMap() {
	minimaxSignMap = make(map[string][][]float64)
}
