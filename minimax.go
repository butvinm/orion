package orion

import (
	"fmt"
	"math/big"
	"strings"
	"sync"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/minimax"
	"github.com/baahl-nyu/lattigo/v6/utils/bignum"
)

// minimaxCache caches previously computed minimax sign coefficients.
var (
	minimaxCache   = make(map[string][][]float64)
	minimaxCacheMu sync.Mutex
)

// GenerateMinimaxSignCoeffs computes composite minimax polynomial
// coefficients for the sign function approximation.
// Returns a flat slice of float64 coefficients (all polynomials concatenated).
func GenerateMinimaxSignCoeffs(degrees []int, prec uint, logAlpha, logErr int, debug bool) ([]float64, error) {
	cleaned := make([]int, 0, len(degrees))
	for _, d := range degrees {
		if d != 0 {
			cleaned = append(cleaned, d)
		}
	}
	if len(cleaned) == 0 {
		return nil, fmt.Errorf("at least one non-zero degree must be provided")
	}

	// Compute total coefficients count
	total := 0
	for _, d := range cleaned {
		total += d + 1
	}

	// Check cache
	key := minimaxCacheKey(cleaned, prec, logAlpha, logErr)
	minimaxCacheMu.Lock()
	if cached, ok := minimaxCache[key]; ok {
		minimaxCacheMu.Unlock()
		flat := make([]float64, 0, total)
		for _, poly := range cached {
			flat = append(flat, poly...)
		}
		return flat, nil
	}
	minimaxCacheMu.Unlock()

	// Generate coefficients
	coeffs := minimax.GenMinimaxCompositePolynomial(
		prec, logAlpha, logErr, cleaned, bignum.Sign, debug,
	)

	// Scale last polynomial from [-1,1] -> [0,1]:
	// divide by 2, add 0.5 to constant term
	lastIdx := len(cleaned) - 1
	two := big.NewFloat(2)
	for i := range coeffs[lastIdx] {
		coeffs[lastIdx][i].Quo(coeffs[lastIdx][i], two)
	}
	coeffs[lastIdx][0].Add(coeffs[lastIdx][0], big.NewFloat(0.5))

	// Convert to float64 and cache
	float64Coeffs := make([][]float64, len(coeffs))
	flat := make([]float64, 0, total)
	for i, poly := range coeffs {
		float64Coeffs[i] = make([]float64, len(poly))
		for j, c := range poly {
			f, _ := c.Float64()
			float64Coeffs[i][j] = f
			flat = append(flat, f)
		}
	}

	minimaxCacheMu.Lock()
	minimaxCache[key] = float64Coeffs
	minimaxCacheMu.Unlock()

	return flat, nil
}

func minimaxCacheKey(degrees []int, prec uint, logAlpha, logErr int) string {
	parts := make([]string, len(degrees))
	for i, d := range degrees {
		parts[i] = fmt.Sprintf("%d", d)
	}
	return fmt.Sprintf("%s|%d|%d|%d", strings.Join(parts, ","), prec, logAlpha, logErr)
}
