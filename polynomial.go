package orion

import (
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// Polynomial wraps a bignum.Polynomial for use with EvalPoly.
type Polynomial struct {
	raw bignum.Polynomial
}

// Raw returns the underlying bignum.Polynomial.
func (p *Polynomial) Raw() bignum.Polynomial {
	return p.raw
}

// GenerateMonomial creates a monomial-basis polynomial from coefficients.
// coeffs[i] is the coefficient of x^i.
func GenerateMonomial(coeffs []float64) *Polynomial {
	poly := bignum.NewPolynomial(bignum.Monomial, coeffs, nil)
	return &Polynomial{raw: poly}
}

// GenerateChebyshev creates a Chebyshev-basis polynomial from coefficients.
// The default interval is [-1, 1].
func GenerateChebyshev(coeffs []float64) *Polynomial {
	poly := bignum.NewPolynomial(bignum.Chebyshev, coeffs, [2]float64{-1.0, 1.0})
	return &Polynomial{raw: poly}
}
