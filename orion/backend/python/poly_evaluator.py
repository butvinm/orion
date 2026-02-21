import torch
import numpy as np


class PolynomialGenerator:
    """Compile-time polynomial generation. No Go PolyEvaluator needed.

    Standalone Go functions (GenerateChebyshev, GenerateMonomial,
    GenerateMinimaxSignCoeffs) work without NewPolynomialEvaluator().
    Validated in Experiment 6 test 4.
    """

    def __init__(self, backend):
        self.backend = backend

    def generate_monomial(self, coeffs):
        if isinstance(coeffs, (torch.Tensor, np.ndarray)):
            coeffs = coeffs.tolist()
        return self.backend.GenerateMonomial(coeffs[::-1])

    def generate_chebyshev(self, coeffs):
        if isinstance(coeffs, (torch.Tensor, np.ndarray)):
            coeffs = coeffs.tolist()
        return self.backend.GenerateChebyshev(coeffs)

    def generate_minimax_sign_coeffs(self, degrees, prec=128, logalpha=12,
                                     logerr=12, debug=False):
        if isinstance(degrees, int):
            degrees = [degrees]
        else:
            degrees = list(degrees)

        degrees = [d for d in degrees if d != 0]
        if len(degrees) == 0:
            raise ValueError(
                "At least one non-zero degree polynomial must be provided to "
                "generate_minimax_sign_coeffs(). "
            )

        coeffs_flat = self.backend.GenerateMinimaxSignCoeffs(
            degrees, prec, logalpha, logerr, int(debug)
        )

        coeffs_flat = torch.tensor(coeffs_flat)
        splits = [degree + 1 for degree in degrees]
        return torch.split(coeffs_flat, splits)

    def get_depth(self, poly):
        return self.backend.GetPolyDepth(poly)


class PolynomialEvaluator(PolynomialGenerator):
    """Inference-time polynomial evaluation. Needs Go PolyEvaluator.

    Constructor calls NewPolynomialEvaluator(). Adds evaluate_polynomial()
    which crashes with nil panic at polyeval.go:75 without it.
    """

    def __init__(self, backend, params=None):
        super().__init__(backend)
        self.params = params
        self.backend.NewPolynomialEvaluator()

    def evaluate_polynomial(self, ciphertensor, poly, out_scale=None):
        from orion.backend.python.tensors import CipherTensor

        if out_scale is None:
            if self.params is not None:
                out_scale = self.params.get_default_scale()
            else:
                out_scale = 1 << 40  # fallback default logscale

        cts_out = []
        for ctxt in ciphertensor.ids:
            ct_out = self.backend.EvaluatePolynomial(ctxt, poly, out_scale)
            cts_out.append(ct_out)

        return CipherTensor(
            ciphertensor.context, cts_out, ciphertensor.shape, ciphertensor.on_shape)
