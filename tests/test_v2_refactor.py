"""Unit tests for Task 3: evaluator type split, context-on-tensor, module refactor.

Tests the new PolynomialGenerator/PolynomialEvaluator, TransformEncoder/TransformEvaluator
split, CipherTensor/PlainTensor context field, and module.compile(context) signature.
"""

import pytest
import types

from orion.backend.python.poly_evaluator import PolynomialGenerator, PolynomialEvaluator
from orion.backend.python.lt_evaluator import TransformEncoder, TransformEvaluator
from orion.backend.python.tensors import CipherTensor, PlainTensor
from orion.nn.module import Module
from orion.nn.operations import Add, Mult, Bootstrap
from orion.nn.activation import Activation, Chebyshev, Sigmoid, _Sign, ReLU
from orion.nn.linear import Linear, Conv2d, LinearTransform
from orion.nn.normalization import BatchNorm1d, BatchNorm2d
from orion.nn.reshape import Flatten
from orion.core.auto_bootstrap import BootstrapPlacer


# -----------------------------------------------------------------------
# 3a: Evaluator type split
# -----------------------------------------------------------------------


class TestPolynomialGeneratorSplit:
    def test_generator_has_generation_methods(self):
        """PolynomialGenerator has generate_monomial, generate_chebyshev, etc."""
        assert hasattr(PolynomialGenerator, 'generate_monomial')
        assert hasattr(PolynomialGenerator, 'generate_chebyshev')
        assert hasattr(PolynomialGenerator, 'generate_minimax_sign_coeffs')
        assert hasattr(PolynomialGenerator, 'get_depth')

    def test_generator_does_not_have_evaluate(self):
        """PolynomialGenerator should NOT have evaluate_polynomial."""
        assert not hasattr(PolynomialGenerator, 'evaluate_polynomial')

    def test_evaluator_inherits_generator(self):
        """PolynomialEvaluator inherits from PolynomialGenerator."""
        assert issubclass(PolynomialEvaluator, PolynomialGenerator)

    def test_evaluator_has_evaluate(self):
        """PolynomialEvaluator has evaluate_polynomial."""
        assert hasattr(PolynomialEvaluator, 'evaluate_polynomial')

    def test_generator_takes_backend_only(self):
        """PolynomialGenerator.__init__ takes only backend, no scheme."""
        import inspect
        sig = inspect.signature(PolynomialGenerator.__init__)
        params = list(sig.parameters.keys())
        assert params == ['self', 'backend']

    def test_evaluator_takes_backend_only(self):
        """PolynomialEvaluator.__init__ takes only backend, no scheme."""
        import inspect
        sig = inspect.signature(PolynomialEvaluator.__init__)
        params = list(sig.parameters.keys())
        assert params == ['self', 'backend']


class TestTransformEncoderSplit:
    def test_encoder_has_encode_methods(self):
        """TransformEncoder has generate_transforms, get_galois_elements."""
        assert hasattr(TransformEncoder, 'generate_transforms')
        assert hasattr(TransformEncoder, 'get_galois_elements')

    def test_encoder_does_not_have_evaluate(self):
        """TransformEncoder should NOT have evaluate_transforms."""
        assert not hasattr(TransformEncoder, 'evaluate_transforms')

    def test_encoder_tracks_galois_elements(self):
        """TransformEncoder has required_galois_elements attribute pattern."""
        import inspect
        source = inspect.getsource(TransformEncoder.__init__)
        assert 'required_galois_elements' in source

    def test_evaluator_has_evaluate(self):
        """TransformEvaluator has evaluate_transforms."""
        assert hasattr(TransformEvaluator, 'evaluate_transforms')

    def test_evaluator_does_not_have_encode(self):
        """TransformEvaluator should NOT have generate_transforms."""
        assert not hasattr(TransformEvaluator, 'generate_transforms')

    def test_encoder_takes_backend_params(self):
        """TransformEncoder.__init__ takes backend and params."""
        import inspect
        sig = inspect.signature(TransformEncoder.__init__)
        params = list(sig.parameters.keys())
        assert params == ['self', 'backend', 'params']

    def test_evaluator_takes_backend_evaluator(self):
        """TransformEvaluator.__init__ takes backend and evaluator."""
        import inspect
        sig = inspect.signature(TransformEvaluator.__init__)
        params = list(sig.parameters.keys())
        assert params == ['self', 'backend', 'evaluator']


# -----------------------------------------------------------------------
# 3b: Context-on-tensor
# -----------------------------------------------------------------------


class TestContextOnTensor:
    def _make_context(self):
        """Create a minimal context namespace."""
        ctx = types.SimpleNamespace(
            backend=types.SimpleNamespace(
                DeletePlaintext=lambda _: None,
                DeleteCiphertext=lambda _: None,
                GetPlaintextScale=lambda _: 1.0,
                GetCiphertextScale=lambda _: 1.0,
                GetCiphertextLevel=lambda _: 5,
                GetPlaintextLevel=lambda _: 5,
            ),
            encoder=types.SimpleNamespace(),
            encryptor=types.SimpleNamespace(),
            evaluator=types.SimpleNamespace(),
            bootstrapper=types.SimpleNamespace(),
            params=types.SimpleNamespace(get_debug_status=lambda: False),
        )
        return ctx

    def test_plaintensor_has_context(self):
        """PlainTensor stores context as .context field."""
        ctx = self._make_context()
        pt = PlainTensor(ctx, [1, 2], (2,))
        assert pt.context is ctx

    def test_ciphertensor_has_context(self):
        """CipherTensor stores context as .context field."""
        ctx = self._make_context()
        ct = CipherTensor(ctx, [1, 2], (2,))
        assert ct.context is ctx

    def test_plaintensor_backward_compat_scheme(self):
        """PlainTensor.scheme returns context (backward compat)."""
        ctx = self._make_context()
        pt = PlainTensor(ctx, [1], (1,))
        assert pt.scheme is ctx

    def test_ciphertensor_backward_compat_scheme(self):
        """CipherTensor.scheme returns context (backward compat)."""
        ctx = self._make_context()
        ct = CipherTensor(ctx, [1], (1,))
        assert ct.scheme is ctx

    def test_plaintensor_derives_backend(self):
        """PlainTensor derives backend from context."""
        ctx = self._make_context()
        pt = PlainTensor(ctx, [1], (1,))
        assert pt.backend is ctx.backend

    def test_ciphertensor_derives_evaluator(self):
        """CipherTensor derives evaluator from context."""
        ctx = self._make_context()
        ct = CipherTensor(ctx, [1], (1,))
        assert ct.evaluator is ctx.evaluator
        assert ct.encryptor is ctx.encryptor
        assert ct.bootstrapper is ctx.bootstrapper

    def test_no_scheme_as_first_param_name(self):
        """CipherTensor/PlainTensor __init__ first param is 'context', not 'scheme'."""
        import inspect
        ct_sig = inspect.signature(CipherTensor.__init__)
        pt_sig = inspect.signature(PlainTensor.__init__)
        ct_params = list(ct_sig.parameters.keys())
        pt_params = list(pt_sig.parameters.keys())
        assert ct_params[1] == 'context'
        assert pt_params[1] == 'context'


# -----------------------------------------------------------------------
# 3c: Module refactor
# -----------------------------------------------------------------------


class TestModuleRefactor:
    def test_module_no_scheme_class_var(self):
        """Module class variable 'scheme' is removed."""
        assert not hasattr(Module, 'scheme')

    def test_module_no_margin_class_var(self):
        """Module class variable 'margin' is removed."""
        assert not hasattr(Module, 'margin')

    def test_module_no_set_scheme(self):
        """Module.set_scheme static method is removed."""
        assert not hasattr(Module, 'set_scheme')

    def test_module_no_set_margin(self):
        """Module.set_margin static method is removed."""
        assert not hasattr(Module, 'set_margin')

    def test_linear_compile_takes_context(self):
        """Linear.compile(context) accepts context parameter."""
        import inspect
        sig = inspect.signature(Linear.compile)
        params = list(sig.parameters.keys())
        assert 'context' in params

    def test_conv2d_compile_takes_context(self):
        """Conv2d.compile(context) accepts context parameter."""
        import inspect
        sig = inspect.signature(Conv2d.compile)
        params = list(sig.parameters.keys())
        assert 'context' in params

    def test_activation_compile_takes_context(self):
        """Activation.compile(context) accepts context parameter."""
        import inspect
        sig = inspect.signature(Activation.compile)
        params = list(sig.parameters.keys())
        assert 'context' in params

    def test_chebyshev_compile_takes_context(self):
        """Chebyshev.compile(context) accepts context parameter."""
        import inspect
        sig = inspect.signature(Chebyshev.compile)
        params = list(sig.parameters.keys())
        assert 'context' in params

    def test_bootstrap_compile_takes_context(self):
        """Bootstrap.compile(context) accepts context parameter."""
        import inspect
        sig = inspect.signature(Bootstrap.compile)
        params = list(sig.parameters.keys())
        assert 'context' in params

    def test_batchnorm1d_compile_takes_context(self):
        """BatchNorm1d.compile(context) accepts context parameter."""
        import inspect
        sig = inspect.signature(BatchNorm1d.compile)
        params = list(sig.parameters.keys())
        assert 'context' in params

    def test_batchnorm2d_compile_takes_context(self):
        """BatchNorm2d.compile(context) accepts context parameter."""
        import inspect
        sig = inspect.signature(BatchNorm2d.compile)
        params = list(sig.parameters.keys())
        assert 'context' in params


# -----------------------------------------------------------------------
# 3d: fit(context) signature
# -----------------------------------------------------------------------


class TestFitContext:
    def test_chebyshev_fit_takes_context(self):
        """Chebyshev.fit(context) accepts context parameter."""
        import inspect
        sig = inspect.signature(Chebyshev.fit)
        params = list(sig.parameters.keys())
        assert 'context' in params

    def test_sign_fit_takes_context(self):
        """_Sign.fit(context) accepts context parameter."""
        import inspect
        sig = inspect.signature(_Sign.fit)
        params = list(sig.parameters.keys())
        assert 'context' in params

    def test_relu_fit_takes_context(self):
        """ReLU.fit(context) accepts context parameter."""
        import inspect
        sig = inspect.signature(ReLU.fit)
        params = list(sig.parameters.keys())
        assert 'context' in params

    def test_bootstrap_fit_takes_context(self):
        """Bootstrap.fit(context) accepts context parameter."""
        import inspect
        sig = inspect.signature(Bootstrap.fit)
        params = list(sig.parameters.keys())
        assert 'context' in params


# -----------------------------------------------------------------------
# 3d: BootstrapPlacer accepts context
# -----------------------------------------------------------------------


class TestBootstrapPlacerContext:
    def test_placer_takes_context(self):
        """BootstrapPlacer.__init__ takes net, network_dag, context."""
        import inspect
        sig = inspect.signature(BootstrapPlacer.__init__)
        params = list(sig.parameters.keys())
        assert params == ['self', 'net', 'network_dag', 'context']


