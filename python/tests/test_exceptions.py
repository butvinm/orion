"""Tests for library-specific exception hierarchies."""

import pytest
from lattigo.errors import FFIError, HandleClosedError, LatticeError
from orion_compiler.errors import CompilationError, CompilerError, ValidationError
from orion_compiler.params import CKKSParams
from orion_evaluator.errors import EvaluatorError, ModelLoadError


class TestLatticeExceptionHierarchy:
    """Test lattigo exception hierarchy."""

    def test_lattice_error_is_exception(self):
        assert issubclass(LatticeError, Exception)

    def test_handle_closed_error_is_lattice_error(self):
        assert issubclass(HandleClosedError, LatticeError)

    def test_ffi_error_is_lattice_error(self):
        assert issubclass(FFIError, LatticeError)

    def test_construct_lattice_error(self):
        e = LatticeError("test message")
        assert str(e) == "test message"

    def test_construct_handle_closed_error(self):
        e = HandleClosedError("handle is closed")
        assert str(e) == "handle is closed"

    def test_construct_ffi_error(self):
        e = FFIError("ffi failed")
        assert str(e) == "ffi failed"

    def test_catch_handle_closed_by_base(self):
        with pytest.raises(LatticeError):
            raise HandleClosedError("closed")

    def test_catch_ffi_error_by_base(self):
        with pytest.raises(LatticeError):
            raise FFIError("ffi failed")

    def test_catch_handle_closed_by_specific(self):
        with pytest.raises(HandleClosedError):
            raise HandleClosedError("closed")

    def test_catch_ffi_error_by_specific(self):
        with pytest.raises(FFIError):
            raise FFIError("ffi failed")


class TestCompilerExceptionHierarchy:
    """Test orion-compiler exception hierarchy."""

    def test_compiler_error_is_exception(self):
        assert issubclass(CompilerError, Exception)

    def test_compilation_error_is_compiler_error(self):
        assert issubclass(CompilationError, CompilerError)

    def test_validation_error_is_compiler_error(self):
        assert issubclass(ValidationError, CompilerError)

    def test_construct_compiler_error(self):
        e = CompilerError("test")
        assert str(e) == "test"

    def test_construct_compilation_error(self):
        e = CompilationError("compilation failed")
        assert str(e) == "compilation failed"

    def test_construct_validation_error(self):
        e = ValidationError("invalid param")
        assert str(e) == "invalid param"

    def test_catch_compilation_by_base(self):
        with pytest.raises(CompilerError):
            raise CompilationError("failed")

    def test_catch_validation_by_base(self):
        with pytest.raises(CompilerError):
            raise ValidationError("invalid")

    def test_catch_compilation_by_specific(self):
        with pytest.raises(CompilationError):
            raise CompilationError("failed")

    def test_catch_validation_by_specific(self):
        with pytest.raises(ValidationError):
            raise ValidationError("invalid")


class TestEvaluatorExceptionHierarchy:
    """Test orion-evaluator exception hierarchy."""

    def test_evaluator_error_is_exception(self):
        assert issubclass(EvaluatorError, Exception)

    def test_model_load_error_is_evaluator_error(self):
        assert issubclass(ModelLoadError, EvaluatorError)

    def test_construct_evaluator_error(self):
        e = EvaluatorError("test")
        assert str(e) == "test"

    def test_construct_model_load_error(self):
        e = ModelLoadError("bad model")
        assert str(e) == "bad model"

    def test_catch_model_load_by_base(self):
        with pytest.raises(EvaluatorError):
            raise ModelLoadError("bad model")

    def test_catch_model_load_by_specific(self):
        with pytest.raises(ModelLoadError):
            raise ModelLoadError("bad model")


class TestValidationErrorIntegration:
    """Test that ValidationError is raised by CKKSParams validation."""

    def test_invalid_logn_raises_validation_error(self):
        with pytest.raises(ValidationError, match="logn must be positive"):
            CKKSParams(logn=0, logq=[40], logp=[40], log_default_scale=40)

    def test_empty_logq_raises_validation_error(self):
        with pytest.raises(ValidationError, match="logq must be non-empty"):
            CKKSParams(logn=13, logq=[], logp=[40], log_default_scale=40)

    def test_invalid_ring_type_raises_validation_error(self):
        with pytest.raises(ValidationError, match="ring_type must be one of"):
            CKKSParams(logn=13, logq=[40], logp=[40], log_default_scale=40, ring_type="invalid")
