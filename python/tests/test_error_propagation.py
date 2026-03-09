"""Verify Go errors surface as Python exceptions with messages."""

import pytest

from lattigo import ffi as lattigo_ffi
from lattigo.ckks import Parameters


class TestErrorPropagation:
    """Verify that Go errors cross the FFI boundary as Python RuntimeError exceptions."""

    def test_invalid_params_raises_runtime_error(self):
        """Invalid CKKS parameters should raise RuntimeError, not crash."""
        with pytest.raises(RuntimeError):
            Parameters(logn=3, logq=[10], logp=[10], log_default_scale=5, ring_type="standard", h=64)

    def test_invalid_ring_type_raises_runtime_error(self):
        """Invalid ring type should raise RuntimeError from the Go bridge."""
        with pytest.raises(RuntimeError):
            Parameters(logn=13, logq=[40, 40], logp=[40], log_default_scale=40, ring_type="invalid_ring", h=192)

    def test_bad_ciphertext_unmarshal_raises(self):
        """Invalid ciphertext bytes should raise RuntimeError."""
        with pytest.raises(RuntimeError):
            lattigo_ffi.rlwe_ciphertext_unmarshal(b"not a valid ciphertext at all!!")

    def test_error_message_is_meaningful(self):
        """Error messages should contain useful diagnostic info."""
        try:
            Parameters(logn=0, logq=[], logp=[], log_default_scale=0, ring_type="standard")
        except RuntimeError as e:
            msg = str(e)
            assert len(msg) > 0, "Error message should not be empty"
        else:
            pytest.fail("Expected RuntimeError")
