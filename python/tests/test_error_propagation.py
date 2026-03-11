"""Verify Go errors surface as Python exceptions with messages."""

import pytest
from lattigo import ffi as lattigo_ffi
from lattigo.ckks import Parameters
from lattigo.errors import FFIError


class TestErrorPropagation:
    """Verify that Go errors cross the FFI boundary as Python FFIError exceptions."""

    def test_invalid_params_raises_ffi_error(self):
        """Invalid CKKS parameters should raise FFIError, not crash."""
        with pytest.raises(FFIError):
            Parameters(
                logn=3, logq=[10], logp=[10], log_default_scale=5, ring_type="standard", h=64
            )

    def test_invalid_ring_type_raises_ffi_error(self):
        """Invalid ring type should raise FFIError from the Go bridge."""
        with pytest.raises(FFIError):
            Parameters(
                logn=13,
                logq=[40, 40],
                logp=[40],
                log_default_scale=40,
                ring_type="invalid_ring",
                h=192,
            )

    def test_bad_ciphertext_unmarshal_raises(self):
        """Invalid ciphertext bytes should raise FFIError."""
        with pytest.raises(FFIError):
            lattigo_ffi.rlwe_ciphertext_unmarshal(b"not a valid ciphertext at all!!")

    def test_error_message_is_meaningful(self):
        """Error messages should contain useful diagnostic info."""
        try:
            Parameters(logn=0, logq=[], logp=[], log_default_scale=0, ring_type="standard")
        except FFIError as e:
            msg = str(e)
            assert len(msg) > 0, "Error message should not be empty"
        else:
            pytest.fail("Expected FFIError")
