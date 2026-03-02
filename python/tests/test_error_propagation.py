"""Verify Go errors surface as Python exceptions with messages."""

import pytest

from lattigo import ffi as lattigo_ffi
from lattigo.ckks import Parameters


class TestErrorPropagation:
    """Verify that Go errors cross the FFI boundary as Python RuntimeError exceptions."""

    def test_invalid_params_raises_runtime_error(self):
        """Invalid CKKS parameters should raise RuntimeError, not crash."""
        bad_json = '{"logn": 3, "logq": [10], "logp": [10], "logscale": 5, "h": 64, "ring_type": "standard"}'
        with pytest.raises(RuntimeError):
            Parameters.from_json(bad_json)

    def test_malformed_json_raises_runtime_error(self):
        """Malformed JSON should raise RuntimeError with a message."""
        with pytest.raises(RuntimeError):
            Parameters.from_json("{not valid json}")

    def test_bad_ciphertext_unmarshal_raises(self):
        """Invalid ciphertext bytes should raise RuntimeError."""
        with pytest.raises(RuntimeError):
            lattigo_ffi.rlwe_ciphertext_unmarshal(b"not a valid ciphertext at all!!")

    def test_error_message_is_meaningful(self):
        """Error messages should contain useful diagnostic info."""
        try:
            Parameters.from_json("{}")
        except RuntimeError as e:
            msg = str(e)
            assert len(msg) > 0, "Error message should not be empty"
        else:
            pytest.fail("Expected RuntimeError")
