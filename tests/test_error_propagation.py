"""Phase 7 validation: Go errors surface as Python exceptions with messages."""

import pytest

from lattigo import legacy_ffi as ffi


class TestErrorPropagation:
    """Verify that Go errors cross the FFI boundary as Python RuntimeError exceptions."""

    def test_invalid_params_raises_runtime_error(self):
        """Invalid CKKS parameters should raise RuntimeError, not crash."""
        bad_json = '{"logn": 3, "logq": [10], "logp": [10], "logscale": 5, "h": 64, "ring_type": "standard"}'
        with pytest.raises(RuntimeError):
            ffi.new_client(bad_json)

    def test_malformed_json_raises_runtime_error(self):
        """Malformed JSON should raise RuntimeError with a message."""
        with pytest.raises(RuntimeError):
            ffi.new_client("{not valid json}")

    def test_bad_ciphertext_unmarshal_raises(self):
        """Invalid ciphertext bytes should raise RuntimeError."""
        with pytest.raises(RuntimeError):
            ffi.ciphertext_unmarshal(b"not a valid ciphertext at all!!")

    def test_error_message_is_meaningful(self):
        """Error messages should contain useful diagnostic info."""
        try:
            ffi.new_client("{}")
        except RuntimeError as e:
            msg = str(e)
            assert len(msg) > 0, "Error message should not be empty"
        else:
            pytest.fail("Expected RuntimeError")

