"""Tests for model.bin: verify params, manifest, input_level match expected values."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import orion

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.bin")


@pytest.fixture(scope="module")
def compiled():
    """Load model.bin once for all tests."""
    assert os.path.exists(MODEL_PATH), f"model.bin not found at {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        data = f.read()
    return orion.CompiledModel.from_bytes(data)


def test_model_bin_exists():
    assert os.path.exists(MODEL_PATH)
    assert os.path.getsize(MODEL_PATH) > 0


def test_params(compiled):
    p = compiled.params
    assert p.logn == 13
    assert p.logq == (29, 26, 26, 26, 26, 26)
    assert p.logp == (29, 29)
    assert p.logscale == 26
    assert p.h == 8192
    assert p.ring_type == "conjugate_invariant"


def test_manifest(compiled):
    m = compiled.manifest
    assert isinstance(m.galois_elements, frozenset)
    assert len(m.galois_elements) > 0
    assert m.needs_rlk is True
    # MLP with these params has no bootstraps
    assert m.bootstrap_slots == ()
    assert m.boot_logp is None


def test_input_level(compiled):
    assert compiled.input_level == 5


def test_topology(compiled):
    # MLP topology: flatten, fc1, act1, fc2, act2, fc3
    assert len(compiled.topology) == 6
    assert compiled.topology[0] == "flatten"
    assert compiled.topology[-1] == "fc3"


def test_roundtrip_serialization(compiled):
    """Verify to_bytes/from_bytes round-trip preserves all fields."""
    data = compiled.to_bytes()
    reloaded = orion.CompiledModel.from_bytes(data)

    assert reloaded.params == compiled.params
    assert reloaded.config == compiled.config
    assert reloaded.manifest.galois_elements == compiled.manifest.galois_elements
    assert reloaded.manifest.needs_rlk == compiled.manifest.needs_rlk
    assert reloaded.manifest.bootstrap_slots == compiled.manifest.bootstrap_slots
    assert reloaded.input_level == compiled.input_level
    assert reloaded.topology == compiled.topology
    assert len(reloaded.blobs) == len(compiled.blobs)
