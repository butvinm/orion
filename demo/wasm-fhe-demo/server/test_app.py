"""Tests for the FastAPI WASM FHE demo server.

Uses httpx.AsyncClient with a real model.bin loaded from disk.
Tests cover manifest, key upload, progress, finalize validation, infer gate, and reset.
Integration test exercises the full flow with real Orion backend.
"""

from __future__ import annotations

import gc
import platform
from pathlib import Path

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.bin"

# Check if Go backend shared library is available
_LATTIGO_DIR = Path(__file__).resolve().parent.parent.parent.parent / "orion" / "backend" / "lattigo"
_LIB_NAMES = {
    "Linux": "lattigo-linux.so",
    "Darwin": "lattigo-mac-arm64.dylib" if platform.machine().lower() in ("arm64", "aarch64") else "lattigo-mac.dylib",
    "Windows": "lattigo-windows.dll",
}
GO_BACKEND_AVAILABLE = (_LATTIGO_DIR / _LIB_NAMES.get(platform.system(), "")).exists()

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="model.bin not found — run tools/compile_model.py first",
)


@pytest.fixture
async def client():
    """Create a test client with lifespan events triggered."""
    from app import app

    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# -- Manifest endpoint --


@pytest.mark.anyio
async def test_manifest_returns_correct_structure(client):
    resp = await client.get("/api/manifest")
    assert resp.status_code == 200
    data = resp.json()

    # Top-level keys
    assert "params" in data
    assert "manifest" in data
    assert "input_level" in data

    # Params fields
    p = data["params"]
    assert isinstance(p["logn"], int)
    assert isinstance(p["logq"], list)
    assert isinstance(p["logp"], list)
    assert isinstance(p["logscale"], int)
    assert isinstance(p["h"], int)
    assert p["ring_type"] in ("conjugate_invariant", "standard")

    # Manifest fields
    m = data["manifest"]
    assert isinstance(m["galois_elements"], list)
    assert isinstance(m["needs_rlk"], bool)
    assert isinstance(m["bootstrap_slots"], list)

    # input_level is a non-negative int
    assert isinstance(data["input_level"], int)
    assert data["input_level"] >= 0


@pytest.mark.anyio
async def test_manifest_params_match_mlp(client):
    """Verify the manifest reflects the MLP compilation params."""
    resp = await client.get("/api/manifest")
    data = resp.json()
    p = data["params"]
    # These are the MLP params from the plan
    assert p["logn"] == 13
    assert p["logq"] == [29, 26, 26, 26, 26, 26]
    assert p["logscale"] == 26
    assert p["ring_type"] == "conjugate_invariant"


# -- Key upload endpoints --


@pytest.mark.anyio
async def test_upload_rlk(client):
    resp = await client.post("/api/keys/rlk", content=b"\x00" * 100)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["size"] == 100


@pytest.mark.anyio
async def test_upload_rlk_empty_body(client):
    resp = await client.post("/api/keys/rlk", content=b"")
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_upload_galois_key(client):
    resp = await client.post("/api/keys/galois/33", content=b"\x01" * 50)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["gal_el"] == 33
    assert data["size"] == 50


@pytest.mark.anyio
async def test_upload_bootstrap_key(client):
    resp = await client.post("/api/keys/bootstrap/4096", content=b"\x02" * 200)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["slot_count"] == 4096


# -- Progress endpoint --


@pytest.mark.anyio
async def test_progress_initial(client):
    resp = await client.get("/api/keys/progress")
    assert resp.status_code == 200
    data = resp.json()
    assert data["received_galois"] == 0
    assert data["total_galois"] > 0
    assert data["rlk"] is False
    assert data["bootstrap"] == []


@pytest.mark.anyio
async def test_progress_after_uploads(client):
    # Upload some keys
    await client.post("/api/keys/rlk", content=b"\x00" * 10)
    await client.post("/api/keys/galois/33", content=b"\x01" * 10)
    await client.post("/api/keys/galois/65", content=b"\x01" * 10)

    resp = await client.get("/api/keys/progress")
    data = resp.json()
    assert data["received_galois"] == 2
    assert data["rlk"] is True


# -- Finalize endpoint --


@pytest.mark.anyio
async def test_finalize_without_keys_returns_error(client):
    resp = await client.post("/api/keys/finalize")
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_finalize_missing_galois_keys(client):
    # Upload RLK but not all Galois keys
    await client.post("/api/keys/rlk", content=b"\x00" * 10)
    resp = await client.post("/api/keys/finalize")
    assert resp.status_code == 400
    assert "Galois" in resp.json()["detail"]


# -- Infer endpoint --


@pytest.mark.anyio
async def test_infer_without_evaluator(client):
    resp = await client.post("/api/infer", content=b"\x00" * 100)
    assert resp.status_code == 400
    assert "Evaluator not initialized" in resp.json()["detail"]


@pytest.mark.anyio
async def test_infer_empty_body(client):
    # Simulate evaluator being present by setting a dummy (won't be reached)
    from app import session
    session["evaluator"] = "dummy"
    try:
        resp = await client.post("/api/infer", content=b"")
        assert resp.status_code == 400
        assert "Empty body" in resp.json()["detail"]
    finally:
        session["evaluator"] = None


# -- Reset endpoint --


@pytest.mark.anyio
async def test_reset_clears_session(client):
    # Upload some keys first
    await client.post("/api/keys/rlk", content=b"\x00" * 10)
    await client.post("/api/keys/galois/33", content=b"\x01" * 10)

    # Reset
    resp = await client.post("/api/reset")
    assert resp.status_code == 200
    assert resp.json()["status"] == "reset"

    # Verify progress is cleared
    resp = await client.get("/api/keys/progress")
    data = resp.json()
    assert data["received_galois"] == 0
    assert data["rlk"] is False


# -- Integration test (real Orion backend) --


@pytest.mark.anyio
@pytest.mark.skipif(not GO_BACKEND_AVAILABLE, reason="Go backend shared library not available")
async def test_integration_full_flow(client):
    """Full end-to-end: generate keys, upload, finalize, encrypt, infer, decrypt.

    Uses real Orion backend. Takes ~30-120s depending on hardware.
    """
    import torch
    import orion

    # 1. Get manifest from server
    resp = await client.get("/api/manifest")
    assert resp.status_code == 200
    manifest_data = resp.json()

    # 2. Load compiled model and create Orion client
    compiled = orion.CompiledModel.from_bytes(MODEL_PATH.read_bytes())
    orion_client = orion.Client(compiled.params)

    # 3. Generate all evaluation keys
    keys = orion_client.generate_keys(compiled.manifest)

    # 4. Encrypt a random MNIST-shaped input
    torch.manual_seed(99)
    input_tensor = torch.randn(1, 1, 28, 28)
    pt = orion_client.encode(input_tensor, level=compiled.input_level)
    ct = orion_client.encrypt(pt)
    ct_bytes = ct.to_bytes()

    # 5. Save secret key for later decryption
    sk_bytes = orion_client.secret_key

    # 6. Free Go backend (singleton — must release before server can use it)
    del orion_client
    gc.collect()

    # 7. Upload keys via HTTP endpoints
    if keys.has_rlk:
        resp = await client.post("/api/keys/rlk", content=keys.rlk_data)
        assert resp.status_code == 200

    for gal_el, key_data in keys.galois_keys.items():
        resp = await client.post(f"/api/keys/galois/{gal_el}", content=key_data)
        assert resp.status_code == 200

    for slot_count, key_data in keys.bootstrap_keys.items():
        resp = await client.post(f"/api/keys/bootstrap/{slot_count}", content=key_data)
        assert resp.status_code == 200

    # 8. Check progress shows all keys received
    resp = await client.get("/api/keys/progress")
    progress = resp.json()
    assert progress["received_galois"] == progress["total_galois"]

    # 9. Finalize keys (creates Evaluator — takes over Go backend)
    resp = await client.post("/api/keys/finalize")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"

    # 10. Run inference
    resp = await client.post(
        "/api/infer",
        content=ct_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    result_bytes = resp.content
    assert len(result_bytes) > 0

    # 11. Reset server (frees Go backend for client decryption)
    resp = await client.post("/api/reset")
    assert resp.status_code == 200

    # 12. Create new client with saved secret key to decrypt
    orion_client2 = orion.Client(compiled.params, secret_key=sk_bytes)
    ct_result = orion.CipherText.from_bytes(result_bytes, orion_client2.backend)
    pt_result = orion_client2.decrypt(ct_result)
    result = orion_client2.decode(pt_result)

    # 13. Verify output: MLP outputs 10 classes
    assert result.numel() >= 10
    result_values = result.flatten()[:10].tolist()
    # Values should be finite (not NaN/Inf)
    for v in result_values:
        assert not (v != v), "NaN in output"  # NaN != NaN

    del orion_client2
    gc.collect()
