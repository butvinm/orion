"""Tests for the FastAPI WASM FHE demo server.

Uses httpx.AsyncClient with a real model.bin loaded from disk.
Tests cover manifest, key upload, progress, finalize validation, infer gate, and reset.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.bin"

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
