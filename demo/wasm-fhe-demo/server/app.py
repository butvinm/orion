"""FastAPI server for the WASM FHE demo.

Loads a pre-compiled MLP model and provides endpoints for:
- Serving the model manifest (params + key requirements)
- Receiving evaluation keys (RLK, Galois, bootstrap) one at a time
- Finalizing key upload and constructing the Evaluator
- Running FHE inference on encrypted inputs
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

import orion
from orion.models import MLP

# -- Session state (single-tenant demo) --

session: dict = {}
_backend_lock = asyncio.Lock()

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.bin"


def _load_model() -> orion.CompiledModel:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model.bin not found at {MODEL_PATH}")
    data = MODEL_PATH.read_bytes()
    return orion.CompiledModel.from_bytes(data)


@asynccontextmanager
async def lifespan(app: FastAPI):
    compiled = _load_model()
    session["compiled"] = compiled
    session["rlk"] = None
    session["galois_keys"] = {}
    session["bootstrap_keys"] = {}
    session["evaluator"] = None
    yield
    # Cleanup evaluator on shutdown
    if session.get("evaluator") is not None:
        del session["evaluator"]
    session.clear()


app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# -- API endpoints --


@app.get("/api/manifest")
async def get_manifest():
    """Return CKKS params and key manifest for the client."""
    compiled: orion.CompiledModel = session["compiled"]
    p = compiled.params
    m = compiled.manifest
    return {
        "params": {
            "logn": p.logn,
            "logq": list(p.logq),
            "logp": list(p.logp),
            "logscale": p.logscale,
            "h": p.h,
            "ring_type": p.ring_type,
        },
        "manifest": {
            "galois_elements": sorted(m.galois_elements),
            "bootstrap_slots": list(m.bootstrap_slots),
            "boot_logp": list(m.boot_logp) if m.boot_logp else None,
            "needs_rlk": m.needs_rlk,
        },
        "input_level": compiled.input_level,
    }


@app.post("/api/keys/rlk")
async def upload_rlk(request: Request):
    """Upload the relinearization key (raw Lattigo bytes)."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    session["rlk"] = body
    return {"status": "ok", "size": len(body)}


@app.post("/api/keys/galois/{gal_el}")
async def upload_galois_key(gal_el: int, request: Request):
    """Upload a single Galois (rotation) key."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    session["galois_keys"][gal_el] = body
    return {"status": "ok", "gal_el": gal_el, "size": len(body)}


@app.post("/api/keys/bootstrap/{slot_count}")
async def upload_bootstrap_key(slot_count: int, request: Request):
    """Upload bootstrap key for a given slot count."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    session["bootstrap_keys"][slot_count] = body
    return {"status": "ok", "slot_count": slot_count, "size": len(body)}


@app.post("/api/keys/finalize")
async def finalize_keys():
    """Construct EvalKeys and create the Evaluator."""
    compiled: orion.CompiledModel = session["compiled"]
    manifest = compiled.manifest

    # Validate all required keys are present
    if manifest.needs_rlk and session["rlk"] is None:
        raise HTTPException(status_code=400, detail="RLK required but not uploaded")

    missing_galois = set(manifest.galois_elements) - set(session["galois_keys"].keys())
    if missing_galois:
        raise HTTPException(
            status_code=400,
            detail=f"Missing {len(missing_galois)} Galois keys: {sorted(missing_galois)[:5]}...",
        )

    for slot_count in manifest.bootstrap_slots:
        if slot_count not in session["bootstrap_keys"]:
            raise HTTPException(
                status_code=400,
                detail=f"Missing bootstrap key for slot count {slot_count}",
            )

    # Build EvalKeys directly from raw blobs
    keys = orion.EvalKeys(
        rlk_data=session["rlk"],
        galois_keys=dict(session["galois_keys"]),
        bootstrap_keys=dict(session["bootstrap_keys"]),
    )

    # Create Evaluator (initializes Go backend, loads keys — CPU-bound)
    # Lock serializes access to the Go backend singleton
    net = MLP()
    async with _backend_lock:
        evaluator = await asyncio.to_thread(orion.Evaluator, net, compiled, keys)
        session["evaluator"] = evaluator

    return {"status": "ready"}


@app.get("/api/keys/progress")
async def keys_progress():
    """Return current key upload progress."""
    compiled: orion.CompiledModel = session["compiled"]
    manifest = compiled.manifest
    return {
        "received_galois": len(session["galois_keys"]),
        "total_galois": len(manifest.galois_elements),
        "rlk": session["rlk"] is not None,
        "bootstrap": sorted(session["bootstrap_keys"].keys()),
    }


@app.post("/api/infer")
async def infer(request: Request):
    """Run FHE inference on an encrypted input."""
    if session.get("evaluator") is None:
        raise HTTPException(status_code=400, detail="Evaluator not initialized. Finalize keys first.")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    evaluator: orion.Evaluator = session["evaluator"]
    try:
        # Accept raw ORTXT wire format from WASM client (no Python shape header).
        from orion.backend.orionclient import ffi
        ct_handle = ffi.ciphertext_unmarshal(body)
        ct_in = orion.Ciphertext(ct_handle)
        async with _backend_lock:
            ct_out = await asyncio.to_thread(evaluator.run, ct_in)
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed")
    # Return raw ORTXT wire format (no Python shape header).
    from orion.backend.orionclient import ffi as _ffi
    result_bytes = _ffi.ciphertext_marshal(ct_out.handle)

    return Response(content=result_bytes, media_type="application/octet-stream")


@app.post("/api/reset")
async def reset():
    """Clear session state, destroy evaluator."""
    if session.get("evaluator") is not None:
        del session["evaluator"]
        session["evaluator"] = None
    session["rlk"] = None
    session["galois_keys"] = {}
    session["bootstrap_keys"] = {}
    return {"status": "reset"}


# -- Static files (must be last — catch-all mount) --

CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"
if CLIENT_DIR.exists():
    app.mount("/", StaticFiles(directory=str(CLIENT_DIR), html=True), name="static")
