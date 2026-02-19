"""Compile MLP model for WASM FHE demo.

Compiles the MLP (784->128->128->10) with CKKS parameters suitable for
the WASM demo, saves as model.bin, and prints manifest info for verification.
"""

import sys
import os

# Ensure orion is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch
import orion
from orion.models import MLP

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(DEMO_DIR, "model.bin")

MLP_PARAMS = orion.CKKSParams(
    logn=13,
    logq=(29, 26, 26, 26, 26, 26),
    logp=(29, 29),
    logscale=26,
    h=8192,
    ring_type="conjugate_invariant",
)


def main():
    torch.manual_seed(42)

    net = MLP()
    net.eval()

    # Random MNIST-shaped data for fitting
    fit_data = torch.randn(1, 1, 28, 28)

    print("Compiling MLP for WASM demo...")
    compiler = orion.Compiler(net, MLP_PARAMS)
    compiler.fit(fit_data)
    compiled = compiler.compile()
    del compiler

    # Serialize
    model_bytes = compiled.to_bytes()
    with open(MODEL_PATH, "wb") as f:
        f.write(model_bytes)
    print(f"\nSaved model.bin ({len(model_bytes)} bytes) to {MODEL_PATH}")

    # Verify round-trip
    loaded = orion.CompiledModel.from_bytes(model_bytes)
    print("\nManifest info:")
    print(f"  Galois elements: {len(loaded.manifest.galois_elements)}")
    print(f"  Needs RLK: {loaded.manifest.needs_rlk}")
    print(f"  Bootstrap slots: {loaded.manifest.bootstrap_slots}")
    print(f"  Boot logP: {loaded.manifest.boot_logp}")
    print(f"  Input level: {loaded.input_level}")
    print(f"  Params: logn={loaded.params.logn}, logq={loaded.params.logq}, "
          f"logp={loaded.params.logp}, logscale={loaded.params.logscale}")
    print(f"  Topology ({len(loaded.topology)} modules): {loaded.topology}")


if __name__ == "__main__":
    main()
