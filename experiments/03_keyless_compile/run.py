"""
Experiment 03: Python keyless compilation

Hypothesis: Orion's compile() pipeline (FX tracing -> NetworkDAG -> LevelDAG ->
bootstrap placement -> diagonal packing) can run without generating any
cryptographic keys, and can emit a key requirements manifest listing all
required Galois elements, bootstrap slot counts, and relinearization key flag.

What this proves: That we can cleanly split Orion's compile phase into
"planning" (server, no keys) and "key generation" (client, has sk).
"""

import sys
import os
import json

# Add the project root to path so we can import orion
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torch.utils.data import DataLoader, TensorDataset

import orion
import orion.nn as on


class MLP(on.Module):
    """MLP model definition (was in models/mlp.py, now inlined)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = on.Flatten()
        self.fc1 = on.Linear(784, 128)
        self.bn1 = on.BatchNorm1d(128)
        self.act1 = on.Quad()
        self.fc2 = on.Linear(128, 128)
        self.bn2 = on.BatchNorm1d(128)
        self.act2 = on.Quad()
        self.fc3 = on.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        return self.fc3(x)


def main():
    print("=" * 60)
    print("Experiment 03: Keyless Compilation")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Initialize scheme with params only (no keys)
    # --------------------------------------------------------
    print("\n--- Step 1: Init params only (no keys) ---")

    config = {
        "ckks_params": {
            "LogN": 13,
            "LogQ": [29, 26, 26, 26, 26, 26],
            "LogP": [29, 29],
            "LogScale": 26,
            "H": 8192,
            "RingType": "ConjugateInvariant",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "lattigo",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }

    orion.init_params_only(config)
    print("Scheme initialized with params + encoder only (no keys).")

    # Verify encoder works without keygen
    print("\n--- Verify: Encoder works without keygen ---")
    test_tensor = torch.randn(100)
    ptxt = orion.encode(test_tensor)
    decoded = orion.decode(ptxt)
    max_err = (test_tensor - decoded[:100]).abs().max().item()
    print(f"Encoder test: encode -> decode max error = {max_err:.2e}")
    assert max_err < 1e-3, f"Encoder error too large: {max_err}"
    print("PASS: Go Encoder works without keygen.")

    # --------------------------------------------------------
    # Step 2: Load MLP model and fit
    # --------------------------------------------------------
    print("\n--- Step 2: Load MLP model and fit ---")

    net = MLP(num_classes=10)
    net.eval()

    # Create dummy MNIST-like data for fitting
    dummy_data = torch.randn(64, 1, 28, 28)
    dummy_labels = torch.randint(0, 10, (64,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=32)

    orion.fit(net, dataloader)
    print("Fit completed successfully.")

    # --------------------------------------------------------
    # Step 3: Compile in keyless mode -> get manifest
    # --------------------------------------------------------
    print("\n--- Step 3: Compile in keyless mode ---")

    input_level, manifest = orion.compile(net)

    print(f"\nCompilation completed successfully!")
    print(f"Input level: {input_level}")

    # --------------------------------------------------------
    # Step 4: Print and verify manifest
    # --------------------------------------------------------
    print("\n--- Step 4: Key Requirements Manifest ---")
    print(f"Total Galois elements needed: {len(manifest['galois_elements'])}")
    print(f"  - From linear transforms: "
          f"{len(manifest['linear_transform_galois_elements'])}")
    print(f"  - From power-of-2 rotations: "
          f"{len(manifest['po2_galois_elements'])}")
    print(f"  - From hybrid output rotations: "
          f"{len(manifest['hybrid_output_galois_elements'])}")
    print(f"Bootstrap slot counts: {manifest['bootstrap_slots']}")
    print(f"Needs relinearization key: {manifest['needs_rlk']}")

    print(f"\nAll Galois elements: {manifest['galois_elements']}")

    # Verify manifest is non-empty and well-formed
    assert len(manifest["galois_elements"]) > 0, "Manifest has no galois elements"
    assert manifest["needs_rlk"] is True, "Manifest should need rlk"
    assert isinstance(manifest["bootstrap_slots"], list), "bootstrap_slots should be list"

    # Verify all galois elements are positive integers
    for ge in manifest["galois_elements"]:
        assert isinstance(ge, int) and ge > 0, f"Invalid galois element: {ge}"

    print("\nPASS: Manifest is valid and well-formed.")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 03 RESULTS")
    print("=" * 60)
    print("Hypothesis CONFIRMED:")
    print("  - init_params_only() creates backend + encoder without keys")
    print("  - Go Encoder works without keygen (only needs Params)")
    print("  - compile() runs full pipeline in keyless mode:")
    print("    FX trace -> DAG -> fuse -> diag packing -> bootstrap")
    print("    placement -> module compilation")
    print("  - Key requirements manifest produced with all Galois elements")
    print("  - Bootstrapper key generation skipped, slot counts recorded")
    print(f"  - {len(manifest['galois_elements'])} total unique Galois elements")

    # Save manifest to JSON for reference
    manifest_path = os.path.join(os.path.dirname(__file__), "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    # Clean up
    orion.delete_scheme()
    print("\nDone.")


if __name__ == "__main__":
    main()
