"""Unified FHE inference pipeline for all Orion examples.

Usage:
    python examples/models/run.py mlp
    python examples/models/run.py alexnet --cleartext-only
    python examples/models/run.py resnet --cleartext-only
"""

import argparse
import importlib
import os
import sys
import tempfile
import torch

from orion_compiler import Compiler, CKKSParams

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
AVAILABLE = ["mlp", "lenet", "lola", "alexnet", "vgg", "resnet"]


def load_example(name):
    """Import example module and return (model_instance, config)."""
    sys.path.insert(0, EXAMPLES_DIR)
    mod = importlib.import_module(name)
    sys.path.pop(0)

    torch.manual_seed(42)
    net = mod.Model()

    weights_path = os.path.join(EXAMPLES_DIR, f"{name}_weights.pt")
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        net.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded trained weights from {name}_weights.pt")
    else:
        print("No weights found, using random weights")
    net.eval()

    return net, mod.CONFIG


def cleartext_forward(net, test_input):
    """Run cleartext inference and return flat output list."""
    with torch.no_grad():
        return net(test_input).flatten().tolist()


def compile_and_run(net, config, test_input, cleartext):
    """Compile model and run full FHE pipeline."""
    from orion_evaluator import Model, Evaluator
    from lattigo.ckks import Parameters, Encoder
    from lattigo.rlwe import (
        KeyGenerator,
        Encryptor,
        Decryptor,
        MemEvaluationKeySet,
        Ciphertext as RLWECiphertext,
    )

    # Compile directly to file
    print("Compiling model...")
    ckks_params = CKKSParams(**config["ckks_params"])
    compiler = Compiler(net, ckks_params)
    compiler.fit(test_input)
    model_path = tempfile.mktemp(suffix=".orion")
    compiler.compile_to_file(model_path)
    print(f"Compiled model size: {os.path.getsize(model_path)} bytes")

    # Load in evaluator
    with open(model_path, "rb") as f:
        model = Model.load(f.read())
    os.unlink(model_path)
    params_dict, manifest, input_level = model.client_params()

    # Keygen
    params = Parameters.from_dict(params_dict)
    kg = KeyGenerator(params)
    sk = kg.gen_secret_key()
    pk = kg.gen_public_key(sk)
    encoder = Encoder(params)
    encryptor = Encryptor(params, pk)
    decryptor = Decryptor(params, sk)

    rlk = kg.gen_relin_key(sk) if manifest["needs_rlk"] else None
    gks = [kg.gen_galois_key(sk, int(ge)) for ge in manifest["galois_elements"]]
    evk = MemEvaluationKeySet(rlk=rlk, galois_keys=gks)
    keys_bytes = evk.marshal_binary()

    # Bootstrap keys check
    btp_keys_bytes = None
    bootstrap_slots = manifest.get("bootstrap_slots", [])
    if bootstrap_slots:
        print(f"Model requires bootstrap (slots: {bootstrap_slots})")
        print("Full FHE E2E requires 64+ GB RAM — skipping FHE inference")
        print("Use --cleartext-only to verify model correctness")
        # Cleanup what we've allocated so far
        evk.close()
        for gk in gks:
            gk.close()
        if rlk:
            rlk.close()
        decryptor.close()
        encryptor.close()
        pk.close()
        sk.close()
        kg.close()
        encoder.close()
        params.close()
        model.close()
        return

    evaluator = Evaluator(params_dict, keys_bytes, btp_keys_bytes=btp_keys_bytes)

    # Encrypt
    max_slots = params.max_slots()
    flat = test_input.flatten().double().tolist()
    padded = flat + [0.0] * (max_slots - len(flat))
    scale = params.default_scale()

    pt = encoder.encode(padded, input_level, scale)
    ct = encryptor.encrypt_new(pt)
    ct_bytes = ct.marshal_binary()

    # Forward
    print("Running FHE inference...")
    result_bytes_list = evaluator.forward(model, [ct_bytes])
    result_bytes = result_bytes_list[0]

    # Decrypt
    result_ct = RLWECiphertext.unmarshal_binary(result_bytes)
    result_pt = decryptor.decrypt_new(result_ct)
    decoded = encoder.decode(result_pt, max_slots)

    # Compare
    print(f"\nCleartext output: {[f'{v:.4f}' for v in cleartext]}")
    print(f"FHE output:      {[f'{decoded[i]:.4f}' for i in range(len(cleartext))]}")

    diffs = [abs(cleartext[i] - decoded[i]) for i in range(len(cleartext))]
    mae = sum(diffs) / len(diffs)
    max_diff = max(diffs)
    print(f"\nMAE: {mae:.6f}")
    print(f"Max diff: {max_diff:.6f}")

    assert mae < 0.1, f"MAE {mae} exceeds tolerance 0.1"
    print("PASS: MAE within tolerance")

    # Cleanup
    result_pt.close()
    result_ct.close()
    evaluator.close()
    evk.close()
    for gk in gks:
        gk.close()
    if rlk:
        rlk.close()
    ct.close()
    pt.close()
    decryptor.close()
    encryptor.close()
    pk.close()
    sk.close()
    kg.close()
    encoder.close()
    params.close()
    model.close()


def main():
    parser = argparse.ArgumentParser(description="Run Orion FHE inference examples")
    parser.add_argument("example", choices=AVAILABLE, help="Example model to run")
    parser.add_argument(
        "--cleartext-only",
        action="store_true",
        help="Only run cleartext forward pass (no FHE)",
    )
    args = parser.parse_args()

    net, config = load_example(args.example)

    # Cleartext baseline
    torch.manual_seed(123)
    test_input = torch.randn(*config["input_shape"])
    cleartext = cleartext_forward(net, test_input)
    print(f"Cleartext output: {[f'{v:.4f}' for v in cleartext]}")

    if args.cleartext_only:
        return

    compile_and_run(net, config, test_input, cleartext)


if __name__ == "__main__":
    main()
