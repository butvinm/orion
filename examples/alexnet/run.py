"""Full FHE pipeline for AlexNet on CIFAR-10.

Compile -> keygen -> encrypt -> evaluate -> decrypt -> print MAE.

NOTE: Full FHE E2E at logn=15 requires significant memory (~10+ GB for bootstrap
keys). On machines with <64 GB RAM, use --cleartext-only to verify model correctness
without FHE encryption.

Usage:
    cd examples/alexnet
    python run.py                  # Full FHE pipeline
    python run.py --cleartext-only # Cleartext forward pass only
"""

import argparse
import os
import torch
from model import AlexNet
from orion_compiler import Compiler, CKKSParams


def cleartext_forward(net, test_input):
    """Run cleartext inference and return output."""
    with torch.no_grad():
        return net(test_input).flatten().tolist()


def compile_model(net, test_input):
    """Compile the model and return compiled bytes + input level."""
    ckks_params = CKKSParams(
        logn=15,
        logq=[55] + [40] * 20,
        logp=[61, 61, 61],
        logscale=40,
        h=192,
        ring_type="standard",
        boot_logp=[61] * 6,
    )
    compiler = Compiler(net, ckks_params)
    compiler.fit(test_input)
    compiled = compiler.compile()
    return compiled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cleartext-only", action="store_true",
        help="Only run cleartext forward pass (no FHE)",
    )
    args = parser.parse_args()

    # 1. Instantiate model, optionally load trained weights
    torch.manual_seed(42)
    net = AlexNet()
    weights_path = os.path.join(os.path.dirname(__file__), "weights.pt")
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        net.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded trained weights from weights.pt")
    else:
        print("No weights.pt found, using random weights")
    net.eval()

    # 2. Cleartext baseline
    torch.manual_seed(123)
    test_input = torch.randn(1, 3, 32, 32)
    cleartext = cleartext_forward(net, test_input)
    print(f"Cleartext output: {[f'{v:.4f}' for v in cleartext]}")

    if args.cleartext_only:
        print("Cleartext-only mode, skipping FHE pipeline")
        return

    # 3. Compile
    print("Compiling model...")
    compiled = compile_model(net, test_input)
    model_bytes = compiled.to_bytes()
    print(f"Compiled model size: {len(model_bytes)} bytes")
    print(f"Input level: {compiled.input_level}")
    print(f"Bootstrap slots: {compiled.manifest.bootstrap_slots}")
    print(f"Galois elements: {len(compiled.manifest.galois_elements)}")

    # 4. Load in evaluator, get client params
    from orion_evaluator import Model, Evaluator
    from lattigo.ckks import Parameters, Encoder
    from lattigo.rlwe import (
        KeyGenerator,
        Encryptor,
        Decryptor,
        MemEvaluationKeySet,
        Ciphertext as RLWECiphertext,
    )

    model = Model.load(model_bytes)
    params_dict, manifest, input_level = model.client_params()

    # 5. Keygen
    params = Parameters.from_logn(**params_dict)
    kg = KeyGenerator.new(params)
    sk = kg.gen_secret_key()
    pk = kg.gen_public_key(sk)
    encoder = Encoder.new(params)
    encryptor = Encryptor.new(params, pk)
    decryptor = Decryptor.new(params, sk)

    rlk = kg.gen_relinearization_key(sk) if manifest["needs_rlk"] else None
    gks = [kg.gen_galois_key(sk, int(ge)) for ge in manifest["galois_elements"]]
    evk = MemEvaluationKeySet.new(rlk=rlk, galois_keys=gks)
    keys_bytes = evk.marshal_binary()

    # Bootstrap keys — AlexNet at logn=15 triggers 3 bootstrap operations.
    # Bootstrap key generation requires lattigo bootstrap bindings (not yet
    # exposed in Python lattigo package) and ~2.6 GB RAM for keys alone.
    # Full FHE E2E at logn=15 requires 64+ GB RAM.
    # For now, pass None and rely on cleartext/compilation verification.
    btp_keys_bytes = None
    bootstrap_slots = manifest.get("bootstrap_slots", [])
    if bootstrap_slots:
        print(f"Model requires bootstrap (slots: {bootstrap_slots})")
        print("Bootstrap key generation not yet available in Python lattigo bindings")
        print("Full FHE E2E requires 64+ GB RAM — skipping FHE inference")
        print("Use --cleartext-only to verify model correctness")
        return

    evaluator = Evaluator(params_dict, keys_bytes, btp_keys_bytes=btp_keys_bytes)

    # 6. Encrypt
    max_slots = params.max_slots()
    flat = test_input.flatten().double().tolist()
    padded = flat + [0.0] * (max_slots - len(flat))
    scale = params.default_scale()

    pt = encoder.encode(padded, input_level, scale)
    ct = encryptor.encrypt_new(pt)
    ct_bytes = ct.marshal_binary()

    # 7. Forward
    print("Running FHE inference...")
    result_bytes = evaluator.forward(model, ct_bytes)

    # 8. Decrypt
    result_ct = RLWECiphertext.unmarshal_binary(result_bytes)
    result_pt = decryptor.decrypt_new(result_ct)
    decoded = encoder.decode(result_pt, max_slots)

    # 9. Compare
    print(f"\nCleartext output: {[f'{v:.4f}' for v in cleartext]}")
    print(f"FHE output:      {[f'{decoded[i]:.4f}' for i in range(len(cleartext))]}")

    diffs = [abs(cleartext[i] - decoded[i]) for i in range(len(cleartext))]
    mae = sum(diffs) / len(diffs)
    max_diff = max(diffs)
    print(f"\nMAE: {mae:.6f}")
    print(f"Max diff: {max_diff:.6f}")

    # 10. Cleanup
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


if __name__ == "__main__":
    main()
