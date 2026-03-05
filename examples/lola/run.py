"""Full FHE pipeline for LoLA on MNIST.

Compile -> keygen -> encrypt -> evaluate -> decrypt -> print MAE.

Usage: cd examples/lola && python run.py
"""

import os
import torch
from model import LoLA
from orion_compiler import Compiler, CKKSParams
from orion_evaluator import Model, Evaluator
from lattigo.ckks import Parameters, Encoder
from lattigo.rlwe import (
    KeyGenerator,
    Encryptor,
    Decryptor,
    MemEvaluationKeySet,
    Ciphertext as RLWECiphertext,
)


def main():
    # 1. Instantiate model, optionally load trained weights
    torch.manual_seed(42)
    net = LoLA()
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
    test_input = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        cleartext = net(test_input).flatten().tolist()

    # 3. Compile
    # LoLA: Conv2d(1,5,2,s=2) → BN2d → Quad → Flatten → Linear(980,100) → BN1d → Quad → Linear(100,10)
    ckks_params = CKKSParams(
        logn=13,
        logq=[29, 26, 26, 26, 26, 26, 26, 26, 26, 26],
        logp=[29, 29],
        logscale=26,
        h=8192,
        ring_type="conjugate_invariant",
    )
    compiler = Compiler(net, ckks_params)
    compiler.fit(torch.randn(1, 1, 28, 28))
    compiled = compiler.compile()
    model_bytes = compiled.to_bytes()

    # 4. Load in evaluator, get client params
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

    evaluator = Evaluator(params_dict, keys_bytes)

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

    assert mae < 0.1, f"MAE {mae} exceeds tolerance 0.1"
    print("PASS: MAE within tolerance")

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
