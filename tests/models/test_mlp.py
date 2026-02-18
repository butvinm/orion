import gc

import torch
import orion
import orion.models as models
from orion.core.utils import get_mnist_datasets, mae


MLP_PARAMS = orion.CKKSParams(
    logn=13,
    logq=(29, 26, 26, 26, 26, 26),
    logp=(29, 29),
    logscale=26,
    h=8192,
    ring_type="conjugate_invariant",
)


def _cleanup():
    gc.collect()


def test_mlp():
    torch.manual_seed(42)

    trainloader, testloader = get_mnist_datasets(data_dir="./data", batch_size=1)
    net = models.MLP()

    inp, _ = next(iter(testloader))

    net.eval()
    out_clear = net(inp)

    # Compile
    compiler = orion.Compiler(net, MLP_PARAMS)
    compiler.fit(trainloader)
    compiled = compiler.compile()

    compiled_bytes = compiled.to_bytes()
    del compiler
    _cleanup()

    # Client: generate keys, encode, encrypt
    compiled = orion.CompiledModel.from_bytes(compiled_bytes)
    client = orion.Client(compiled.params)
    keys = client.generate_keys(compiled.manifest)

    pt = client.encode(inp, level=compiled.input_level)
    ct = client.encrypt(pt)
    ct_bytes = ct.to_bytes()
    keys_bytes = keys.to_bytes()
    sk_bytes = client.secret_key

    del client
    _cleanup()

    # Evaluator: run FHE inference
    compiled = orion.CompiledModel.from_bytes(compiled_bytes)
    keys = orion.EvalKeys.from_bytes(keys_bytes)

    net_eval = models.MLP()
    evaluator = orion.Evaluator(net_eval, compiled, keys)

    ct_in = orion.CipherText.from_bytes(ct_bytes, evaluator.backend)
    ct_out = evaluator.run(ct_in)
    ct_out_bytes = ct_out.to_bytes()

    del evaluator
    _cleanup()

    # Client: decrypt
    client = orion.Client(compiled.params, secret_key=sk_bytes)
    ct_result = orion.CipherText.from_bytes(ct_out_bytes, client.backend)
    pt_result = client.decrypt(ct_result)
    out_fhe = client.decode(pt_result)

    dist = mae(out_clear, out_fhe[:, :10].float())

    del client
    _cleanup()

    assert dist < 0.5, f"MAE {dist:.6f} exceeds threshold"
