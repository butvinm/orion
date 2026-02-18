import time
import math
import torch
import orion
import orion.models as models
from orion.core.utils import (
    get_mnist_datasets,
    mae,
    train_on_mnist
)

torch.manual_seed(42)

# Model and data
trainloader, testloader = get_mnist_datasets(data_dir="../data", batch_size=1)
net = models.MLP()

# Train model (optional)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# train_on_mnist(net, data_dir="../data", epochs=1, device=device)

inp, _ = next(iter(testloader))

# Cleartext inference
net.eval()
out_clear = net(inp)

# Compile the network for FHE
params = orion.CKKSParams(
    logn=13,
    logq=(29, 26, 26, 26, 26, 26),
    logp=(29, 29),
    logscale=26,
    h=8192,
    ring_type="conjugate_invariant",
)

compiler = orion.Compiler(net, params)
compiler.fit(inp, batch_size=128)
compiled = compiler.compile()
compiled_bytes = compiled.to_bytes()
del compiler

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

# Evaluator: run FHE inference
compiled = orion.CompiledModel.from_bytes(compiled_bytes)
keys = orion.EvalKeys.from_bytes(keys_bytes)
net_eval = models.MLP()
evaluator = orion.Evaluator(net_eval, compiled, keys)

ct_in = orion.CipherText.from_bytes(ct_bytes, evaluator.backend)

print("\nStarting FHE inference", flush=True)
start = time.time()
ct_out = evaluator.run(ct_in)
end = time.time()

ct_out_bytes = ct_out.to_bytes()
del evaluator

# Client: decrypt
client = orion.Client(compiled.params, secret_key=sk_bytes)
ct_result = orion.CipherText.from_bytes(ct_out_bytes, client.backend)
pt_result = client.decrypt(ct_result)
out_fhe = client.decode(pt_result)

print()
print(out_clear)
print(out_fhe[:, :10].float())

dist = mae(out_clear, out_fhe[:, :10].float())
print(f"\nMAE: {dist:.4f}")
print(f"Precision: {-math.log2(dist):.4f}")
print(f"Runtime: {end-start:.4f} secs.\n")
del client
