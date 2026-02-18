import time
import math
import torch
import orion
import orion.models as models
from orion.core.utils import (
    get_cifar_datasets,
    mae,
    train_on_cifar
)

torch.manual_seed(42)

# Model and data
trainloader, testloader = get_cifar_datasets(data_dir="../data", batch_size=1)
net = models.ResNet20()

# Train model (optional)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# train_on_cifar(net, data_dir="../data", epochs=1, device=device)

inp, _ = next(iter(testloader))

# Cleartext inference
net.eval()
out_clear = net(inp)

# Compile the network for FHE (ResNet needs bootstrapping)
params = orion.CKKSParams(
    logn=16,
    logq=(55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40),
    logp=(61, 61, 61),
    logscale=40,
    h=192,
    ring_type="standard",
    boot_logp=(61, 61, 61, 61, 61, 61, 61, 61),
)

compiler = orion.Compiler(net, params)
compiler.fit(inp)
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
net_eval = models.ResNet20()
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
