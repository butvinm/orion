import time
import math

import torch
import orion
from orion.core.utils import get_mnist_datasets, mae, train_on_mnist

import models

torch.manual_seed(42)
scheme = orion.init_scheme("test_parameters.yaml")

batch_size = scheme.get_batch_size()
trainloader, testloader = get_mnist_datasets(
    data_dir="../data", batch_size=batch_size)

net = models.MLP()
#train_on_mnist(net, data_dir="../data", epochs=1, device="cuda")

inp, _ = next(iter(testloader))
net.eval()

out_clear = net(inp) # Cleartext inference

# Fit and compile (temporarily increase batch size for speed)
orion.fit(net, trainloader, batch_size=128)
input_level = orion.compile(net)

# Encode and encrypt
vec_ptxt = orion.encode(inp, input_level)
vec_ctxt = orion.encrypt(vec_ptxt)

net.he() # Set model to FHE inference mode

print("\nStarting FHE inference", flush=True)
start = time.time()
out_ctxt = net(vec_ctxt) # perform FHE inference
end = time.time()

out_ptxt = out_ctxt.decrypt()
out_fhe = out_ptxt.decode()

print()
print(out_clear)
print(out_fhe)

dist = mae(out_clear, out_fhe)
print(f"\nMAE: {dist:.4f}")
print(f"Precision: {-math.log2(dist):.4f}")
print(f"Runtime: {end-start:.4f} secs.\n")