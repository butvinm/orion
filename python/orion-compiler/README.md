# orion-v2-compiler

Orion FHE compiler — traces, fits, and compiles PyTorch neural networks for encrypted inference using the CKKS scheme.

Part of the [Orion](https://github.com/butvinm/orion) FHE framework.

## Usage

```python
import orion_compiler.nn as on
from orion_compiler import Compiler, CKKSParams

class MLP(on.Module):
    def __init__(self):
        super().__init__()
        self.flatten = on.Flatten()
        self.fc1 = on.Linear(784, 128)
        self.act1 = on.Quad()
        self.fc2 = on.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        return self.fc2(x)

net = MLP()
compiler = Compiler(net, CKKSParams(logn=14, logq=[55, 40, 40, 40], logp=[61, 61], log_default_scale=40))
compiler.fit(dataloader)
compiled = compiler.compile()
model_bytes = compiled.to_bytes()
```

## Modules

- `orion_compiler` — `Compiler`, `CKKSParams`, `CompiledModel`, `Graph`, `GraphNode`, `GraphEdge`, `KeyManifest`, `CompilerConfig`, `CostProfile`
- `orion_compiler.nn` — FHE-compatible layers (cleartext-only forward)
- `orion_compiler.core` — Compilation algorithms (tracer, packing, level assignment, auto-bootstrap, galois)
