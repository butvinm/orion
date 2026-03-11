# Orion

An opinionated fork of [baahl-nyu/orion](https://github.com/baahl-nyu/orion) — a research-grade FHE framework for deep learning inference.

The original Orion is tightly coupled: global state, implicit context, no separation between compilation and evaluation. This fork refactors it for practical use by splitting the system into three independent packages with explicit APIs and full access to underlying [Lattigo](https://github.com/tuneinsight/lattigo) primitives.

## Packages

| Package                                      | Language   | Description                                                                     |
| -------------------------------------------- | ---------- | ------------------------------------------------------------------------------- |
| [`lattigo`](python/lattigo/)                 | Python     | Bindings for Lattigo CKKS primitives (keygen, encrypt, decrypt, encode)         |
| [`orion-compiler`](python/orion-compiler/)   | Python     | Traces PyTorch models and compiles them for encrypted inference (fit → compile) |
| [`orion-evaluator`](python/orion-evaluator/) | Python     | Bindings for the Go evaluator — runs inference on ciphertexts                   |
| [`evaluator`](evaluator/)                    | Go         | Core FHE evaluator: loads compiled models, executes the computation graph       |
| [`js/lattigo`](js/lattigo/)                  | TypeScript | Lattigo CKKS via Go→WASM, runs in browser                                       |

The compiler produces a `.orion` file containing the computation graph and model weights. The client uses `lattigo` directly for key generation and encryption. The evaluator loads the compiled model and evaluation keys, then runs `forward()` on ciphertexts.

## Examples

Model examples in [`examples/models/`](examples/models/):

```bash
python examples/models/run.py mlp        # MNIST MLP — small, runs on any machine
python examples/models/run.py lenet      # MNIST LeNet
python examples/models/run.py lola       # MNIST LoLa

# CIFAR-10 models (bootstrap-enabled, require 64+ GB RAM for full FHE)
python examples/models/run.py alexnet --cleartext-only
python examples/models/run.py vgg --cleartext-only
python examples/models/run.py resnet --cleartext-only
```

Browser demo in [`examples/wasm-demo/`](examples/wasm-demo/) — encrypted MNIST inference where the secret key never leaves the browser.

## Installation

**Prerequisites:** Go 1.22+, Python 3.9–3.12, C compiler, `libgmp-dev`, `libssl-dev`.

```bash
# Ubuntu
sudo apt install -y build-essential libgmp-dev libssl-dev

git clone https://github.com/butvinm/orion.git && cd orion

# Build the Go shared library (required before Python packages)
python tools/build_lattigo.py

# Install Python packages
pip install -e python/lattigo
pip install -e python/orion-compiler
pip install -e python/orion-evaluator
```

## Usage

```python
import orion_compiler.nn as on
from orion_compiler import Compiler, CKKSParams
from lattigo.ckks import Parameters, Encoder
from lattigo.rlwe import KeyGenerator, Encryptor, Decryptor, MemEvaluationKeySet
from orion_evaluator import Model, Evaluator

# 1. Define model
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

# 2. Compile
compiler = Compiler(MLP(), CKKSParams(logn=14, logq=[55, 40, 40, 40], logp=[61, 61], logscale=40))
compiler.fit(dataloader)
compiled = compiler.compile()
model_bytes = compiled.to_bytes()

# 3. Encrypt (client-side, using Lattigo directly)
params = Parameters.from_logn(logn=14, logq=[55, 40, 40, 40], logp=[61, 61], logscale=40)
kg = KeyGenerator.new(params)
sk, pk = kg.gen_secret_key(), kg.gen_public_key(sk)
encoder = Encoder.new(params)
encryptor = Encryptor.new(params, pk)

pt = encoder.encode(input_values, level=compiled.input_level, scale=params.default_scale())
ct = encryptor.encrypt_new(pt)

# 4. Evaluate (server-side)
model = Model.load(model_bytes)
evaluator = Evaluator(model.client_params()[0], evk.marshal_binary())
result_bytes = evaluator.forward(model, ct.marshal_binary())

# 5. Decrypt (client-side)
from lattigo.rlwe import Ciphertext as RLWECiphertext
result_ct = RLWECiphertext.unmarshal_binary(result_bytes)
decryptor = Decryptor.new(params, sk)
output = encoder.decode(decryptor.decrypt_new(result_ct), params.max_slots())
```

## Tests

```bash
pytest python/tests/              # Python tests
go test ./evaluator/...           # Go evaluator tests
cd js/lattigo && npm test         # JS/WASM tests
```

## License

See [LICENSE](LICENSE).
