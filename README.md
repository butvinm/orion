# Orion

An opinionated fork of [baahl-nyu/orion](https://github.com/baahl-nyu/orion) — a framework for encrypted deep learning inference. Orion compiles PyTorch neural networks for homomorphic evaluation using the CKKS scheme.

The original project is designed as a research tool for demonstrating FHE inference. This fork narrows the scope to production use.

### Changes from the original

- Built exclusively on [Lattigo](https://github.com/tuneinsight/lattigo) with full access to underlying primitives (keygen, encrypt, decrypt)
- Split into three independent packages: compiler, Lattigo bindings, and evaluator
- Instance-based API — no global state
- Explicit context passing between compilation and evaluation stages
- Custom binary `.orion` model format (replaces HDF5)

## Packages

| Package                                                                                                | Language   | Description                                                                     |
| ------------------------------------------------------------------------------------------------------ | ---------- | ------------------------------------------------------------------------------- |
| [`orion-v2-lattigo`](python/lattigo/) ([PyPI](https://pypi.org/project/orion-v2-lattigo/))             | Python     | Bindings for Lattigo CKKS primitives (keygen, encrypt, decrypt, encode)         |
| [`orion-v2-compiler`](python/orion-compiler/) ([PyPI](https://pypi.org/project/orion-v2-compiler/))    | Python     | Traces PyTorch models and compiles them for encrypted inference (fit → compile) |
| [`orion-v2-evaluator`](python/orion-evaluator/) ([PyPI](https://pypi.org/project/orion-v2-evaluator/)) | Python     | Bindings for the Go evaluator — runs inference on ciphertexts                   |
| [`evaluator`](evaluator/)                                                                              | Go         | Core FHE evaluator: loads compiled models, executes the computation graph       |
| [`js/lattigo`](js/lattigo/)                                                                            | TypeScript | Lattigo CKKS via Go→WASM, runs in browser                                       |

The compiler produces a `.orion` file containing the computation graph and model weights. The client uses `lattigo` directly for key generation and encryption. The evaluator loads the compiled model and evaluation keys, then runs `forward()` on ciphertexts.

## Examples

Model examples in [`examples/models/`](examples/models/):

```bash
python examples/models/run.py mlp          # MNIST MLP — ~3 GB RAM
python examples/models/run.py lenet        # MNIST LeNet — ~3 GB RAM
python examples/models/run.py lola         # MNIST LoLa — ~3 GB RAM

# CIFAR-10 models — bootstrap-enabled, need significant RAM for FHE keygen + eval
python examples/models/run.py alexnet      # AlexNet — ~130 GB RAM
python examples/models/run.py vgg          # VGG — ~130 GB RAM
python examples/models/run.py resnet       # ResNet — ~130 GB RAM
```

Use `--cleartext-only` to skip FHE and verify model correctness without memory requirements.

Browser demo in [`examples/wasm-demo/`](examples/wasm-demo/) — encrypted MNIST inference where the secret key never leaves the browser.

Full FHE demo with bootstrap: [`examples/c3ae-demo/`](examples/c3ae-demo/) — age estimation on encrypted face images.

## Installation

### From PyPI (Linux only)

```bash
pip install orion-v2-lattigo orion-v2-compiler orion-v2-evaluator
```

Requires Python 3.11+. Wheels include prebuilt CGO shared libraries for Linux x86_64.

### From source

**Prerequisites:** Go 1.22+, Python 3.11+, C compiler, `libgmp-dev`, `libssl-dev`, [uv](https://docs.astral.sh/uv/).

```bash
# Ubuntu
sudo apt install -y build-essential libgmp-dev libssl-dev

git clone https://github.com/butvinm/orion.git && cd orion

# Build Go shared libraries (lattigo + evaluator bridges)
python tools/build_lattigo.py

# Install all Python packages (uv workspace)
uv sync
```

## Usage

```python
import orion_compiler.nn as on
from orion_compiler import Compiler, CKKSParams
from lattigo.ckks import Parameters, Encoder
from lattigo.rlwe import (
    KeyGenerator, Encryptor, Decryptor, MemEvaluationKeySet,
    Ciphertext as RLWECiphertext,
)
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
ckks = CKKSParams(logn=14, logq=[55, 40, 40, 40], logp=[61, 61],
                   log_default_scale=40, ring_type="conjugate_invariant")
compiler = Compiler(MLP(), ckks)
compiler.fit(dataloader)
compiler.compile_to_file("model.orion")

# 3. Load compiled model and get client params
with open("model.orion", "rb") as f:
    model = Model.load(f.read())
params_dict, manifest, input_level = model.client_params()

# 4. Keygen + encrypt (client-side, using Lattigo directly)
params = Parameters.from_dict(params_dict)
kg = KeyGenerator(params)
sk, pk = kg.gen_secret_key(), kg.gen_public_key(sk)
encoder = Encoder(params)
encryptor = Encryptor(params, pk)

rlk = kg.gen_relin_key(sk) if manifest["needs_rlk"] else None
gks = [kg.gen_galois_key(sk, int(ge)) for ge in manifest["galois_elements"]]
evk = MemEvaluationKeySet(rlk=rlk, galois_keys=gks)

pt = encoder.encode(input_values, level=input_level, scale=params.default_scale())
ct = encryptor.encrypt_new(pt)

# 5. Evaluate (server-side)
evaluator = Evaluator(params_dict, evk.marshal_binary())
result_bytes_list = evaluator.forward(model, [ct.marshal_binary()])

# 6. Decrypt (client-side)
result_ct = RLWECiphertext.unmarshal_binary(result_bytes_list[0])
decryptor = Decryptor(params, sk)
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
