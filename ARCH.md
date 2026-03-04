# Orion Architecture

## Vision

Orion is an FHE framework for deep neural network inference. It takes PyTorch models, analyzes them, and produces artifacts that enable encrypted inference using the CKKS scheme.

Three core components:

1. **Compiler** — Python + Lattigo (via Go bridge). Analyzes PyTorch models and produces a portable compiled model with the FHE computation graph, packed numerical data, and all metadata needed for inference. Uses Lattigo for crypto-specific computations (polynomial generation, parameter validation, cost estimation) — no reimplementation.
2. **Evaluator** — Pure Go. Loads the compiled model, CKKS-encodes the packed data into Lattigo format at load time, and runs FHE inference on ciphertexts.
3. **Lattigo Bindings** — Go core with Python and JS wrappers. Thin bindings for Lattigo's key generation, encryption, decryption, encoding, and decoding. Not Orion-specific.

Plus Python evaluator bindings (**orion-evaluator**) for self-contained Python workflows. No client helper library — users interact with Lattigo directly for all cryptographic operations.

## Components

### orion-compiler (Python + Lattigo)

**Dependencies:** PyTorch, numpy, Lattigo (via Go bridge).

The compiler uses Lattigo for crypto-specific operations during `fit()`: polynomial approximation (Chebyshev fitting, minimax/Remez for ReLU) and CKKS parameter construction. Key manifest computation (Galois elements) is now pure Python — the `compile()` step has zero Go/Lattigo dependency.

The compiler performs:

- FX tracing of PyTorch models (or, in the future, parsing ONNX models)
- Statistical fitting (value ranges per layer)
- Polynomial approximation for activations (delegates to Lattigo)
- Network DAG construction, level assignment, bootstrap placement
- Diagonal extraction from weight matrices (packing)
- Key manifest computation (pure Python BSGS algorithm, no Lattigo dependency)
- Cost profiling (bootstrap count, key counts; rotation count deferred)

**Output:** A compiled model file (`.orion`) containing a JSON header with the computation graph (nodes, edges, metadata) and binary blobs with packed numerical data (raw float64 diagonal matrices, bias vectors, polynomial coefficients). No Lattigo binary formats — the compiled model is a pure mathematical description that any CKKS implementation can consume.

### orion-evaluator (Go)

**Pure Go, depends on Lattigo.** Currently uses the `baahl-nyu/lattigo` fork, which is an older Lattigo snapshot with one addition: `ShallowCopy()` for concurrent bootstrapping. Upstream `tuneinsight/lattigo` has since diverged (new buffer pool system, changed lintrans constructor API). Migration to upstream is planned after the main refactoring is complete.

The evaluator has two core types — **Model** and **Evaluator**:

**Model** — loaded once, shared across evaluators, immutable after creation:

1. Parses the compiled model file (JSON header + binary blobs)
2. CKKS-encodes the raw packed diagonals into Lattigo LinearTransforms (one-time startup cost, ~5-30s depending on model size)
3. Stores the computation graph, pre-encoded transforms, and polynomial coefficients
4. Exposes client params (CKKS parameters, key manifest, input level) via `model.ClientParams()`

**Evaluator** — created per-client, holds evaluation keys:

1. Accepts evaluation keys from the user
2. Constructs a Lattigo evaluator instance parameterized by those keys
3. Runs FHE inference by walking the model's computation graph: `evaluator.Forward(model, ciphertexts)`
4. Returns result ciphertexts

All methods return `(result, error)`. Errors include level mismatches, missing Galois keys, bootstrap failures, and scale overflows.

### lattigo bindings (Go + Python/JS wrappers)

**Not Orion-specific.** Thin wrappers exposing Lattigo's:

- Key generation (secret key, relinearization key, Galois keys, bootstrap keys)
- Encryption (plaintext → ciphertext)
- Decryption (ciphertext → plaintext)
- Encoding (float64 vector → CKKS plaintext)
- Decoding (CKKS plaintext → float64 vector)
- Serialization (marshal/unmarshal for all object types)

Anyone doing FHE with Lattigo from Python or JS can use these bindings, regardless of whether they use Orion.

### orion-evaluator (Python bindings to Go evaluator)

**Dependencies:** `lattigo` (Python).

Thin Python bindings over the Go `evaluator/` package, following the same CGO bridge pattern as `python/lattigo/`. Ships a separate `.so` that wraps the evaluator-specific Go code (see Phase 3.5 for the full bridge C export table). Accepts raw Lattigo `MarshalBinary` bytes for keys and ciphertexts — no Orion-specific serialization formats.

Python API:

```python
from orion_evaluator import Model, Evaluator

# Load model (one-time cost: parses .orion, CKKS-encodes diagonals)
model = Model.load(open("model.orion", "rb").read())
params, manifest, input_level = model.client_params()

# Create evaluator with CKKS params dict and Lattigo-serialized eval keys
evaluator = Evaluator(params, keys_bytes)

# Run inference (Lattigo ciphertext bytes in, Lattigo ciphertext bytes out)
result_bytes = evaluator.forward(model, ct_bytes)

# Cleanup
evaluator.close()
model.close()
```

The `Model` object is immutable and safe to share across multiple `Evaluator` instances. Each `Evaluator` holds one client's evaluation keys and is **not** thread-safe (same constraint as the Go evaluator — Lattigo evaluators carry internal buffers).

`Model` and `Evaluator` are `GoHandle`-wrapped — they follow the same RAII pattern as other bridge objects (idempotent `close()`, `__del__` fallback).

This package exists purely for convenience — it enables self-contained Python examples and testing without a separate Go server. Production deployments should use the Go evaluator directly for performance and to avoid CGO overhead.

### No orion-client package

There is no `orion-client` package. Encryption, decryption, key generation, and encoding/decoding are the user's domain — Orion must not constrain access to Lattigo primitives (see Design Principles). Users interact with Lattigo directly via the `lattigo` Python/JS bindings for all cryptographic operations. This enables threshold encryption, custom key management, or any protocol Lattigo supports without Orion getting in the way.

Tensor-to-CKKS-slot mapping (flatten, pad to slot count, split into chunks) is trivial and left to the user. Example code is provided in `examples/` but not packaged as a library.

`CKKSParams` and `CompiledModel` live in `orion-compiler` — they are compiler output types.

## End-to-End Example

Three actors, three languages, three machines.

### 1. Train and compile (Python, developer machine)

```python
import torch
from orion_compiler import Compiler, CKKSParams
from orion_compiler.nn import Linear, BatchNorm1d, Quad, Flatten, Module

# Define the model using Orion's FHE-compatible layers
class MLP(Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.fc1 = Linear(784, 64)
        self.bn1 = BatchNorm1d(64)
        self.act1 = Quad()
        self.fc2 = Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.bn1(self.fc1(x)))
        return self.fc2(x)

# Train the model (standard PyTorch, nothing Orion-specific)
net = MLP()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
for epoch in range(10):
    for x, y in train_loader:
        loss = torch.nn.functional.cross_entropy(net(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Compile for FHE (no keys, no encryption — but uses Lattigo via bridge)
params = CKKSParams(
    logn=14,
    logq=[55, 40, 40, 40, 40, 40, 40, 40, 55],
    logp=[56, 56],
    logscale=40,
)
compiler = Compiler(net, params)
compiler.fit(train_loader)
compiled = compiler.compile()
with open("model.orion", "wb") as f:
    f.write(compiled.to_bytes())
```

The compiler produces `model.orion` — a JSON header describing the computation graph plus binary blobs with raw packed numerical data (float64 diagonals, bias vectors, polynomial coefficients). Lattigo is used during compilation (via Go bridge) for polynomial approximation, parameter validation, and key manifest computation, but the output contains no Lattigo binary formats. The compiled model is a portable mathematical description.

Note: `CKKSParams` and `CompiledModel` live in `orion_compiler`. The Go evaluator parses `.orion` files directly and returns client params as JSON — it does not depend on Python types.

### 2. Serve inference (Go, server)

```go
package main

import (
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
    "sync"

    "github.com/baahl-nyu/lattigo/v6/core/rlwe"
    "github.com/baahl-nyu/lattigo/v6/schemes/ckks"

    "github.com/baahl-nyu/orion/evaluator"
)

func main() {
    // Load model once — CKKS-encodes packed diagonals at load time (one-time cost)
    data, _ := os.ReadFile("model.orion")
    model, _ := evaluator.LoadModel(data)

    // Cache CKKS params for evaluator construction
    orionParams, _, _ := model.ClientParams()
    ckksParams, _ := orionParams.NewCKKSParameters()

    var sessions sync.Map

    // GET /params — client needs CKKS params, key manifest, and input level
    // to generate the right keys and encode at the right level
    http.HandleFunc("GET /params", func(w http.ResponseWriter, r *http.Request) {
        params, manifest, inputLevel := model.ClientParams()
        paramsJSON, _ := json.Marshal(params)
        manifestJSON, _ := json.Marshal(manifest)
        json.NewEncoder(w).Encode(map[string]any{
            "ckks_params":  json.RawMessage(paramsJSON),
            "key_manifest": json.RawMessage(manifestJSON),
            "input_level":  inputLevel,
        })
    })

    // POST /session — client uploads eval keys, server creates evaluator
    http.HandleFunc("POST /session", func(w http.ResponseWriter, r *http.Request) {
        keysData, _ := io.ReadAll(r.Body)
        evk := &rlwe.MemEvaluationKeySet{}
        evk.UnmarshalBinary(keysData)
        eval, _ := evaluator.NewEvaluatorFromKeySet(ckksParams, evk)
        id := generateID()
        sessions.Store(id, eval)
        json.NewEncoder(w).Encode(map[string]string{"id": id})
    })

    // POST /session/{id}/infer — client sends ciphertext, evaluator uses model + cached keys
    http.HandleFunc("POST /session/{id}/infer", func(w http.ResponseWriter, r *http.Request) {
        val, _ := sessions.Load(r.PathValue("id"))
        eval := val.(*evaluator.Evaluator)

        ctData, _ := io.ReadAll(r.Body)
        ct := &rlwe.Ciphertext{}
        ct.UnmarshalBinary(ctData)
        result, _ := eval.Forward(model, ct)
        resultBytes, _ := result.MarshalBinary()
        w.Write(resultBytes)
    })

    http.ListenAndServe(":8080", nil)
}
```

Pure Go. No Python, no PyTorch, no skeleton network. The model is loaded once and shared across all evaluators. Each evaluator holds one client's evaluation keys. `eval.Forward(model, ct)` walks the model's computation graph using that evaluator's keys. Eval keys are uploaded once per session — not per request (Galois keys alone can be 100MB+).

### 3. Encrypt and query (JavaScript, browser)

```javascript
import {
  CKKSParameters,
  KeyGenerator,
  Encryptor,
  Decryptor,
  Encoder,
  Ciphertext,
  MemEvaluationKeySet,
} from "lattigo";

// Fetch params from server — CKKS params, key manifest, input level
const params = await fetch("/params").then((r) => r.json());

// Generate keys in the browser (runs in WASM, secret key never leaves the client)
const ckks = CKKSParameters.fromJSON(params.ckks);
const keygen = KeyGenerator.new(ckks);
const sk = keygen.genSecretKey();
const pk = keygen.genPublicKey(sk);
const rlk = keygen.genRelinKey(sk);
const gks = params.manifest.galois_elements.map((el) =>
  keygen.genGaloisKey(sk, el),
);
const evalKeys = MemEvaluationKeySet.new(rlk, gks);

// Upload eval keys once — server caches them in a session
const { id: sessionId } = await fetch("/session", {
  method: "POST",
  body: evalKeys.marshalBinary(),
}).then((r) => r.json());

// Encode + encrypt the input (flatten, pad to slots, encode, encrypt)
const encoder = Encoder.new(ckks);
const slots = ckks.maxSlots();
const padded = new Float64Array(slots);
padded.set(inputData.flat());
const pt = encoder.encode(padded, params.inputLevel, ckks.defaultScale());
const encryptor = Encryptor.new(ckks, pk);
const ct = encryptor.encryptNew(pt);

// Run inference — only ciphertext sent, keys are already on server
const response = await fetch(`/session/${sessionId}/infer`, {
  method: "POST",
  body: ct.marshalBinary(),
});

// Decrypt the result and decode
const resultCt = Ciphertext.unmarshalBinary(
  new Uint8Array(await response.arrayBuffer()),
);
const decryptor = Decryptor.new(ckks, sk);
const resultPt = decryptor.decryptNew(resultCt);
const output = encoder.decode(resultPt, slots).slice(0, 10);
```

The secret key is generated in the browser and never leaves it. Lattigo bindings (Go compiled to WASM, ~8 MB uncompressed / ~3 MB gzipped) handle all cryptographic operations and serialization. Tensor-to-slot mapping (flatten, pad) is trivial user code — no library needed. For models requiring bootstrap keys, see `examples/wasm-demo/client/client.ts` for the full flow including `BootstrapParameters` construction.

For a self-contained Python example (compile + keygen + encrypt + evaluate + decrypt in one script), see the `CLAUDE.md` end-to-end usage section or the Phase 5 `run.py` template below.

## Compiled Model Format

The compiled model uses a simple binary container: magic + JSON header + length-prefixed binary blobs. No third-party serialization libraries — Go reads it with `encoding/json` + `encoding/binary` (stdlib), Python with `json` + `struct` (stdlib), JS with `JSON.parse` + `DataView` (native).

### Binary layout

```
[8]   MAGIC ("ORION\x00\x02\x00")
[4]   HEADER_LEN (uint32 LE)
[N]   HEADER_JSON (utf-8)
[4]   BLOB_COUNT (uint32 LE)
for each blob:
  [8]  BLOB_LEN (uint64 LE)
  [N]  BLOB_DATA
```

### Header JSON structure

```json
{
  "version": 2,
  "params": {
    "logn": 14,
    "logq": [55, 40, 40, 40, 40, 40, 40, 40, 55],
    "logp": [56, 56],
    "logscale": 40,
    "h": 192,
    "ring_type": "conjugate_invariant",
    "boot_logp": []
  },
  "config": {
    "margin": 2,
    "embedding_method": "hybrid",
    "fuse_modules": true
  },
  "manifest": {
    "galois_elements": [1, 2, 4, 8, 16, 32, 64, 128],
    "bootstrap_slots": [],
    "boot_logp": [],
    "needs_rlk": true
  },
  "input_level": 7,
  "cost": {
    "bootstrap_count": 0,
    "galois_key_count": 14,
    "bootstrap_key_count": 0
  },
  "graph": {
    "input": "flatten",
    "output": "fc3",
    "nodes": [
      {
        "name": "flatten",
        "op": "flatten",
        "level": 7,
        "depth": 0,
        "config": {}
      },
      {
        "name": "fc1",
        "op": "linear_transform",
        "level": 7,
        "depth": 1,
        "shape": {
          "input": [1, 784],
          "output": [1, 128],
          "fhe_input": [1, 8192],
          "fhe_output": [1, 8192]
        },
        "config": {
          "bsgs_ratio": 2.0,
          "output_rotations": 0
        },
        "blob_refs": {
          "diag_0_0": 0,
          "bias": 1
        }
      },
      {
        "name": "act1",
        "op": "quad",
        "level": 6,
        "depth": 1,
        "config": {}
      }
    ],
    "edges": [
      { "src": "flatten", "dst": "fc1" },
      { "src": "fc1", "dst": "act1" }
    ]
  }
}
```

### What blobs contain

The compiled model stores **raw numerical data**, not Lattigo-serialized artifacts. This keeps the format Lattigo-version-independent and portable to any CKKS implementation.

**Diagonal blobs** (for `linear_transform` nodes):

```
[4]              NUM_DIAGS (uint32 LE)
[NUM_DIAGS × 4]  DIAG_INDICES (int32 LE)
[NUM_DIAGS × max_slots × 8]  VALUES (float64 LE, IEEE 754)
```

Fixed-stride layout. The evaluator reads the raw float64 diagonals and CKKS-encodes them into Lattigo LinearTransforms at model load time (one-time cost).

**Bias blobs:** raw float64 arrays (IEEE 754 LE), same padding/slot layout as diagonals.

**Polynomial coefficients** for Chebyshev and activation nodes are small (degree ≤ 63) and stored inline in the node's JSON `config` — no blob needed.

### Size comparison

Storing raw diagonals instead of Lattigo-encoded data reduces blob size significantly. For logn=14, level=5:

|                                   | Per diagonal                         | Ratio |
| --------------------------------- | ------------------------------------ | ----- |
| Raw float64                       | max_slots × 8 = 128 KB               | 1×    |
| CKKS-encoded (NTT+Montgomery+RNS) | (level+1) × ring_degree × 8 = 768 KB | 6×    |

### Why not ONNX / protobuf / FlatBuffers

**ONNX** — Orion's custom ops (`linear_transform`, `chebyshev`, `bootstrap`) are opaque to all ONNX tooling. Packed diagonals would need to be stored as fake `uint8` tensors. Structured metadata (`CKKSParams`, `KeyManifest`) becomes string key-value pairs with no type safety. The 2GB protobuf limit requires external data management. No FHE framework uses ONNX as its compiled output format.

**Protobuf** — adds a third-party dependency and a `protoc` build step to handle ~10KB of structured metadata. The heavy data (blobs) is opaque `bytes` in protobuf anyway. JSON handles the metadata equally well with zero dependencies.

**FlatBuffers** — zero-copy deserialization is appealing for large blobs, but the evaluator CKKS-encodes all diagonals at load time anyway, so zero-copy provides no benefit. The data is consumed once, not read repeatedly.

The JSON + binary blob format requires zero dependencies in any language, is human-inspectable (`jq .graph.nodes model.orion.json`), and is already the pattern used in the current codebase.

### Go types for the header

```go
type CompiledHeader struct {
    Version    int            `json:"version"`
    Params     HeaderParams   `json:"params"`
    Config     HeaderConfig   `json:"config"`
    Manifest   HeaderManifest `json:"manifest"`
    InputLevel int            `json:"input_level"`
    Cost       HeaderCost     `json:"cost"`       // informational, not validated by evaluator
    Graph      HeaderGraph    `json:"graph"`
    BlobCount  int            `json:"blob_count"`
}

type HeaderGraph struct {
    Input  string       `json:"input"`
    Output string       `json:"output"`
    Nodes  []HeaderNode `json:"nodes"`
    Edges  []HeaderEdge `json:"edges"`
}

type HeaderNode struct {
    Name     string            `json:"name"`
    Op       string            `json:"op"`
    Level    int               `json:"level"`
    Depth    int               `json:"depth"`
    Shape    map[string][]int  `json:"shape"`
    Config   json.RawMessage   `json:"config"`
    BlobRefs map[string]int    `json:"blob_refs"`
}

type HeaderEdge struct {
    Src string `json:"src"`
    Dst string `json:"dst"`
}
```

Op-specific configs are `json.RawMessage`, parsed by a type switch when loading:

```go
switch node.Op {
case "linear_transform":
    var cfg LinearTransformConfig
    if err := json.Unmarshal(node.ConfigRaw, &cfg); err != nil { ... }
case "polynomial":
    var cfg PolynomialConfig
    if err := json.Unmarshal(node.ConfigRaw, &cfg); err != nil { ... }
case "bootstrap":
    var cfg BootstrapConfig
    if err := json.Unmarshal(node.ConfigRaw, &cfg); err != nil { ... }
// ...
default:
    return fmt.Errorf("unknown op: %s", node.Op)
}
```

## Repo Structure

Monorepo. Go module at root (`github.com/baahl-nyu/orion`). One Go package — `evaluator`, plus shared types at root. Three Python packages — `lattigo` (Go bridge `.so` + Python FFI), `orion-compiler` (torch + compilation), and `orion-evaluator` (Go evaluator bridge `.so`). No `orion-client` — users use Lattigo directly. The evaluator, Python bridge, and JS bridge each import Lattigo directly — no shared Go wrapper package.

```
orion/
├── go.mod                          # module github.com/baahl-nyu/orion
├── params.go, keys.go, ...         # Shared types: Params, Manifest, Polynomial (used by evaluator + bridges)
│
├── evaluator/                      # Go pkg: Orion FHE inference engine
│                                   #   imports Lattigo directly
│
├── python/
│   ├── lattigo/                    # pip install lattigo (ships bridge .so)
│   │   ├── bridge/                 #   Go CGO bridge (builds .so, bundled into package)
│   │   │                           #   imports Lattigo directly
│   │   └── lattigo/                #   Python pkg: ctypes bindings, GoHandle RAII
│   │
│   ├── orion-compiler/             # pip install orion-compiler (depends on lattigo)
│   │   └── orion_compiler/         #   Compiler, nn/, core/, CKKSParams, CompiledModel
│   │
│   ├── orion-evaluator/            # pip install orion-evaluator (ships bridge .so)
│   │   ├── bridge/                 #   Go CGO bridge wrapping evaluator/ pkg
│   │   │                           #   imports evaluator/ and Lattigo directly
│   │   └── orion_evaluator/        #   Python pkg: Model, Evaluator (ctypes over bridge)
│   │
│   └── tests/
│
├── js/
│   ├── lattigo/                    # npm package: Lattigo WASM bindings
│   │   ├── bridge/                 #   Go WASM bridge (builds .wasm), imports Lattigo directly
│   │   └── src/                    #   TypeScript wrappers over WASM
│   │
│   └── examples/                   # JS example code (tensor-to-slot mapping is user code)
│
├── examples/
│   └── wasm-demo/                  # Browser demo: Go server + WASM client
│   # Phase 5 (planned): mlp/, lenet/, lola/, alexnet/, vgg/, resnet/, yolo/
│
└── docs/
```

**`evaluator/`** (`github.com/baahl-nyu/orion/evaluator`) — the only Orion-specific Go package. Reads `.orion` files, CKKS-encodes diagonals at load time, walks the computation graph. Imports Lattigo directly.

**`lattigo`** (`pip install lattigo`) — Python package shipping the Go bridge `.so`. Contains all ctypes bindings (`ffi.py`, `GoHandle`). Not Orion-specific — exposes Lattigo's keygen, encrypt, decrypt, encode, decode, marshal/unmarshal, polynomial generation, parameter construction. The only Python package requiring Go/CGO to build from source (pre-built wheels eliminate this).

**`orion-compiler`** (`pip install orion-compiler`) — Python + Go bridge dependency. Compiler, nn modules, core algorithms, `CKKSParams`, `CompiledModel`. Calls `lattigo` for polynomial generation and parameter validation.

**`orion-evaluator`** (`pip install orion-evaluator`) — Python package shipping a Go bridge `.so` that wraps the `evaluator/` package. Exposes `Model` (load `.orion`, CKKS-encode at load time) and `Evaluator` (forward pass with Lattigo-serialized eval keys and ciphertexts). Requires Go/CGO to build from source (like `lattigo`). Enables self-contained Python examples without a separate Go server.

**`js/lattigo/`** — npm package shipping the WASM bridge. Same pattern as Python: Go bridge builds `.wasm`, TypeScript wrappers expose Lattigo ops. Imports Lattigo directly.

## Dependencies

| Component         | Go import                              | `pip install`     | Depends on                 | Does NOT depend on        |
| ----------------- | -------------------------------------- | ----------------- | -------------------------- | ------------------------- |
| `evaluator/`      | `github.com/baahl-nyu/orion/evaluator` | —                 | Lattigo                    | Python, compiler, bridges |
| `lattigo`         | (CGO, builds .so)                      | `lattigo`         | Lattigo (Go)               | Anything Orion            |
| `orion-compiler`  | —                                      | `orion-compiler`  | `lattigo`, torch, networkx | `evaluator/`              |
| `orion-evaluator` | (CGO, builds .so)                      | `orion-evaluator` | `evaluator/`, Lattigo      | torch, compiler           |
| `js/lattigo/`     | (WASM, builds .wasm)                   | —                 | Lattigo                    | `evaluator/`              |

No circular dependencies. No shared Go packages. `lattigo` and `orion-evaluator` are the packages with Go code. `orion-compiler` depends on `lattigo` for compile-time operations. `orion-evaluator` has no Python package dependencies.

## Build and Install

Source-only builds for now. Pre-built wheels with bundled Go shared library are a natural next step.

**Lattigo bindings:** `cd python/lattigo && pip install -e .` — triggers Go compilation of `bridge/` into a shared library, bundled into the package.

**Orion compiler:** `cd python/orion-compiler && pip install -e .` — pulls in `lattigo` + torch.

**JS/WASM:** `python tools/build_lattigo_wasm.py` compiles `js/lattigo/bridge/` to `js/lattigo/wasm/lattigo.wasm` (~8 MB uncompressed, ~3 MB gzipped) and copies `wasm_exec.js` from the Go distribution. Then `cd js/lattigo && npm install && npm run build` builds the TypeScript wrappers to `dist/`.

**Browser demo:** See `examples/wasm-demo/`. Steps: (1) generate model: `cd examples/wasm-demo && python generate_model.py`, (2) build client: `cd examples/wasm-demo/client && npm install && npm run build`, (3) run server: `cd examples/wasm-demo/server && go run . ../model.orion`, (4) open http://localhost:8080.

**Orion evaluator (Python bindings):** `cd python/orion-evaluator && pip install -e .` — triggers Go compilation of `bridge/` into a shared library wrapping the `evaluator/` package. Requires the `evaluator/` Go package (repo root).

**Go Evaluator:** Standard `go build`. Users import `"github.com/baahl-nyu/orion/evaluator"` in their server code.

## Testing

- **Go:** `_test.go` files alongside source (Go convention). `go test ./...`
- **Python:** `python/tests/` with pytest. All tests require the Go shared lib (compiler uses Lattigo).
- **JS/WASM:** `cd js/lattigo && npm test` — vitest integration tests exercising TypeScript wrappers via WASM. Requires the WASM binary (`python tools/build_lattigo_wasm.py`) to be built first.
- **E2E / cross-language:** Runnable examples in `examples/` that validate correctness. Compiler (Python) produces `.orion` → evaluator (Go) consumes it → numerical results compared against cleartext PyTorch output.

Python-side evaluation uses `orion-evaluator` bindings (Go evaluator via CGO bridge). Self-contained Python E2E tests: compile → encrypt → evaluate (via bindings) → decrypt → compare to cleartext.

## Roadmap

Four phases. No backward compatibility — the Python evaluator is deleted in Phase 1, the project is broken until the Go evaluator lands in Phase 2. A cleartext graph validator provides testing coverage in between.

### Phase 1: New compiled model format

The `.orion` v2 format is the contract between compiler and evaluator. Everything else builds on it.

#### 1.1 Rewrite `CompiledModel` dataclass

**File:** `orion_compiler/compiled_model.py`

The v2 format replaces the v1 flat `topology` + `module_metadata` + Lattigo-serialized blobs with a `graph` (nodes + edges) + `cost` profile + raw float64 blobs. V2 `CompiledModel` fields:

```python
@dataclass
class CostProfile:
    bootstrap_count: int
    galois_key_count: int       # = len(manifest.galois_elements)
    bootstrap_key_count: int    # = len(manifest.bootstrap_slots)
    # rotation_count deferred — requires counting actual eval-time rotations per node

@dataclass
class Graph:
    input: str              # name of input node
    output: str             # name of output node
    nodes: list[GraphNode]
    edges: list[GraphEdge]

@dataclass
class GraphNode:
    name: str
    op: str                 # one of the op types below
    level: int
    depth: int
    shape: dict | None      # {"input": [...], "output": [...], "fhe_input": [...], "fhe_output": [...]}
    config: dict            # op-specific, see table below
    blob_refs: dict[str, int] | None  # key -> blob index

@dataclass
class GraphEdge:
    src: str
    dst: str
```

Change serialization magic from `ORMDL\x00\x01\x00` to `ORION\x00\x02\x00`. The `to_bytes()` / `from_bytes()` methods emit/parse the JSON header structure defined in the "Compiled Model Format" section above. The `_pack_container` / `_unpack_container` helpers stay — same binary container pattern, new magic and JSON structure.

`EvalKeys` and `KeyManifest` are unchanged.

#### 1.2 Define node op types and their configs

Every node in `graph.nodes` has an `op` string and an op-specific `config` dict. Complete enumeration:

| `op`               | `config` keys                                                                                                                             | `blob_refs` keys                           | Notes                                                                            |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------- |
| `flatten`          | _(empty)_                                                                                                                                 | _(none)_                                   | Shape-only, no computation                                                       |
| `linear_transform` | `bsgs_ratio` (float), `output_rotations` (int)                                                                                            | `diag_{row}_{col}` (one per block), `bias` | Includes Linear, Conv2d, AvgPool2d                                               |
| `quad`             | _(empty)_                                                                                                                                 | _(none)_                                   | x² + rescale. `depth: 1`                                                         |
| `polynomial`       | `coeffs` (list[float]), `basis` (`"monomial"` or `"chebyshev"`), `prescale` (float), `postscale` (float), `constant` (float)              | _(none)_                                   | Coefficients inline (degree ≤ 63). Covers Sigmoid, SiLU, GELU, generic Chebyshev |
| `bootstrap`        | `input_level` (int), `input_min` (float), `input_max` (float), `prescale` (float), `postscale` (float), `constant` (float), `slots` (int) | _(none)_                                   | `slots` = 2^ceil(log2(fhe_input elements))                                       |
| `add`              | _(empty)_                                                                                                                                 | _(none)_                                   | Two incoming edges (residual connection)                                         |
| `mult`             | _(empty)_                                                                                                                                 | _(none)_                                   | Two incoming edges                                                               |

All nodes carry `level` (int) and `depth` (int) as top-level fields. `shape` is required for `linear_transform` and `bootstrap`, optional for others.

**ReLU deviation:** There is no single `relu` op type. ReLU traces to its sub-components via PyTorch FX: `mult` -> `polynomial` x N -> `mult`. Each sub-component is a separate graph node with its own level/depth. Chebyshev coefficients for the minimax sign polynomials are stored inline on each `polynomial` node's `config`. The Go evaluator walks these nodes individually — no special `evalReLU` handler needed.

**Fused batch norms:** Not present in the graph. The compiler calls `remove_fused_batchnorms()` before graph construction, folding batch norm weights into the preceding `linear_transform`. Unfused batch norms are not supported.

**Fork/join filtering:** The compiler's `NetworkDAG` uses auxiliary `fork` and `join` nodes for residual connections (needed by the bootstrap solver). These are filtered out during edge extraction: `A -> fork -> B, C` becomes `A -> B` and `A -> C`; `A, B -> join -> C` becomes `A -> C` and `B -> C`. Only real operation nodes appear in the final graph.

#### 1.3 Raw diagonal blob format

Implement `pack_raw_diagonals()` and `unpack_raw_diagonals()` helpers.

**Input:** one block of diagonals — `dict[int, list[float]]` mapping diagonal index to padded float64 values (length = `max_slots`). This is one `(row, col)` entry from `module.diagonals`.

**Output blob:**

```
[4]                          NUM_DIAGS (uint32 LE)
[NUM_DIAGS × 4]              DIAG_INDICES (int32 LE, sorted ascending)
[NUM_DIAGS × max_slots × 8]  VALUES (float64 LE, IEEE 754)
```

Fixed stride: the evaluator can seek to diagonal `i` at offset `4 + NUM_DIAGS*4 + i*max_slots*8`.

**Bias blobs:** raw float64 array, length = `max_slots`, zero-padded. No header — the evaluator knows the slot count from params.

#### 1.4 Modify compiler to emit v2 format

**File:** `orion/compiler.py`, method `Compiler.compile()`

Changes to the compilation loop:

1. **Emit graph edges.** After `network_dag.build_dag()` and all processing (fusion, bootstrap placement), extract edges from the DAG, filtering out fork/join auxiliary nodes (re-linking `A -> fork -> B` as `A -> B`, `A -> join -> B` as `A -> B`). Serialize as `[{"src": u, "dst": v}, ...]`.

2. **Insert bootstrap nodes into the graph.** Currently, bootstraps are forward hooks attached to modules — they don't exist as DAG nodes. Change `BootstrapPlacer` (or post-process the DAG) to insert explicit `bootstrap` nodes with edges. If bootstrap is placed after `act1` and `act1` → `fc2`:
   - Remove edge `act1` → `fc2`
   - Add node `boot_0` (op: `bootstrap`)
   - Add edges `act1` → `boot_0` → `fc2`

3. **Store raw diagonals instead of Lattigo blobs.** Replace CKKS-encoded Lattigo serialization with raw float64 diagonals:

   ```python
   # New: store raw float64 diagonals, compute Galois elements in pure Python
   module.generate_diagonals(last=...)  # populates module.diagonals
   # Galois elements computed via pure Python BSGS (orion.core.galois) — no Go calls:
   galois_elems = compute_galois_elements_for_linear_transform(
       diag_indices_per_block, slots, bsgs_ratio, logn, ring_type)
   # Serialize raw diags:
   for (row, col), diags_block in module.diagonals.items():
       blob = pack_raw_diagonals(diags_block, max_slots)
       blob_refs[f"diag_{row}_{col}"] = len(blobs)
       blobs.append(blob)
   ```

   The `compile()` step has **zero Go/Lattigo dependency** — only `fit()` still needs Go (for minimax polynomial generation in ReLU).

4. **Emit node metadata in `GraphNode` format.** Replace the current `_extract_module_metadata()` per-type dicts with unified `GraphNode` construction. Map module classes to op strings:

   | Module class                           | `op` string                         |
   | -------------------------------------- | ----------------------------------- |
   | `Linear`, `Conv2d`, `AvgPool2d`        | `linear_transform`                  |
   | `Quad`                                 | `quad`                              |
   | `Activation` (monomial)                | `polynomial` (basis: `"monomial"`)  |
   | `Chebyshev`, `Sigmoid`, `SiLU`, `GELU` | `polynomial` (basis: `"chebyshev"`) |
   | `Bootstrap`                            | `bootstrap`                         |
   | `Add`                                  | `add`                               |
   | `Mult`                                 | `mult`                              |
   | `Flatten`                              | `flatten`                           |

   Note: `ReLU` decomposes into sub-nodes — see "ReLU deviation" in 1.2 above. `BatchNorm1d`/`BatchNorm2d` are fused into the preceding `linear_transform` and excluded from the graph.

5. **Compute cost profile.** Count bootstrap count and key counts (Galois keys = `len(manifest.galois_elements)`, bootstrap keys = `len(manifest.bootstrap_slots)`). Rotation count is deferred — it requires counting actual eval-time rotations per node. The cost profile is informational metadata for the user — the evaluator does not validate it.

6. **Determine graph input/output.** The first node in topological order with no predecessors is `graph.input`. The last node with no successors is `graph.output`.

7. **ReLU handling.** ReLU decomposes into sub-nodes — see "ReLU deviation" in 1.2 above.

#### 1.5 Delete Python evaluator

**Delete:** `orion/evaluator.py`

**Edit:** `orion/__init__.py` — remove `Evaluator` import and export.

The FHE forward paths in nn modules (`if self.he_mode` branches in `Linear.forward()`, `Quad.forward()`, etc.) become dead code. Leave them for now — they'll be stripped in Phase 3.

#### 1.6 Cleartext graph validator

**New file:** `tests/test_compiled_format.py`

Test that validates the `.orion` v2 format without any CKKS operations:

1. **Structural tests:**
   - `CompiledModel.to_bytes()` → `CompiledModel.from_bytes()` roundtrip preserves all fields
   - All `blob_refs` point to valid blob indices (0 ≤ idx < len(blobs))
   - All edge `src`/`dst` reference existing node names
   - `graph.input` and `graph.output` exist in node list
   - Topological sort of edges is acyclic
   - Every non-input node has at least one incoming edge
   - `add`/`mult` nodes have exactly two incoming edges

2. **Numerical tests** (compile an MLP, walk the graph in cleartext):
   - For each `linear_transform` node: unpack raw diags from blob, reconstruct dense weight matrix from diagonals (inverse of diagonal extraction), do `numpy.matmul(W, x) + bias`
   - For each `quad` node: `x = x * x`
   - For each `polynomial` node: evaluate polynomial using numpy (Horner's method for monomial, Clenshaw for Chebyshev)
   - Compare final output against `net(x).detach().numpy()` — tolerance ≤ 1e-10 (cleartext, no FHE noise)

3. **Blob format tests:**
   - `pack_raw_diagonals()` → `unpack_raw_diagonals()` roundtrip
   - Diagonal indices sorted ascending
   - Each diagonal has exactly `max_slots` values
   - Bias blob length = `max_slots * 8` bytes

#### 1.7 Update existing tests

Tests that import or use `Evaluator` will fail. Mark them `@pytest.mark.skip(reason="Python evaluator removed — Phase 2 provides Go evaluator")`. Keep all `Compiler` and `Client` tests running.

#### Phase 1 acceptance checklist

- [x] `CompiledModel` uses magic `ORION\x00\x02\x00` and version 2
- [x] JSON header contains `graph` with `nodes`, `edges`, `input`, `output` (no `topology` or `modules` keys)
- [x] JSON header contains `cost` profile
- [x] All `linear_transform` blobs contain raw float64 diagonals, not Lattigo-serialized data
- [x] Bias blobs are raw float64 arrays
- [x] Polynomial coefficients are inline in node `config` (no blobs)
- [x] ReLU decomposed into sub-nodes (`mult`, `polynomial` x N, `mult`) — no single `relu` op
- [x] Bootstrap nodes appear as explicit graph nodes with edges (not hook metadata)
- [x] Fused batch norms absent from graph (removed by `remove_fused_batchnorms()`, weights folded into preceding linear)
- [x] `orion/evaluator.py` deleted, `Evaluator` removed from `orion/__init__.py`
- [x] `CompiledModel.to_bytes()` → `from_bytes()` roundtrip passes
- [x] Cleartext graph validator passes for MLP (compile → read → numpy walk → compare to PyTorch)
- [x] All non-evaluator tests pass (`pytest tests/ -k "not evaluator"` or equivalent)
- [x] `.orion` file size is ~6× smaller than v1 for the same model (raw vs CKKS-encoded diagonals)

---

### Phase 2: Pure Go evaluator

Reads `.orion` v2, walks the computation graph, runs FHE inference. No Python, no skeleton network. Built against `baahl-nyu/lattigo` (current fork). Migration to upstream `tuneinsight/lattigo` deferred until after the refactoring is complete.

#### 2.1 Binary container reader

**New file:** `evaluator/format.go`

Implement the binary container parser in Go:

1. Verify magic bytes (`ORION\x00\x02\x00`)
2. Read header length (uint32 LE), parse JSON header into Go structs (types already defined in "Go types for the header" section above)
3. Read blob count (uint32 LE), read each blob (uint64 LE length + data)

Implement `ParseDiagonalBlob(data []byte, maxSlots int) (map[int][]float64, error)` — reads the fixed-stride diagonal format into `diagIndex → []float64`.

Implement `ParseBiasBlob(data []byte, maxSlots int) ([]float64, error)` — reads raw float64 array.

#### 2.2 `Model` type

**New file:** `evaluator/model.go`

```go
type Model struct {
    header      *CompiledHeader
    clientParam orion.Params                                         // cached for ClientParams()
    params      ckks.Parameters
    graph       *Graph                                               // processed graph with adjacency + topo order
    transforms  map[string]map[string]lintrans.LinearTransformation  // node_name -> {"diag_0_0": LT, ...}
    biases      map[string]*rlwe.Plaintext                           // node_name -> CKKS-encoded bias
    polys       map[string]bignum.Polynomial                         // node_name -> Lattigo polynomial
    ltConfigs   map[string]*LinearTransformConfig                    // node_name -> parsed LT config
    polyConfigs map[string]*PolynomialConfig                         // node_name -> parsed poly config
}
```

`LoadModel(data []byte) (*Model, error)`:

1. Parse binary container → `CompiledHeader` + blobs
2. Build `ckks.Parameters` from header params
3. Create a temporary `ckks.Encoder` for encoding (no keys needed)
4. For each node where `op == "linear_transform"`:
   - For each blob*ref `diag*{row}\_{col}`→ parse diagonal blob → call`lintrans.NewLinearTransformation(params, diagMap, level, bsgsRatio)` or the equivalent Lattigo constructor to CKKS-encode
   - For blob_ref `bias` → parse bias blob → create `ckks.NewPlaintext(params, level-depth)` with scale `rlwe.NewScale(params.DefaultScale())` → `encoder.Encode(biasVec, pt)` → store `*rlwe.Plaintext`. The bias scale must match the ciphertext scale after LT + rescale (`DefaultScale()` ≈ `2^logscale`).
5. For each node where `op == "polynomial"`:
   - Read `config.coeffs` and `config.basis`
   - If `"chebyshev"`: `bignum.NewPolynomial(bignum.Chebyshev, coeffs, nil)`
   - If `"monomial"`: `bignum.NewPolynomial(bignum.Monomial, coeffs, nil)`
6. Build `*Graph`: compute topological sort, build adjacency map (`nodeName → []inputNodeNames`)
7. Validate: all blob_refs resolved, all edge endpoints exist, graph is acyclic

`ClientParams() (orion.Params, orion.Manifest, int)` — returns params, manifest, and input level for the client.

The `Model` is immutable after `LoadModel()` returns. Safe to share across goroutines.

#### 2.3 Graph representation

**New file:** `evaluator/graph.go`

```go
type Graph struct {
    Input    string
    Output   string
    Nodes    map[string]*Node       // name -> node
    Order    []string               // topological sort
    Inputs   map[string][]string    // node_name -> ordered list of input node names
}
```

`buildGraph(header *CompiledHeader) (*Graph, error)`:

- Index nodes by name
- Build reverse adjacency from edges: for each edge `{src, dst}`, append `src` to `Inputs[dst]`
- Compute topological sort via Kahn's algorithm (edges already available)
- Validate: `Input` node has no incoming edges, `Output` node has no outgoing edges, exactly one connected component

For nodes with multiple inputs (`add`, `mult`, residual joins): `Inputs[name]` has length ≥ 2. The order of inputs matters — it's the order edges were listed in the JSON.

#### 2.4 `Evaluator` type and `Forward()`

**New file:** `evaluator/evaluator.go`

```go
type Evaluator struct {
    params   ckks.Parameters
    encoder  *ckks.Encoder
    eval     *ckks.Evaluator
    linEval  *lintrans.Evaluator
    polyEval *polynomial.Evaluator
}

func NewEvaluatorFromKeySet(ckksParams ckks.Parameters, keys *rlwe.MemEvaluationKeySet) (*Evaluator, error)
func (e *Evaluator) Forward(model *Model, input *rlwe.Ciphertext) (*rlwe.Ciphertext, error)
func (e *Evaluator) Close()
```

The `Evaluator` is created directly from Lattigo types — no intermediate serialization bundle. It holds per-client evaluation keys. No secret key.

An `Evaluator` is **not goroutine-safe** — Lattigo evaluators carry internal NTT/decomposition buffers that are reused across operations. Use one `Evaluator` per goroutine, or protect `Forward()` with a mutex.

`Forward()` walks `model.graph.Order`:

```go
func (e *Evaluator) Forward(model *Model, input *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
    results := make(map[string]*rlwe.Ciphertext)
    results[model.graph.Input] = input

    for _, name := range model.graph.Order {
        if name == model.graph.Input {
            continue
        }
        node := model.graph.Nodes[name]
        // gather inputs from results map, dispatch to op-specific handler via switch
        out, err := e.dispatch(model, node, results)
        if err != nil {
            return nil, fmt.Errorf("node %s: %w", name, err)
        }
        results[name] = out
    }
    return results[model.graph.Output], nil
}
```

**Scope limitation:** Currently targets single-ciphertext-in, single-ciphertext-out models (MLP, LeNet, LoLA, small conv nets). Multi-ciphertext block-matrix routing (when weight matrices exceed the slot count in both dimensions) is deferred.

**Memory:** The `results` map keeps all intermediate ciphertexts alive until `Forward()` returns. Implementations may free intermediate results once all downstream consumers have been evaluated.

#### 2.5 Op implementations

Each op is a method on `Evaluator`. All underlying Lattigo calls are in `evaluator/evaluator.go`.

**`linear_transform`:**

```go
func (e *Evaluator) evalLinearTransform(model *Model, node *Node, ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error)
```

1. Look up pre-encoded `LinearTransformation` objects from `model.transforms[node.Name]`
2. For each `(row, col)` block: `linEval.EvaluateNew(ct, lt)` → accumulate with `evaluator.Add`
3. Rescale result: `evaluator.Rescale(ct)`
4. Add bias: `evaluator.Add(ct, model.biases[node.Name])`
5. If `config.output_rotations > 0`: apply hybrid output rotation (rotate and add `output_rotations` times, halving the rotation distance each time)

**`quad`:**

```go
func (e *Evaluator) evalQuad(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error)
```

1. `evaluator.MulRelinNew(ct, ct)` (ct × ct + relinearize)
2. `evaluator.Rescale(ct)`

**`polynomial`:**

```go
func (e *Evaluator) evalPolynomial(model *Model, node *Node, ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error)
```

1. Look up `model.polys[node.Name]`
2. If `config.prescale != 1`: `evaluator.MulScalar(ct, prescale)`, `evaluator.Rescale(ct)`. If `config.constant != 0`: `evaluator.AddScalar(ct, constant)`. Together these map the input into the polynomial's domain (e.g., [-1, 1] for Chebyshev). Skipped when `fuse_modules=true` (prescale/constant absorbed into preceding LT weights/bias).
3. `polyEval.Evaluate(ct, poly, targetScale)` (Lattigo's Paterson-Stockmeyer or baby-step-giant-step polynomial evaluator)
4. If `config.postscale != 1`: `evaluator.MulScalar(ct, postscale)`, `evaluator.Rescale(ct)`

**ReLU:** No `evalReLU` handler — ReLU decomposes into sub-nodes (see "ReLU deviation" in 1.2). Total depth per ReLU: `sum(ceil(log2(d+1)) for d in degrees) + 2`. For default degrees `[15, 15, 27]`: depth = 4 + 4 + 5 + 2 = **15 levels**.

**`bootstrap`** (not yet implemented — returns error):

```go
func (e *Evaluator) evalBootstrap(node *Node, ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error)
```

1. If `config.constant != 0`: `evaluator.AddScalar(ct, constant)`
2. Encode `prescale` as plaintext, `evaluator.Mul(ct, prescalePt)`, `evaluator.Rescale(ct)`
3. `bootstrappers[config.slots].Bootstrap(ct)` (Lattigo bootstrap)
4. If `config.postscale != 1`: `evaluator.MulScalar(ct, postscale)`, `evaluator.Rescale(ct)`
5. If `config.constant != 0`: `evaluator.AddScalar(ct, -constant)`

**`add`:**

```go
func (e *Evaluator) evalAdd(ct0, ct1 *rlwe.Ciphertext) (*rlwe.Ciphertext, error)
```

1. `evaluator.AddNew(ct0, ct1)`

**`mult`:**

```go
func (e *Evaluator) evalMult(ct0, ct1 *rlwe.Ciphertext) (*rlwe.Ciphertext, error)
```

1. `evaluator.MulRelinNew(ct0, ct1)`
2. `evaluator.Rescale(ct)`

**`flatten`:** no-op. Return input unchanged.

Note: `batch_norm` is never emitted as a graph node. The compiler fuses batch norm weights into the preceding `linear_transform` via `remove_fused_batchnorms()` before graph construction (see 1.4). `LoadModel()` returns an error if an unknown op type is encountered.

#### 2.6 E2E testing

**Test fixture generation** (Python script, run before `go test`):

1. Compile MLP → write `testdata/mlp.orion`
2. Generate random input, compute cleartext expected output → write `testdata/mlp.input.json` (float64 array), `testdata/mlp.expected.json` (float64 array)

Only `.orion` files and cleartext JSON cross the Python→Go boundary. Keys and ciphertexts use Lattigo's native `MarshalBinary`/`UnmarshalBinary` — no custom Orion serialization formats. Phase 3 deletes the Python `EvalKeys`/`ORKEY` format and `Ciphertext` shape header in favor of direct Lattigo serialization.

**Go test** (`evaluator/evaluator_test.go`):

1. `LoadModel("testdata/mlp.orion")` → model
2. `model.ClientParams()` → params, manifest, inputLevel
3. `params.NewCKKSParameters()` → ckksParams
4. `rlwe.NewKeyGenerator(ckksParams)` → generate sk, pk, rlk, Galois keys from manifest
5. `rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...)` → evk
6. `NewEvaluatorFromKeySet(ckksParams, evk)` → evaluator
7. Encode + encrypt via `ckks.Encoder` + `rlwe.Encryptor`
8. `eval.Forward(model, ct)` → result
9. Decrypt + decode via `rlwe.Decryptor` + `ckks.Encoder`, compare against `mlp.expected.json`

Repeat for at least: MLP (linear-only), model with Chebyshev activations, model with bootstrap (if feasible in test time).

#### Phase 2 acceptance checklist

- [x] `evaluator.LoadModel(data)` successfully parses `.orion` v2 files produced by the Python compiler
- [x] `model.ClientParams()` returns correct CKKS params, key manifest, and input level
- [x] `evaluator.NewEvaluatorFromKeySet(ckksParams, evk)` constructs from Lattigo types directly (keys generated via `rlwe.KeyGenerator`)
- [x] `eval.Forward(model, ct)` produces correct results for MLP (tolerance ≤ 0.02 — inherent CKKS noise with logscale=26 parameters yields ~0.01 max error; 1e-3 was overly optimistic for these params)
- [x] `eval.Forward()` handles: `linear_transform`, `quad`, `polynomial`, `flatten`. Deferred: `add`, `mult` have unit tests but no E2E model exercises them; `bootstrap` deferred to future work
- [x] Multiple `Evaluator` instances sharing one `Model` work correctly (concurrent reads)
- [x] `Evaluator.Close()` releases resources (nils evaluator fields); calling `Forward()` on a closed evaluator returns an error
- [x] All methods return `(result, error)` — no panics on malformed input
- [x] `go test ./evaluator/...` passes (43 tests)
- [x] E2E test: Python compile → Go keygen + encrypt → Go evaluate → Go decrypt → correct output (MLP + Sigmoid models)

---

### Phase 3: Package restructuring

Split the monolith into three packages: `lattigo` (Python Lattigo bindings), `orion-compiler` (Python), and `orion-evaluator` (Python bindings to Go evaluator). No `orion-client` package — users interact with Lattigo directly for keygen, encrypt, decrypt. Can start after Phase 1, parallelizable with Phase 2 (except moving the Go evaluator).

#### 3.1 Extract `python/lattigo/`

Create `python/lattigo/` with its own `pyproject.toml` (`pip install lattigo`).

**Move into `python/lattigo/`:**

| From                                     | To                              |
| ---------------------------------------- | ------------------------------- |
| `orionclient/bridge/`                    | `python/lattigo/bridge/`        |
| `orion/backend/orionclient/ffi.py`       | `python/lattigo/lattigo/ffi.py` |
| `GoHandle` class and all ctypes bindings | `python/lattigo/lattigo/`       |

**Bridge API surface after cleanup.** With no Python evaluator and no `Client` class (users use Lattigo directly), the bridge retains only Lattigo primitives and compile-time helpers:

| Keep (Lattigo primitives + compile-time)                                                       | Remove                                                                                   |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `NewKeyGenerator`, `GenSecretKey`, `GenRelinearizationKey`, `GenGaloisKey`, `GenBootstrapKeys` | All `Eval*` functions (`EvalAdd`, `EvalSub`, `EvalMul`, `EvalPoly`, etc.)                |
| `NewEncoder`, `Encode`, `Decode`                                                               | `NewEvaluator`, `EvalClose`                                                              |
| `NewEncryptor`, `Encrypt`, `NewDecryptor`, `Decrypt`                                           | `NewEvalKeyBundle` and all `EvalKeyBundle*` setters                                      |
| `NewParameters` (from JSON)                                                                    | `NewClient`, `ClientClose`, `ClientEncode`, `ClientDecode`, `ClientEncrypt`, etc.        |
| `MarshalBinary`, `UnmarshalBinary` (for all Lattigo types)                                     | `GenerateLinearTransformFromParams`, `LinearTransformRequiredGaloisElements` (dead code) |
| `NewMemEvaluationKeySet`                                                                       | `LinearTransformMarshal`, `LinearTransformUnmarshal`                                     |
| `GeneratePolynomialMonomial`, `GeneratePolynomialChebyshev`, `GenerateMinimaxSignCoeffs`       | `CombineSingleCiphertexts` and multi-ciphertext wrappers                                 |
| `DeleteHandle`                                                                                 |                                                                                          |

Note: `GenerateLinearTransformFromParams` and `LinearTransformRequiredGaloisElements` are confirmed dead code — `fit()` and `compile()` never call them (verified experimentally in Phase 3 planning). `Client*` functions are removed because users use Lattigo primitives directly, enabling threshold encryption and custom key management.

The bridge Go code (`bridge/*.go`) is updated to remove the deleted exports and to expose Lattigo primitives directly (KeyGenerator, Encoder, Encryptor, Decryptor) instead of the old `Client` abstraction. The shared library shrinks significantly.

#### 3.2 Extract `python/orion-compiler/`

Create `python/orion-compiler/` with its own `pyproject.toml` (`pip install orion-compiler`, depends on `lattigo`, `torch`, `networkx`).

**Move into `python/orion-compiler/orion_compiler/`:**

| From                      | To                                 |
| ------------------------- | ---------------------------------- |
| `orion/compiler.py`       | `orion_compiler/compiler.py`       |
| `orion/nn/`               | `orion_compiler/nn/`               |
| `orion/core/`             | `orion_compiler/core/`             |
| `orion/params.py`         | `orion_compiler/params.py`         |
| `orion/compiled_model.py` | `orion_compiler/compiled_model.py` |

`CKKSParams`, `CompiledModel`, `KeyManifest`, and `CompilerConfig` live in `orion_compiler` — they are compiler output types. The compiler imports `lattigo` for compile-time Go bridge operations (polynomial generation, parameter validation).

**Deleted (not moved):**

- `orion/client.py` — replaced by direct Lattigo usage
- `orion/ciphertext.py` — replaced by direct Lattigo usage (after dead code removal in 3.3)
- `orion/backend/` — FFI code moves to `python/lattigo/`

#### 3.3 Clean up dead code

With the Python evaluator gone (Phase 1), `Client` class deleted, and packages split:

1. **Strip FHE forward paths from nn modules.** Remove the `if self.he_mode` branches in `forward()` methods of `Linear`, `Conv2d`, `Quad`, `Activation`, `Chebyshev`, `Bootstrap`, `Add`, `Mult`, `Flatten`, `BatchNorm`. These modules become compile-only: they have `fit(context)` and cleartext `forward()`.

2. **Remove `he()` toggle from `Module`.** The `he_mode` flag, `he()` method, and `eval()`/`train()` overrides are dead. Modules always run in cleartext mode (for tracing and fitting).

3. **Delete `orion/client.py` entirely.** Users use Lattigo bindings directly for keygen, encode, encrypt, decrypt, decode.

4. **Delete `orion/ciphertext.py` entirely.** The `Ciphertext` and `PlainText` Python wrapper classes (with `context`, dead arithmetic methods, custom serialization with redundant shape header) are replaced by direct Lattigo usage. Users work with Lattigo's native ciphertext/plaintext types via the `lattigo` Python bindings.

5. **Delete `EvalKeys` class and `ORKEY` container format.** Users serialize keys using Lattigo's native `MemEvaluationKeySet.MarshalBinary()`. No custom key serialization format.

6. **Remove `_EvalContext` and `Evaluator._reconstruct_*` code.** Already deleted in Phase 1 (evaluator.py), but verify no remnants in other files.

7. **Remove dead `LinearTransform.compile()` and `LinearTransform.evaluate_transforms()` methods.** Confirmed never called in the standard pipeline (see Phase 3 investigation).

8. **Delete `TransformEncoder` class** from `compiler_backend.py` and `Compiler._lt_evaluator` / `ctx.lt_evaluator`. Confirmed dead code — pure Python BSGS replaced these.

#### 3.4 Restructure Go modules

After Phase 2 delivers the `evaluator/` package:

- Root `go.mod`: `module github.com/baahl-nyu/orion` (single module for evaluator + shared types)
- `evaluator/` becomes a package under the root module
- `python/lattigo/bridge/` retains its own `go.mod` (CGO shared library, separate build)
- Move shared types (`Params`, `Manifest`, etc.) from `orionclient/` to root-level package
- Delete `orionclient/` (bridge moves to `python/lattigo/bridge/`, client logic deleted — users use Lattigo directly)

#### 3.5 Create `python/orion-evaluator/`

After Phase 2 (Go evaluator exists) and 3.4 (Go modules restructured).

**New directory:** `python/orion-evaluator/`

**Bridge (`python/orion-evaluator/bridge/`):**

A separate CGO shared library that wraps the `evaluator/` Go package. Exports:

| C export                                                     | Go implementation                                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `EvalLoadModel(data, len) → handle`                          | `evaluator.LoadModel(data) → *Model`                                                        |
| `EvalModelClientParams(h) → paramsJSON, manifestJSON, level` | `model.ClientParams()` → JSON with Params, Manifest, inputLevel                             |
| `EvalModelClose(h)`                                          | release handle (no-op, Model is immutable)                                                  |
| `EvalNewEvaluator(paramsJSON, keysData, keysLen) → handle`   | parse params → `NewCKKSParameters()` → `UnmarshalBinary(keys)` → `NewEvaluatorFromKeySet()` |
| `EvalForward(evalH, modelH, ctData, ctLen) → ctOut`          | `rlwe.Ciphertext.UnmarshalBinary()` → `eval.Forward()` → `MarshalBinary()`                  |
| `EvalClose(h)`                                               | `eval.Close()`, release handle                                                              |

The evaluator bridge accepts raw Lattigo `MarshalBinary` bytes for both keys and ciphertexts. Keys are `MemEvaluationKeySet.MarshalBinary()` output. Ciphertexts are Lattigo `rlwe.Ciphertext.MarshalBinary()` output. No Orion-specific serialization formats.

All handles are `cgo.Handle` values, same pattern as `python/lattigo/bridge/`. Error propagation via `errOut` parameter.

The bridge has its own `go.mod` that imports both `github.com/baahl-nyu/orion/evaluator` and Lattigo. It builds a platform-specific `.so`/`.dylib`/`.dll`.

**Python package (`python/orion-evaluator/orion_evaluator/`):**

```python
class Model:
    """Loaded .orion model. Immutable, shareable across Evaluators."""

    @classmethod
    def load(cls, data: bytes) -> "Model":
        """Parse .orion v2 and CKKS-encode diagonals (one-time cost)."""

    def client_params(self) -> tuple[dict, dict, int]:
        """Returns (params_dict, manifest_dict, input_level)."""

    def close(self) -> None: ...

class Evaluator:
    """Per-client evaluator. Holds eval keys. Not thread-safe."""

    def __init__(self, params: dict, keys_bytes: bytes) -> None:
        """Create evaluator from CKKS params dict and Lattigo MemEvaluationKeySet.MarshalBinary() bytes."""

    def forward(self, model: Model, ct_bytes: bytes) -> bytes:
        """Run FHE inference. Accepts and returns Lattigo ciphertext MarshalBinary bytes."""

    def close(self) -> None: ...
```

Both classes wrap `GoHandle` with the standard RAII pattern (idempotent `close()`, `__del__` fallback).

**`pyproject.toml`:** `pip install orion-evaluator`. No Python package dependencies (uses its own CGO bridge). Build system triggers Go compilation of `bridge/`.

#### Phase 3 acceptance checklist

- [x] `cd python/lattigo && pip install -e .` succeeds, builds `.so`
- [x] `cd python/orion-compiler && pip install -e .` succeeds (depends on lattigo, torch, networkx)
- [x] `cd python/orion-evaluator && pip install -e .` succeeds, builds `.so`
- [x] `from lattigo import ckks, rlwe` works (Lattigo primitives directly accessible)
- [x] `from orion_compiler import Compiler, CKKSParams, CompiledModel` works
- [x] `from orion_evaluator import Model, Evaluator` works
- [x] Compiler E2E: compile MLP → produces valid `.orion` v2
- [x] Lattigo E2E: keygen + encode + encrypt/decrypt roundtrip using Lattigo bindings directly
- [x] Python E2E: compile → encrypt (Lattigo) → evaluate (orion-evaluator) → decrypt (Lattigo) → correct output
- [x] Evaluator accepts `MemEvaluationKeySet.MarshalBinary()` bytes for keys
- [x] Evaluator accepts/returns `rlwe.Ciphertext.MarshalBinary()` bytes for ciphertexts
- [x] Lattigo bridge `.so` does not export `Client*` or `Eval*` functions
- [x] No `he_mode`, `he()`, or FHE forward branches in nn modules
- [x] `orion/client.py`, `orion/ciphertext.py` deleted (no `Client`, `Ciphertext`, `PlainText` classes)
- [x] `EvalKeys` class and `ORKEY` format deleted
- [x] `orionclient/` directory deleted
- [x] `go.mod` at repo root, `go test ./evaluator/...` passes
- [x] No circular imports between packages
- [x] `python/tests/` pass with the new package structure

---

### Phase 4: JS/WASM bindings

Starts after Phases 1–3 are stable.

#### 4.1 `js/lattigo/` — WASM Lattigo bindings

Build the Go bridge code to WASM target (`GOOS=js GOARCH=wasm`). The WASM binary exposes the same client operations as the Python bridge:

- `KeyGenerator`: `genSecretKey()`, `genPublicKey(sk)`, `genRelinearizationKey(sk)`, `genGaloisKey(sk, element)`
- `BootstrapParameters`: `fromLiteral(params, literalJSON)`, `genEvaluationKeys(sk)`
- `Encoder`: `encode(values, level, scale)`, `decode(plaintext)`
- `Encryptor`: `encrypt(plaintext)`
- `Decryptor`: `decrypt(ciphertext)`
- Serialization: `marshal()`/`unmarshal()` on all types

TypeScript wrappers in `js/lattigo/src/` provide ergonomic API over the raw WASM exports. Memory management: Go objects tracked by the WASM runtime's finalizer or explicit `.free()` calls.

Expected binary size: ~8 MB uncompressed, ~3 MB gzipped.

#### 4.2 JS examples

Tensor-to-slot mapping (flatten, pad, split) is trivial user code — no JS library needed. `CKKSParams` and `KeyManifest` are TypeScript interfaces parsed from the server's JSON response. Example code in `js/examples/` demonstrates the full flow.

#### 4.3 Browser demo

`examples/wasm-demo/`:

- Go HTTP server: loads `.orion` model, exposes `/params`, `/session`, `/session/{id}/infer`
- HTML/JS client: loads WASM, generates keys, encrypts input, sends to server, decrypts result
- End-to-end demonstration: browser-side secret key never leaves the client

#### Phase 4 acceptance checklist

- [x] `js/lattigo/` builds to `.wasm` (< 10 MB uncompressed)
- [x] TypeScript wrappers compile without errors
- [x] Full key generation flow (SK → PK → RLK → Galois keys → MemEvaluationKeySet) → `encrypt()` → `decrypt()` roundtrip works in Node.js
- [x] JS example: keygen → encode → encrypt → decrypt → decode roundtrip works in Node.js
- [x] Browser demo: compile MLP (Python) → serve (Go) → query (browser) → correct decrypted result
- [x] WASM loads and initializes in < 3 seconds on modern browser
- [x] No Go objects leaked after `.free()` calls

---

### Phase 5: MNIST examples (MLP, LeNet, LoLA)

MNIST model architectures move out of the library into `examples/`. The `orion-compiler` package ships no pre-built models — it provides `orion_compiler.nn` layers, and users compose their own architectures. Each example is a self-contained project showing how to define, train, compile, and run encrypted inference for a specific architecture.

This phase covers MNIST models only (MLP, LeNet, LoLA) — they use `Quad` activations and require no bootstrapping. CIFAR-10 models (AlexNet, VGG, ResNet) and YOLO require bootstrapping support in the Go evaluator and are deferred to Phase 6.

Starts after Phase 3 (packages split, `orion-evaluator` available).

> **Note:** The old `examples/` and `demo/` directories were deleted after Phase 3 — they imported from the deleted `orion` monolith and were completely broken. This phase creates them from scratch using the new package API (`lattigo`, `orion_compiler`, `orion_evaluator`).

#### 5.1 Move models from `orion/models/` to `examples/`

Delete `orion/models/` (or `orion_compiler/models/` after Phase 3). Each model becomes a standalone example:

| Current location          | New location        | Dataset  | Key FHE features                                 |
| ------------------------- | ------------------- | -------- | ------------------------------------------------ |
| `orion/models/mlp.py`     | `examples/mlp/`     | MNIST    | Simplest: Linear + Quad                          |
| `orion/models/lenet.py`   | `examples/lenet/`   | MNIST    | Conv2d + pooling                                 |
| `orion/models/lola.py`    | `examples/lola/`    | MNIST    | Lightweight: 1 Conv + 1 FC                       |
| `orion/models/alexnet.py` | `examples/alexnet/` | CIFAR-10 | Deeper CNN, SiLU (Chebyshev polynomial)          |
| `orion/models/vgg.py`     | `examples/vgg/`     | CIFAR-10 | Deep CNN, ReLU (minimax sign approximation)      |
| `orion/models/resnet.py`  | `examples/resnet/`  | CIFAR-10 | Residual connections (`Add`), bootstrapping      |
| `orion/models/yolo.py`    | `examples/yolo/`    | Custom   | Object detection, ResNet34 backbone, large model |

#### 5.2 Example directory structure

Each example directory contains:

```
examples/<model>/
├── model.py          # Model definition using orion_compiler.nn layers
├── train.py          # Training script (standard PyTorch training loop)
├── run.py            # Full pipeline: compile → encrypt → evaluate → decrypt
└── README.md         # What the model does, CKKS params rationale, expected output
```

**`model.py`** — the architecture definition. Imports only from `orion_compiler.nn`. No training logic, no FHE parameters. Users can copy this file into their own project and modify it.

**`train.py`** — standard PyTorch training. Downloads dataset, trains the model, saves weights to `weights.pt`. Uses `orion.core.utils` helpers (`get_mnist_datasets`, `get_cifar_datasets`, `train_on_mnist`, `train_on_cifar`) where applicable. Can be skipped — examples work with random weights for demonstrating the FHE pipeline (accuracy will be random, but the pipeline is correct).

**`run.py`** — the FHE pipeline:

```python
import numpy as np
import torch
from lattigo import ckks, rlwe
from orion_compiler import Compiler, CKKSParams
from orion_evaluator import Model, Evaluator
from model import Net  # import from local model.py

# Load trained weights (optional — works with random weights too)
net = Net()
if Path("weights.pt").exists():
    net.load_state_dict(torch.load("weights.pt"))
net.eval()

# Cleartext baseline
inp = get_sample_input()  # dataset-specific
out_clear = net(inp)

# Compile
params = CKKSParams(...)  # model-specific, documented in README.md
compiler = Compiler(net, params)
compiler.fit(train_loader)
compiled = compiler.compile()

# Keygen + encrypt (using Lattigo directly)
ckks_params = ckks.Parameters.from_json(compiled.params.to_bridge_json())
keygen = rlwe.KeyGenerator.new(ckks_params)
sk = keygen.gen_secret_key()
pk = keygen.gen_public_key(sk)
rlk = keygen.gen_relinearization_key(sk)
gks = [keygen.gen_galois_key(sk, el) for el in compiled.manifest.galois_elements]
keys_bytes = rlwe.MemEvaluationKeySet.new(rlk, gks).marshal_binary()

encoder = ckks.Encoder.new(ckks_params)
encryptor = rlwe.Encryptor.new(ckks_params, pk)
input_vals = np.zeros(ckks_params.max_slots(), dtype=np.float64)
input_vals[:inp.numel()] = inp.flatten().numpy().astype(np.float64)
pt = encoder.encode(input_vals, compiled.input_level, ckks_params.default_scale())
ct_bytes = encryptor.encrypt_new(pt).marshal_binary()

# Evaluate (Go evaluator via Python bindings)
model = Model.load(compiled.to_bytes())
params_dict, _, _ = model.client_params()
evaluator = Evaluator(params_dict, keys_bytes)
result_bytes = evaluator.forward(model, ct_bytes)

# Decrypt and compare (using Lattigo directly)
decryptor = rlwe.Decryptor.new(ckks_params, sk)
result_ct = rlwe.Ciphertext.unmarshal_binary(result_bytes)
result_pt = decryptor.decrypt_new(result_ct)
out_fhe = encoder.decode(result_pt, ckks_params.max_slots())[:out_clear.numel()]
print(f"Cleartext: {out_clear}")
print(f"FHE:       {out_fhe}")
print(f"MAE:       {np.mean(np.abs(out_clear.detach().numpy().flatten() - out_fhe)):.6f}")

evaluator.close()
model.close()
```

**`README.md`** — per-example documentation:

- What the model does (classification task, dataset, accuracy)
- Architecture diagram (layers, shapes, activation types)
- Why these CKKS parameters (logn, modulus chain depth, scale, bootstrap needs)
- Expected FHE inference time and precision
- How to train from scratch vs. using pre-trained weights

#### 5.3 Model-specific notes

**MLP, LeNet, LoLA** (MNIST, no bootstrap):

- Small modulus chains (`logn=13`, 5-6 levels)
- `Quad` activation (x², depth 1)
- No bootstrap needed
- FHE inference < 10s

**AlexNet** (CIFAR-10, Chebyshev activations):

- Deeper chain for `SiLU` polynomial approximation
- `ring_type="standard"` (complex slots needed for CIFAR)
- Demonstrates Chebyshev polynomial fitting via `compiler.fit()`

**VGG** (CIFAR-10, ReLU):

- `ReLU` uses composite minimax sign polynomials — very deep (15 levels per ReLU)
- Requires bootstrap to reset levels between ReLU blocks
- Longest modulus chain, largest key sizes
- Demonstrates the most expensive FHE activation

**ResNet** (CIFAR-10, residual connections):

- `Add` nodes for skip connections
- Bootstrap placement across residual blocks
- Demonstrates how the compiler handles branching DAGs

**YOLOv1** (object detection):

- Largest model — demonstrates scalability
- ResNet34 backbone with detection head
- Multi-ciphertext packing (if supported) or notes on limitations

#### 5.4 Update training utilities

Move dataset/training helpers currently in `orion/core/utils.py` to a shared location accessible from examples:

- `get_mnist_datasets()`, `train_on_mnist()` — used by MLP, LeNet, LoLA
- `get_cifar_datasets()`, `train_on_cifar()` — used by AlexNet, VGG, ResNet

These can stay in `orion_compiler.core.utils` (the compiler package still depends on torch) or be extracted to a small shared module in `examples/common/`. Decision: keep in `orion_compiler.core.utils` — the compiler already depends on torch, and training utilities are useful beyond examples (e.g., test fixtures).

#### Phase 5 acceptance checklist

- [x] MNIST models (`mlp.py`, `lenet.py`, `lola.py`) removed from `models/` — remaining models (AlexNet, VGG, ResNet, YOLO) stay until Phase 6
- [x] `examples/mlp/` — model.py, train.py, run.py, README.md all present and working
- [x] `examples/lenet/` — same structure, working
- [x] `examples/lola/` — same structure, working
- [x] Each MNIST `run.py` produces correct FHE output end-to-end (MAE < 0.1)
- [x] Each `train.py` trains to reasonable accuracy on MNIST
- [x] Each `README.md` documents CKKS parameter choices and expected performance
- [x] No imports of `orion_compiler.models.{mlp,lenet,lola}` anywhere in the codebase
- [x] `examples/wasm-demo/` still works (Phase 4 deliverable, not modified here)

---

### Phase 6: Bootstrapping and CIFAR-10 examples

Implement bootstrapping support in the Go evaluator, then create examples for the deeper CIFAR-10 architectures and YOLO. These models require bootstrapping because their activation functions (SiLU via Chebyshev polynomials, ReLU via minimax sign approximation) consume many more levels than the simple `Quad` activation used by MNIST models.

Starts after Phase 5 (MNIST examples complete, Go evaluator proven end-to-end).

#### 6.1 Bootstrapping in Go evaluator

Add bootstrap op handling to `evaluator/evaluator.go`. The compiler already inserts `bootstrap` nodes in the computation graph when the modulus chain is exhausted — the Go evaluator needs to execute them using Lattigo's `bootstrapping.Evaluator`.

#### 6.2 CIFAR-10 and YOLO examples

| Example             | Dataset  | Key FHE features                                 |
| ------------------- | -------- | ------------------------------------------------ |
| `examples/alexnet/` | CIFAR-10 | Deeper CNN, SiLU (Chebyshev polynomial)          |
| `examples/vgg/`     | CIFAR-10 | Deep CNN, ReLU (minimax sign approximation)      |
| `examples/resnet/`  | CIFAR-10 | Residual connections (`Add`), bootstrapping      |
| `examples/yolo/`    | Custom   | Object detection, ResNet34 backbone, large model |

Each example follows the same structure as Phase 5: `model.py`, `train.py`, `run.py`, `README.md`.

#### Phase 6 acceptance checklist

- [ ] Go evaluator handles `bootstrap` ops correctly
- [ ] `examples/alexnet/` — model.py, train.py, run.py, README.md all present and working
- [ ] `examples/vgg/` — same structure, working
- [ ] `examples/resnet/` — same structure, working (includes bootstrap)
- [ ] `examples/yolo/` — same structure, working (or documented limitations for multi-CT)
- [ ] Each `run.py` produces correct FHE output end-to-end (MAE within expected tolerance)
- [ ] Each `train.py` trains to reasonable accuracy on CIFAR-10
- [ ] Each `README.md` documents CKKS parameter choices and expected performance
- [ ] Remaining models removed from `models/` directory — `models/` deleted entirely
- [ ] No imports of `orion_compiler.models` anywhere in the codebase

---

### Dependency graph

```
Phase 1 (format)
    │
    ├──────────────────┐
    ▼                  ▼
Phase 2 (Go eval)  Phase 3 (package split, except 3.5/3.6)
    │                  │
    └──────┬───────────┘
           ▼
    Phase 3.5 (move Go evaluator)
    Phase 3.6 (orion-evaluator Python bindings)
           │
     ┌─────┴─────┐
     ▼           ▼
Phase 4       Phase 5
(JS/WASM)     (MNIST examples)
                 │
                 ▼
              Phase 6
              (bootstrapping + CIFAR/YOLO examples)
```

## Resolved Questions

- **Roadmap ordering:** Defined above.
- **Remez algorithm for ReLU:** The compiler calls Lattigo's minimax package directly via the Go bridge. No pre-computation or Python reimplementation needed.
- **Lattigo upstream migration:** Deferred until after the main refactoring (Phases 1–3). Built against `baahl-nyu/lattigo` fork for now.
