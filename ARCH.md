# Orion Architecture

## Vision

Orion is an FHE framework for deep neural network inference. It takes PyTorch models, analyzes them, and produces artifacts that enable encrypted inference using the CKKS scheme.

Three core components:

1. **Compiler** — Python + Lattigo (via Go bridge). Analyzes PyTorch models and produces a portable compiled model with the FHE computation graph, packed numerical data, and all metadata needed for inference. Uses Lattigo for crypto-specific computations (polynomial generation, parameter validation, cost estimation) — no reimplementation.
2. **Evaluator** — Pure Go. Loads the compiled model, CKKS-encodes the packed data into Lattigo format at load time, and runs FHE inference on ciphertexts.
3. **Lattigo Bindings** — Go core with Python and JS wrappers. Thin bindings for Lattigo's key generation, encryption, decryption, encoding, and decoding. Not Orion-specific.

Plus a client helper library (**orion-client**, Python/JS) that handles tensor-to-CKKS-slot mapping on top of lattigo bindings.

## Components

### orion-compiler (Python + Lattigo)

**Dependencies:** PyTorch, numpy, Lattigo (via Go bridge).

The compiler uses Lattigo for everything crypto-specific: polynomial approximation (Chebyshev fitting, minimax/Remez for ReLU), CKKS parameter construction and validation, depth/cost calculations, and key manifest computation. This avoids reimplementing Lattigo semantics in Python — the compiler asks Lattigo directly, so there is zero risk of drift between compiler assumptions and evaluator behavior.

The compiler performs:

- FX tracing of PyTorch models (or, in the future, parsing ONNX models)
- Statistical fitting (value ranges per layer)
- Polynomial approximation for activations (delegates to Lattigo)
- Network DAG construction, level assignment, bootstrap placement
- Diagonal extraction from weight matrices (packing)
- Key manifest computation (delegates to Lattigo)
- Cost profiling (rotation count, bootstrap count, multiplicative depth, estimated eval key size)

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

The evaluator accepts the Lattigo crypto context FROM THE USER. It never owns secret keys. This keeps Orion independent from the encryption scheme — users can use threshold encryption, custom key management, or any protocol Lattigo supports.

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

### orion-client (Python/JS helper)

A published package (pip/npm) on top of lattigo bindings. Handles the Orion-specific tensor-to-CKKS-slot mapping:

- `encode_tensor(tensor, level, lattigo_encoder) → list[PlainText]` — flatten, pad to slot count, split into chunks, call `lattigo.encode` per chunk
- `decode_tensor(list[PlainText], shape, lattigo_encoder) → tensor` — call `lattigo.decode` per plaintext, concatenate, reshape

The tensor-to-slot mapping logic (flatten, pad, split, reshape) is pure Python/JS — no Go, no FFI. The per-chunk CKKS encoding/decoding is delegated to lattigo bindings (which are Go, or Go→WASM for JS).

## End-to-End Example

Three actors, three languages, three machines.

### 1. Train and compile (Python, developer machine)

```python
import torch
from orion_compiler import Compiler
from orion_compiler.models import MLP
from orion_client import CKKSParams

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
compiler.compile("model.orion")
```

The compiler produces `model.orion` — a JSON header describing the computation graph plus binary blobs with raw packed numerical data (float64 diagonals, bias vectors, polynomial coefficients). Lattigo is used during compilation (via Go bridge) for polynomial approximation, parameter validation, and key manifest computation, but the output contains no Lattigo binary formats. The compiled model is a portable mathematical description.

Note: `CKKSParams` lives in `orion_client` because the client needs it for keygen and encoding. The compiler imports it from there via its `orion-client` dependency.

### 2. Serve inference (Go, server)

```go
package main

import (
    "encoding/json"
    "net/http"
    "os"
    "sync"

    "github.com/butvinm/orion/evaluator"
)

func main() {
    // Load model once — CKKS-encodes packed diagonals at load time (one-time cost)
    data, _ := os.ReadFile("model.orion")
    model, _ := evaluator.LoadModel(data)

    var evaluators sync.Map

    // GET /params — client needs CKKS params, key manifest, and input level
    // to generate the right keys and encode at the right level
    http.HandleFunc("GET /params", func(w http.ResponseWriter, r *http.Request) {
        json.NewEncoder(w).Encode(model.ClientParams())
    })

    // POST /session — client uploads eval keys, server creates evaluator
    http.HandleFunc("POST /session", func(w http.ResponseWriter, r *http.Request) {
        keys, _ := evaluator.ReadEvalKeys(r.Body)
        eval, _ := evaluator.NewEvaluator(model.Params(), keys)
        id := generateID()
        evaluators.Store(id, eval)
        json.NewEncoder(w).Encode(map[string]string{"id": id})
    })

    // POST /session/{id}/infer — client sends ciphertext, evaluator uses model + cached keys
    http.HandleFunc("POST /session/{id}/infer", func(w http.ResponseWriter, r *http.Request) {
        val, _ := evaluators.Load(r.PathValue("id"))
        eval := val.(*evaluator.Evaluator)

        ct, _ := evaluator.ReadCiphertexts(r.Body)
        result, _ := eval.Forward(model, ct)
        w.Write(result.Marshal())
    })

    http.ListenAndServe(":8080", nil)
}
```

Pure Go. No Python, no PyTorch, no skeleton network. The model is loaded once and shared across all evaluators. Each evaluator holds one client's evaluation keys. `eval.Forward(model, ct)` walks the model's computation graph using that evaluator's keys. Eval keys are uploaded once per session — not per request (Galois keys alone can be 100MB+). The user provides the Lattigo crypto context — Orion never constrains how keys are generated or managed.

### 3. Encrypt and query (JavaScript, browser)

```javascript
import { KeyGenerator, Encryptor, Decryptor, Encoder } from "lattigo";
import { encodeTensor, decodeTensor } from "orion-client";

// Fetch params from server — CKKS params, key manifest, input level
const params = await fetch("/params").then((r) => r.json());

// Generate keys in the browser (runs in WASM, secret key never leaves the client)
const keygen = new KeyGenerator(params.ckks);
const secretKey = keygen.genSecretKey();
const evalKeys = keygen.genEvalKeys(params.manifest);

// Upload eval keys once — server caches them in a session
const { id: sessionId } = await fetch("/session", {
  method: "POST",
  body: evalKeys.marshal(),
}).then((r) => r.json());

// Encode + encrypt the input tensor
const encoder = new Encoder(params.ckks);
const plaintexts = encodeTensor(inputData, params.inputLevel, encoder);
const encryptor = new Encryptor(params.ckks, secretKey);
const ciphertexts = plaintexts.map((pt) => encryptor.encrypt(pt));

// Run inference — only ciphertext sent, keys are already on server
const response = await fetch(`/session/${sessionId}/infer`, {
  method: "POST",
  body: marshalCiphertexts(ciphertexts),
});

// Decrypt the result and decode back to a tensor
const resultCts = unmarshalCiphertexts(await response.arrayBuffer());
const decryptor = new Decryptor(params.ckks, secretKey);
const resultPts = resultCts.map((ct) => decryptor.decrypt(ct));
const output = decodeTensor(resultPts, outputShape, encoder);
```

The secret key is generated in the browser and never leaves it. Lattigo bindings (Go compiled to WASM, ~8 MB uncompressed / ~3 MB gzipped) handle all cryptographic operations and serialization. `encodeTensor`/`decodeTensor` from orion-client are pure JS helpers for tensor-to-slot mapping. `marshalCiphertexts`/`unmarshalCiphertexts` are Lattigo serialization — the wire format is application-specific, not mandated by Orion.

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
    "rotation_count": 248,
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
type CompiledModel struct {
    Version    int            `json:"version"`
    Params     CKKSParams     `json:"params"`
    Config     CompilerConfig `json:"config"`
    Manifest   KeyManifest    `json:"manifest"`
    InputLevel int            `json:"input_level"`
    Cost       CostProfile    `json:"cost"`       // informational, not validated by evaluator
    Graph      Graph          `json:"graph"`
}

type Graph struct {
    Input  string `json:"input"`
    Output string `json:"output"`
    Nodes  []Node `json:"nodes"`
    Edges  []Edge `json:"edges"`
}

type Node struct {
    Name     string            `json:"name"`
    Op       string            `json:"op"`
    Level    int               `json:"level"`
    Depth    int               `json:"depth"`
    Shape    *NodeShape        `json:"shape,omitempty"`
    Config   json.RawMessage   `json:"config"`
    BlobRefs map[string]int    `json:"blob_refs,omitempty"`
}

type Edge struct {
    Src string `json:"src"`
    Dst string `json:"dst"`
}
```

Op-specific configs are `json.RawMessage`, parsed by a type switch when loading:

```go
switch node.Op {
case "linear_transform":
    var cfg LinearTransformConfig
    if err := json.Unmarshal(node.Config, &cfg); err != nil { ... }
case "chebyshev":
    var cfg ChebyshevConfig
    if err := json.Unmarshal(node.Config, &cfg); err != nil { ... }
case "bootstrap":
    var cfg BootstrapConfig
    if err := json.Unmarshal(node.Config, &cfg); err != nil { ... }
// ...
default:
    return fmt.Errorf("unknown op: %s", node.Op)
}
```

## Repo Structure

Monorepo. Go module at root (`github.com/butvinm/orion`). One Go package — `evaluator`. Three Python packages — `lattigo` (Go bridge `.so` + Python FFI), `orion-client` (pure Python: tensor-to-slot mapping + client API), and `orion-compiler` (torch + compilation). The evaluator, Python bridge, and JS bridge each import Lattigo directly — no shared Go wrapper package.

```
orion/
├── go.mod                          # module github.com/butvinm/orion
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
│   ├── orion-client/               # pip install orion-client (pure Python, no Go)
│   │   └── orion_client/           #   Client, CKKSParams, CompiledModel, Ciphertext,
│   │                               #   tensor-to-slot mapping
│   │
│   ├── orion-compiler/             # pip install orion-compiler (pure Python, no Go)
│   │   └── orion_compiler/         #   Compiler, nn/, core/, models/
│   │
│   └── tests/
│
├── js/
│   ├── lattigo/                    # npm package: Lattigo WASM bindings
│   │   ├── bridge/                 #   Go WASM bridge (builds .wasm), imports Lattigo directly
│   │   └── src/                    #   TypeScript wrappers over WASM
│   │
│   └── orion-client/               # npm package: pure TS, tensor-to-slot mapping
│       └── src/
│
├── examples/
│   ├── mlp/                        # E2E: compile (Python) → serve (Go) → query
│   └── wasm-demo/                  # Browser demo: Go server + WASM client
│
└── docs/
```

**`evaluator/`** (`github.com/butvinm/orion/evaluator`) — the only Orion-specific Go package. Reads `.orion` files, CKKS-encodes diagonals at load time, walks the computation graph. Imports Lattigo directly.

**`lattigo`** (`pip install lattigo`) — Python package shipping the Go bridge `.so`. Contains all ctypes bindings (`ffi.py`, `GoHandle`). Not Orion-specific — exposes Lattigo's keygen, encrypt, decrypt, encode, decode, polynomial generation, parameter construction. The only Python package requiring Go/CGO to build from source (pre-built wheels eliminate this).

**`orion-client`** (`pip install orion-client`) — pure Python. Client, CKKSParams, CompiledModel, Ciphertext, PlainText, tensor-to-slot mapping. No Go, no FFI, no `.so`. Calls `lattigo` for all cryptographic operations.

**`orion-compiler`** (`pip install orion-compiler`) — pure Python. Compiler, nn modules, core algorithms, pre-built models. No Go. Calls `lattigo` for polynomial generation and parameter validation.

**No Python evaluator.** Evaluation is Go-only.

**`js/lattigo/`** — npm package shipping the WASM bridge. Same pattern as Python: Go bridge builds `.wasm`, TypeScript wrappers expose Lattigo ops. Imports Lattigo directly.

**`js/orion-client/`** — npm package, pure TypeScript. Tensor-to-slot mapping. Mirrors the Python `orion-client` split.

## Dependencies

| Component          | Go import                            | `pip install`    | Depends on                                 | Does NOT depend on        |
| ------------------ | ------------------------------------ | ---------------- | ------------------------------------------ | ------------------------- |
| `evaluator/`       | `github.com/butvinm/orion/evaluator` | —                | Lattigo                                    | Python, compiler, bridges |
| `lattigo`          | (CGO, builds .so)                    | `lattigo`        | Lattigo (Go)                               | Anything Orion            |
| `orion-client`     | —                                    | `orion-client`   | `lattigo`, numpy                           | torch, `evaluator/`       |
| `orion-compiler`   | —                                    | `orion-compiler` | `orion-client`, `lattigo`, torch, networkx | `evaluator/`              |
| `js/lattigo/`      | (WASM, builds .wasm)                 | —                | Lattigo                                    | `evaluator/`              |
| `js/orion-client/` | —                                    | —                | `js/lattigo/` .wasm                        | Go (pure TS)              |

No circular dependencies. No shared Go packages. `lattigo` is the only package with Go code. `orion-client` and `orion-compiler` are pure Python.

## Build and Install

Source-only builds for now. Pre-built wheels with bundled Go shared library are a natural next step.

**Lattigo bindings:** `cd python/lattigo && pip install -e .` — triggers Go compilation of `bridge/` into a shared library, bundled into the package.

**Orion client:** `cd python/orion-client && pip install -e .` — pure Python, pulls in `lattigo`.

**Orion compiler:** `cd python/orion-compiler && pip install -e .` — pure Python, pulls in `orion-client` + torch.

**JS:** Build script in `js/` compiles `js/lattigo/` to WASM (~8 MB uncompressed, ~3 MB gzipped). This single .wasm provides all lattigo bindings.

**Go Evaluator:** Standard `go build`. Users import `"github.com/butvinm/orion/evaluator"` in their server code.

## Testing

- **Go:** `_test.go` files alongside source (Go convention). `go test ./...`
- **Python:** `python/tests/` with pytest. All tests require the Go shared lib (compiler uses Lattigo).
- **E2E / cross-language:** Runnable examples in `examples/` that validate correctness. Compiler (Python) produces `.orion` → evaluator (Go) consumes it → numerical results compared against cleartext PyTorch output.

There is no Python-side evaluator. Testing compiled models requires Go.

## Roadmap

Four phases. No backward compatibility — the Python evaluator is deleted in Phase 1, the project is broken until the Go evaluator lands in Phase 2. A cleartext graph validator provides testing coverage in between.

### Phase 1: New compiled model format

The `.orion` v2 format is the contract between compiler and evaluator. Everything else builds on it.

#### 1.1 Rewrite `CompiledModel` dataclass

**File:** `orion/compiled_model.py`

Replace the current `CompiledModel` fields:

```python
# Current (v1)
params: CKKSParams
config: CompilerConfig
manifest: KeyManifest
input_level: int
module_metadata: dict      # module_name -> per-type metadata dict
topology: list[str]        # flat execution order
blobs: list[bytes]         # Lattigo-serialized LinearTransform blobs + raw bias blobs

# New (v2)
params: CKKSParams
config: CompilerConfig
manifest: KeyManifest
input_level: int
cost: CostProfile          # new: informational, not validated by evaluator
graph: Graph               # new: nodes + edges (replaces topology + module_metadata)
blobs: list[bytes]         # raw float64 diagonal blobs + raw bias blobs (no Lattigo artifacts)
```

New supporting types:

```python
@dataclass
class CostProfile:
    rotation_count: int
    bootstrap_count: int
    galois_key_count: int       # = len(manifest.galois_elements)
    bootstrap_key_count: int    # = len(manifest.bootstrap_slots)

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

| `op`               | `config` keys                                                                                                                                          | `blob_refs` keys                           | Notes                                                                                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `flatten`          | _(empty)_                                                                                                                                              | _(none)_                                   | Shape-only, no computation                                                                                                                                                   |
| `linear_transform` | `bsgs_ratio` (float), `output_rotations` (int)                                                                                                         | `diag_{row}_{col}` (one per block), `bias` | Includes Linear, Conv2d, AvgPool2d                                                                                                                                           |
| `quad`             | _(empty)_                                                                                                                                              | _(none)_                                   | x² + rescale. `depth: 1`                                                                                                                                                     |
| `polynomial`       | `coeffs` (list[float]), `basis` (`"monomial"` or `"chebyshev"`), `prescale` (float), `postscale` (float), `constant` (float)                           | _(none)_                                   | Coefficients inline (degree ≤ 63). Covers Sigmoid, SiLU, GELU, generic Chebyshev                                                                                             |
| `relu`             | `degrees` (list[int], length 3), `prec` (int), `logalpha` (int), `logerr` (int), `prescale` (float), `postscale` (float), `coeffs` (list[list[float]]) | _(none)_                                   | `coeffs` stores pre-computed minimax sign polynomial coefficients per degree, so evaluator doesn't need to regenerate                                                        |
| `bootstrap`        | `input_level` (int), `input_min` (float), `input_max` (float), `prescale` (float), `postscale` (float), `constant` (float), `slots` (int)              | _(none)_                                   | `slots` = 2^ceil(log2(fhe_input elements))                                                                                                                                   |
| `add`              | _(empty)_                                                                                                                                              | _(none)_                                   | Two incoming edges (residual connection)                                                                                                                                     |
| `mult`             | _(empty)_                                                                                                                                              | _(none)_                                   | Two incoming edges                                                                                                                                                           |
| `batch_norm`       | `fused` (bool)                                                                                                                                         | _(none)_                                   | If `fused: true`, node is informational only (weights folded into preceding linear). Unfused batch norms are not supported — the compiler must fuse all batch norms or error |

All nodes carry `level` (int) and `depth` (int) as top-level fields. `shape` is required for `linear_transform` and `bootstrap`, optional for others.

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

1. **Emit graph edges.** After `network_dag.build_dag()` and all processing (fusion, bootstrap placement), iterate `network_dag.edges()` and serialize as `[{"src": u, "dst": v}, ...]`.

2. **Insert bootstrap nodes into the graph.** Currently, bootstraps are forward hooks attached to modules — they don't exist as DAG nodes. Change `BootstrapPlacer` (or post-process the DAG) to insert explicit `bootstrap` nodes with edges. If bootstrap is placed after `act1` and `act1` → `fc2`:
   - Remove edge `act1` → `fc2`
   - Add node `boot_0` (op: `bootstrap`)
   - Add edges `act1` → `boot_0` → `fc2`

3. **Store raw diagonals instead of Lattigo blobs.** Replace the current flow:

   ```python
   # Current: CKKS-encode and serialize Lattigo object
   module.compile(self._context)  # generates Lattigo LinearTransforms
   blob_data = self.backend.SerializeLinearTransform(tid)
   ```

   With:

   ```python
   # New: store raw float64 diagonals
   module.generate_diagonals(last=...)  # populates module.diagonals
   # Still create Lattigo LT transiently for Galois element computation:
   lt_ids = self._lt_evaluator.generate_transforms(module)
   self._lt_evaluator.delete_transforms(lt_ids)
   # Serialize raw diags:
   for (row, col), diags_block in module.diagonals.items():
       blob = pack_raw_diagonals(diags_block)
       blob_refs[f"diag_{row}_{col}"] = len(blobs)
       blobs.append(blob)
   ```

4. **Emit node metadata in `GraphNode` format.** Replace the current `_extract_module_metadata()` per-type dicts with unified `GraphNode` construction. Map module classes to op strings:

   | Module class                           | `op` string                         |
   | -------------------------------------- | ----------------------------------- |
   | `Linear`, `Conv2d`, `AvgPool2d`        | `linear_transform`                  |
   | `Quad`                                 | `quad`                              |
   | `Activation` (monomial)                | `polynomial` (basis: `"monomial"`)  |
   | `Chebyshev`, `Sigmoid`, `SiLU`, `GELU` | `polynomial` (basis: `"chebyshev"`) |
   | `ReLU`                                 | `relu`                              |
   | `Bootstrap`                            | `bootstrap`                         |
   | `Add`                                  | `add`                               |
   | `Mult`                                 | `mult`                              |
   | `Flatten`                              | `flatten`                           |
   | `BatchNorm1d`, `BatchNorm2d`           | `batch_norm`                        |

5. **Compute cost profile.** Count total rotations, bootstrap count, and key counts (Galois keys = `len(manifest.galois_elements)`, bootstrap keys = `len(manifest.bootstrap_slots)`). The cost profile is informational metadata for the user — the evaluator does not validate it.

6. **Determine graph input/output.** The first node in topological order with no predecessors is `graph.input`. The last node with no successors is `graph.output`.

7. **Store ReLU minimax coefficients.** Currently the compiler calls `generate_minimax_sign_coeffs()` during compilation but only stores the `degrees`/`prec`/`logalpha`/`logerr` parameters. The Go evaluator would need to regenerate the coefficients. Instead, store the actual coefficients in `config.coeffs` as a list of lists (one per polynomial in the composition).

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

- [ ] `CompiledModel` uses magic `ORION\x00\x02\x00` and version 2
- [ ] JSON header contains `graph` with `nodes`, `edges`, `input`, `output` (no `topology` or `modules` keys)
- [ ] JSON header contains `cost` profile
- [ ] All `linear_transform` blobs contain raw float64 diagonals, not Lattigo-serialized data
- [ ] Bias blobs are raw float64 arrays
- [ ] Polynomial coefficients are inline in node `config` (no blobs)
- [ ] ReLU minimax coefficients stored in node `config.coeffs`
- [ ] Bootstrap nodes appear as explicit graph nodes with edges (not hook metadata)
- [ ] Fused batch norms either absent from graph or marked `fused: true`
- [ ] `orion/evaluator.py` deleted, `Evaluator` removed from `orion/__init__.py`
- [ ] `CompiledModel.to_bytes()` → `from_bytes()` roundtrip passes
- [ ] Cleartext graph validator passes for MLP (compile → read → numpy walk → compare to PyTorch)
- [ ] All non-evaluator tests pass (`pytest tests/ -k "not evaluator"` or equivalent)
- [ ] `.orion` file size is ~6× smaller than v1 for the same model (raw vs CKKS-encoded diagonals)

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
    header     CompiledModel                                    // parsed JSON header
    transforms map[string]map[string]*lintrans.LinearTransformation // node_name -> {"diag_0_0": LT, ...}
    biases     map[string]*rlwe.Plaintext                       // node_name -> CKKS-encoded bias
    polys      map[string]*polynomial.Polynomial                // node_name -> Lattigo polynomial
    graph      *Graph                                           // processed graph with adjacency + topo order
}
```

`LoadModel(data []byte) (*Model, error)`:

1. Parse binary container → `CompiledModel` header + blobs
2. Build `ckks.Parameters` from header params
3. Create a temporary `ckks.Encoder` for encoding (no keys needed)
4. For each node where `op == "linear_transform"`:
   - For each blob*ref `diag*{row}\_{col}`→ parse diagonal blob → call`lintrans.NewLinearTransformation(params, diagMap, level, bsgsRatio)` or the equivalent Lattigo constructor to CKKS-encode
   - For blob_ref `bias` → parse bias blob → `encoder.Encode(biasVec, level-depth)` → store `*rlwe.Plaintext`
5. For each node where `op == "polynomial"`:
   - Read `config.coeffs` and `config.basis`
   - If `"chebyshev"`: `bignum.NewPolynomial(bignum.Chebyshev, coeffs, nil)`
   - If `"monomial"`: `bignum.NewPolynomial(bignum.Monomial, coeffs, nil)`
6. For each node where `op == "relu"`:
   - Read `config.coeffs` (list of lists) → create one `bignum.Polynomial` per sub-polynomial
7. Build `*Graph`: compute topological sort, build adjacency map (`nodeName → []inputNodeNames`)
8. Validate: all blob_refs resolved, all edge endpoints exist, graph is acyclic

`ClientParams() ClientParams` — returns `{Params, Manifest, InputLevel}` for the client.

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

`buildGraph(header CompiledModel) (*Graph, error)`:

- Index nodes by name
- Build reverse adjacency from edges: for each edge `{src, dst}`, append `src` to `Inputs[dst]`
- Compute topological sort via Kahn's algorithm (edges already available)
- Validate: `Input` node has no incoming edges, `Output` node has no outgoing edges, exactly one connected component

For nodes with multiple inputs (`add`, `mult`, residual joins): `Inputs[name]` has length ≥ 2. The order of inputs matters — it's the order edges were listed in the JSON.

#### 2.4 `Evaluator` type and `Forward()`

**New file:** `evaluator/evaluator.go`

```go
type Evaluator struct {
    params       ckks.Parameters
    encoder      *ckks.Encoder
    evaluator    *ckks.Evaluator
    linEval      *lintrans.Evaluator
    polyEval     *polynomial.Evaluator
    bootstrappers map[int]*bootstrapping.Evaluator
}

func NewEvaluator(params CKKSParams, keys EvalKeyBundle) (*Evaluator, error)
func (e *Evaluator) Forward(model *Model, input []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error)
func (e *Evaluator) Close()
```

The `Evaluator` is created from `orionclient.NewEvaluator()` (reuse existing constructor that builds Lattigo evaluator from params + keys) or by directly constructing the Lattigo types. It holds per-client evaluation keys. No secret key.

An `Evaluator` is **not goroutine-safe** — Lattigo evaluators carry internal NTT/decomposition buffers that are reused across operations. Use one `Evaluator` per goroutine, or protect `Forward()` with a mutex.

`Forward()` walks `model.graph.Order`:

```go
func (e *Evaluator) Forward(model *Model, input []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
    results := make(map[string][]*rlwe.Ciphertext)
    results[model.graph.Input] = input

    for _, name := range model.graph.Order {
        if name == model.graph.Input {
            continue
        }
        node := model.graph.Nodes[name]
        inputs := gatherInputs(results, model.graph.Inputs, name)
        out, err := e.evalNode(model, node, inputs)
        if err != nil {
            return nil, fmt.Errorf("node %s: %w", name, err)
        }
        results[name] = out
    }
    return results[model.graph.Output], nil
}
```

**Scope limitation:** Phase 2 targets single-ciphertext-in, single-ciphertext-out models (MLP, LeNet, LoLA, small conv nets). Multi-ciphertext block-matrix routing (when weight matrices exceed the slot count in both dimensions) is deferred. The `[]*rlwe.Ciphertext` type signature is the extensibility point for future multi-CT support.

**Memory:** The `results` map keeps all intermediate ciphertexts alive until `Forward()` returns. Implementations may free intermediate results once all downstream consumers have been evaluated.

#### 2.5 Op implementations

Each op is a method on `Evaluator`. All underlying Lattigo calls already exist in `orionclient/evaluator.go`.

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
2. If `config.prescale != 0 && config.constant != 0`: `evaluator.AddScalar(ct, constant)`, `evaluator.MulScalar(ct, prescale)`, `evaluator.Rescale(ct)`
3. `polyEval.Evaluate(ct, poly, targetScale)` (Lattigo's Paterson-Stockmeyer or baby-step-giant-step polynomial evaluator)
4. If `config.postscale != 1`: `evaluator.MulScalar(ct, postscale)`, `evaluator.Rescale(ct)`
5. If `config.constant != 0`: `evaluator.AddScalar(ct, -constant)`

**`relu`:**

```go
func (e *Evaluator) evalReLU(model *Model, node *Node, ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error)
```

ReLU is computed as `relu(x) = x * sign(x)` where the sign function is approximated by a composite of minimax Chebyshev polynomials. The evaluation sequence mirrors `ReLU.forward()` in HE mode (`orion/nn/activation.py`):

1. **Prescale:** `x = x * prescale` (MulScalar + Rescale). Prescale normalizes the input range to [-1, 1] for the sign approximation. If `prescale == 1`, skip.
2. **Compute sign:** save a copy of `x`. Evaluate each polynomial in `config.coeffs` in sequence: for each `coeffs[i]`, construct the Chebyshev polynomial and call `polyEval.Evaluate(x, poly, targetScale)`. Each polynomial refines the sign approximation and consumes `ceil(log2(degree+1))` multiplicative levels. The last polynomial's output scale is set to match `x`'s level for the subsequent multiplication.
3. **Multiply:** `result = x * sign(x)` (MulRelin + Rescale). This produces `|x|` (approximately).
4. **Postscale:** `result = result * postscale` (MulScalar + Rescale). Reverses the prescaling.

Total depth per ReLU: `sum(ceil(log2(d+1)) for d in degrees) + 2` (one multiplication for prescale × input, one for input × sign). For default degrees `[15, 15, 27]`: depth = 4 + 4 + 5 + 2 = **15 levels**.

**`bootstrap`:**

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

**`batch_norm`:** If `fused: true`, no-op (weights folded into preceding linear). Unfused batch norms are not supported — `LoadModel()` returns an error if a `batch_norm` node has `fused: false`.

#### 2.6 E2E testing

**Test fixture generation** (Python script, run before `go test`):

1. Compile MLP → write `testdata/mlp.orion`
2. Create Client, generate keys → write `testdata/mlp.keys` (EvalKeys serialized)
3. Encrypt sample input → write `testdata/mlp.input.ct` (Ciphertext serialized)
4. Compute cleartext expected output → write `testdata/mlp.expected.json` (float64 array)
5. Write `testdata/mlp.params.json` (CKKSParams for Go to construct evaluator)

**Go test** (`evaluator/evaluator_test.go`):

1. `LoadModel("testdata/mlp.orion")`
2. Parse keys, create `NewEvaluator(params, keys)`
3. Unmarshal input ciphertext
4. `result := eval.Forward(model, input)`
5. Create a temporary `Client` (from a generated secret key also saved in testdata), decrypt result
6. Compare decrypted output against `mlp.expected.json` — tolerance accounts for FHE noise (≤ 1e-3 for typical CKKS parameters)

Repeat for at least: MLP (linear-only), model with Chebyshev activations, model with bootstrap (if feasible in test time).

#### Phase 2 acceptance checklist

- [ ] `evaluator.LoadModel(data)` successfully parses `.orion` v2 files produced by the Python compiler
- [ ] `model.ClientParams()` returns correct CKKS params, key manifest, and input level
- [ ] `evaluator.NewEvaluator(params, keys)` constructs from serialized eval keys
- [ ] `eval.Forward(model, ct)` produces correct results for MLP (compare decrypted output to cleartext, tolerance ≤ 1e-3)
- [ ] `eval.Forward()` handles all op types present in test models: `linear_transform`, `quad`, `polynomial`, `flatten`, `add`
- [ ] Multiple `Evaluator` instances sharing one `Model` work correctly (concurrent reads)
- [ ] `Evaluator.Close()` releases all resources, no leaked goroutines or memory
- [ ] All methods return `(result, error)` — no panics on malformed input
- [ ] `go test ./evaluator/...` passes
- [ ] E2E test: Python compile → Python encrypt → Go evaluate → Python decrypt → correct output

---

### Phase 3: Package restructuring

Split the monolith into three Python packages. Can start after Phase 1, parallelizable with Phase 2 (except moving the Go evaluator).

#### 3.1 Extract `python/lattigo/`

Create `python/lattigo/` with its own `pyproject.toml` (`pip install lattigo`).

**Move into `python/lattigo/`:**

| From                                     | To                              |
| ---------------------------------------- | ------------------------------- |
| `orionclient/bridge/`                    | `python/lattigo/bridge/`        |
| `orion/backend/orionclient/ffi.py`       | `python/lattigo/lattigo/ffi.py` |
| `GoHandle` class and all ctypes bindings | `python/lattigo/lattigo/`       |

**Bridge API surface after cleanup.** With no Python evaluator, remove all evaluation-only bridge functions:

| Keep (client + compile-time)                                                                                                  | Remove (evaluation-only)                                                   |
| ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `NewClient`, `ClientClose`                                                                                                    | `NewEvaluator`, `EvalClose`                                                |
| `ClientEncode`, `ClientDecode`                                                                                                | `EvalEncode`                                                               |
| `ClientEncrypt`, `ClientDecrypt`                                                                                              | `EvalAdd`, `EvalSub`, `EvalMul`                                            |
| `ClientGenerateRLK`, `ClientGenerateGaloisKey`, `ClientGenerateBootstrapKeys`, `ClientGenerateKeys`                           | `EvalAddPlaintext`, `EvalSubPlaintext`, `EvalMulPlaintext`                 |
| `ClientSecretKey`, `ClientMaxSlots`, `ClientGaloisElement`, `ClientModuliChain`, `ClientAuxModuliChain`, `ClientDefaultScale` | `EvalAddScalar`, `EvalMulScalar`, `EvalNegate`                             |
| `GenerateLinearTransformFromParams` (compile-time: Galois element computation)                                                | `EvalRotate`, `EvalRescale`                                                |
| `LinearTransformRequiredGaloisElements`                                                                                       | `EvalPoly`, `EvalLinearTransform`, `EvalBootstrap`                         |
| `GeneratePolynomialMonomial`, `GeneratePolynomialChebyshev`                                                                   | `EvalModuliChain`, `EvalDefaultScale`, `EvalMaxSlots`, `EvalGaloisElement` |
| `GenerateMinimaxSignCoeffs`                                                                                                   | `NewEvalKeyBundle` and all `EvalKeyBundle*` setters                        |
| `CiphertextMarshal`, `CiphertextUnmarshal`                                                                                    |                                                                            |
| `LinearTransformMarshal`, `LinearTransformUnmarshal`                                                                          |                                                                            |
| `DeleteHandle`                                                                                                                |                                                                            |

Note: `GenerateLinearTransformFromParams` stays because the compiler creates transient LinearTransforms to query Galois elements. `LinearTransformRequiredGaloisElements` stays for the same reason. Both can be deleted later if Galois element computation is decoupled from LinearTransform creation.

The bridge Go code (`bridge/*.go`) is updated to remove the deleted exports. The shared library shrinks.

#### 3.2 Extract `python/orion-client/`

Create `python/orion-client/` with its own `pyproject.toml` (`pip install orion-client`, depends on `lattigo`).

**Move into `python/orion-client/orion_client/`:**

| From                      | To                               |
| ------------------------- | -------------------------------- |
| `orion/client.py`         | `orion_client/client.py`         |
| `orion/params.py`         | `orion_client/params.py`         |
| `orion/compiled_model.py` | `orion_client/compiled_model.py` |
| `orion/ciphertext.py`     | `orion_client/ciphertext.py`     |

Pure Python. No torch dependency. `numpy` for tensor-to-slot mapping (flatten, pad, split, reshape).

The `Client` class imports `lattigo` for FFI calls (keygen, encrypt, decrypt, encode, decode). `CompiledModel`, `CKKSParams`, `KeyManifest`, `EvalKeys` are pure dataclasses with no FFI dependency.

#### 3.3 Rename remaining to `python/orion-compiler/`

Create `python/orion-compiler/` with its own `pyproject.toml` (`pip install orion-compiler`, depends on `orion-client`, `lattigo`, `torch`, `networkx`).

**Move into `python/orion-compiler/orion_compiler/`:**

| From                | To                           |
| ------------------- | ---------------------------- |
| `orion/compiler.py` | `orion_compiler/compiler.py` |
| `orion/nn/`         | `orion_compiler/nn/`         |
| `orion/core/`       | `orion_compiler/core/`       |
| `orion/models/`     | `orion_compiler/models/`     |

The compiler imports `CKKSParams`, `CompilerConfig`, `CompiledModel` from `orion_client`. It imports `lattigo` for compile-time Go bridge operations (polynomial generation, linear transform Galois element queries, parameter validation).

#### 3.4 Clean up dead code

With the Python evaluator gone (Phase 1) and packages split:

1. **Strip FHE forward paths from nn modules.** Remove the `if self.he_mode` branches in `forward()` methods of `Linear`, `Conv2d`, `Quad`, `Activation`, `Chebyshev`, `Bootstrap`, `Add`, `Mult`, `Flatten`, `BatchNorm`. These modules become compile-only: they have `compile(context)`, `fit(context)`, and cleartext `forward()`.

2. **Remove `he()` toggle from `Module`.** The `he_mode` flag, `he()` method, and `eval()`/`train()` overrides are dead. Modules always run in cleartext mode (for tracing and fitting).

3. **Remove context propagation from `Ciphertext`.** The `context` attribute on `Ciphertext` was used to carry the evaluator handle during FHE forward passes. With no Python evaluation, `Ciphertext` becomes a simpler wrapper: shape + serialized Go handle (for client-side encrypt/decrypt only).

4. **Remove `_EvalContext` and `Evaluator._reconstruct_*` code.** Already deleted in Phase 1 (evaluator.py), but verify no remnants in other files.

#### 3.5 Move Go evaluator

After Phase 2 delivers the `evaluator/` package:

- Place `evaluator/` at repo root
- Root `go.mod`: `module github.com/butvinm/orion`
- `python/lattigo/bridge/` retains its own `go.mod` (CGO shared library, separate build)
- Delete `orionclient/` (its functionality split between `python/lattigo/bridge/` and `evaluator/`)

#### Phase 3 acceptance checklist

- [ ] `cd python/lattigo && pip install -e .` succeeds, builds `.so`
- [ ] `cd python/orion-client && pip install -e .` succeeds (pure Python)
- [ ] `cd python/orion-compiler && pip install -e .` succeeds (pure Python + torch)
- [ ] `from lattigo import ffi, GoHandle` works
- [ ] `from orion_client import Client, CKKSParams, CompiledModel, EvalKeys` works
- [ ] `from orion_compiler import Compiler` works
- [ ] `from orion_compiler.models import MLP` works
- [ ] Compiler E2E: compile MLP → produces valid `.orion` v2
- [ ] Client E2E: load compiled model → generate keys → encrypt/decrypt roundtrip
- [ ] Bridge `.so` does not export evaluation functions (`EvalAdd`, `EvalMul`, etc.)
- [ ] No `he_mode`, `he()`, or FHE forward branches in nn modules
- [ ] `Ciphertext` has no `context` attribute
- [ ] `orionclient/` directory deleted
- [ ] `go.mod` at repo root, `go test ./evaluator/...` passes
- [ ] No circular imports between packages
- [ ] `python/tests/` pass with the new package structure

---

### Phase 4: JS/WASM bindings

Starts after Phases 1–3 are stable.

#### 4.1 `js/lattigo/` — WASM Lattigo bindings

Build the Go bridge code to WASM target (`GOOS=js GOARCH=wasm`). The WASM binary exposes the same client operations as the Python bridge:

- `KeyGenerator`: `genSecretKey()`, `genPublicKey()`, `genRLK()`, `genGaloisKey(element)`, `genBootstrapKeys(slots, logP)`, `genEvalKeys(manifest)`
- `Encoder`: `encode(values, level, scale)`, `decode(plaintext)`
- `Encryptor`: `encrypt(plaintext)`
- `Decryptor`: `decrypt(ciphertext)`
- Serialization: `marshal()`/`unmarshal()` on all types

TypeScript wrappers in `js/lattigo/src/` provide ergonomic API over the raw WASM exports. Memory management: Go objects tracked by the WASM runtime's finalizer or explicit `.free()` calls.

Expected binary size: ~8 MB uncompressed, ~3 MB gzipped.

#### 4.2 `js/orion-client/` — TypeScript tensor mapping

Pure TypeScript package (`npm install orion-client`). Mirrors Python `orion_client`:

- `encodeTensor(data: Float64Array, level: number, encoder: Encoder): Plaintext[]` — flatten, pad to slot count, split into chunks, encode each
- `decodeTensor(plaintexts: Plaintext[], shape: number[], encoder: Encoder): Float64Array` — decode each, concatenate, reshape
- `CKKSParams`, `KeyManifest` as TypeScript interfaces (parsed from server JSON)

No CKKS operations — delegates to `js/lattigo/` for all crypto.

#### 4.3 Browser demo

`examples/wasm-demo/`:

- Go HTTP server: loads `.orion` model, exposes `/params`, `/session`, `/session/{id}/infer`
- HTML/JS client: loads WASM, generates keys, encrypts input, sends to server, decrypts result
- End-to-end demonstration: browser-side secret key never leaves the client

#### Phase 4 acceptance checklist

- [ ] `js/lattigo/` builds to `.wasm` (< 10 MB uncompressed)
- [ ] TypeScript wrappers compile without errors
- [ ] `genSecretKey()` → `genEvalKeys(manifest)` → `encrypt()` → `decrypt()` roundtrip works in Node.js
- [ ] `encodeTensor()` → `decodeTensor()` roundtrip matches Python `orion_client` output
- [ ] Browser demo: compile MLP (Python) → serve (Go) → query (browser) → correct decrypted result
- [ ] WASM loads and initializes in < 3 seconds on modern browser
- [ ] No Go objects leaked after `.free()` calls

---

### Dependency graph

```
Phase 1 (format)
    │
    ├──────────────────┐
    ▼                  ▼
Phase 2 (Go eval)  Phase 3 (package split, except 3.5)
    │                  │
    └──────┬───────────┘
           ▼
    Phase 3.5 (move Go evaluator)
           │
           ▼
    Phase 4 (JS/WASM)
```

## Resolved Questions

- **Roadmap ordering:** Defined above.
- **Remez algorithm for ReLU:** The compiler calls Lattigo's minimax package directly via the Go bridge. No pre-computation or Python reimplementation needed.
- **Lattigo upstream migration:** Deferred until after the main refactoring (Phases 1–3). Built against `baahl-nyu/lattigo` fork for now.
