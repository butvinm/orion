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

The compiled model uses a simple binary container: JSON header + length-prefixed binary blobs + CRC32 checksum. No third-party serialization libraries — Go reads it with `encoding/json` + `encoding/binary` (stdlib), Python with `json` + `struct` (stdlib), JS with `JSON.parse` + `DataView` (native).

### Binary layout

```
[8]   MAGIC ("ORION\x00\x02\x00")
[4]   HEADER_LEN (uint32 LE)
[N]   HEADER_JSON (utf-8)
[4]   BLOB_COUNT (uint32 LE)
for each blob:
  [8]  BLOB_LEN (uint64 LE)
  [N]  BLOB_DATA
[4]   CRC32 of everything above
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
    "multiplicative_depth": 7,
    "eval_key_size_mb": 97
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
    Cost       CostProfile    `json:"cost"`
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
    Src     string `json:"src"`
    Dst     string `json:"dst"`
    SrcPort int    `json:"src_port,omitempty"`
    DstPort int    `json:"dst_port,omitempty"`
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

1. **Rewrite `CompiledModel` serialization.** New magic (`ORION\x00\x02\x00`), JSON header with `graph.nodes[]` + `graph.edges[]` (replacing the flat topology + module_metadata dicts), raw float64 diagonal blobs (replacing Lattigo-serialized LinearTransform blobs).

2. **Modify compiler output.** Store graph edges from the NetworkDAG (currently built then discarded). Store raw `module.diagonals` as float64 blobs instead of calling `SerializeLinearTransform()`. The compiler still calls `GenerateLinearTransform()` transiently — it's needed to compute required Galois elements via BSGS decomposition — but the Lattigo object is discarded after querying, not serialized.

3. **Delete Python evaluator.** `evaluator.py` (422 lines) becomes dead code the moment the format changes. Delete it. The FHE forward paths in nn modules also become dead code (only the evaluator called them), but can be stripped later in Phase 3.

4. **Cleartext graph validator.** A test that compiles an MLP to `.orion` v2, reads it back, walks the graph with numpy matrix multiplications and polynomial evaluations, and compares against PyTorch cleartext output. Validates format structure and numerical correctness without CKKS.

### Phase 2: Pure Go evaluator

Reads `.orion` v2, walks the computation graph, runs FHE inference. No Python, no skeleton network.

1. **`.orion` file reader and `Model` type.** Parse binary container, build graph from JSON header. For each `linear_transform` node: read raw float64 diagonals from blob, call `GenerateLinearTransform()` to CKKS-encode into Lattigo `LinearTransformation` (one-time startup cost). For each bias: CKKS-encode raw float64. For each polynomial node: parse coefficients from JSON config. The `Model` is immutable after creation and shared across evaluators.

2. **Graph walker and op dispatch.** Topological sort computed at load time. `Forward(model, ct)` walks the sorted nodes, dispatches by `node.Op`. All underlying Lattigo calls already exist in `orionclient/evaluator.go` (`EvalLinearTransform`, `EvalPoly`, `Bootstrap`, `Add`, `Mul`, `Rotate`, `Rescale`) — this phase is mostly wiring.

3. **Per-client `Evaluator`.** Wraps or extends the existing Go `Evaluator`, adding `Forward(model, ct)`. Accepts eval keys from the user, never owns secret keys.

4. **E2E testing.** Python compiles + encrypts, Go evaluates, Python decrypts, compare against cleartext PyTorch output. Test fixtures (`.orion` + serialized ciphertext + expected output) generated by Python, consumed by `go test`.

Built against `baahl-nyu/lattigo` (current fork). Migration to upstream `tuneinsight/lattigo` deferred until after the refactoring is complete.

### Phase 3: Package restructuring

Split the monolith into three Python packages. Can start after Phase 1, parallelizable with Phase 2 (except moving the Go evaluator).

1. **Extract `python/lattigo/`.** Move `orionclient/bridge/` (Go CGO) and `orion/backend/orionclient/ffi.py` (Python ctypes, `GoHandle`). Own `pyproject.toml`. Not Orion-specific. The bridge surface shrinks: with no Python evaluator, it only exposes client ops (keygen, encrypt, decrypt, encode, decode) and compile-time ops (polynomial generation, linear transform creation for Galois element queries, parameter validation). All evaluation ops (`EvalAdd`, `EvalMul`, `EvalPoly`, etc.) are removed from the bridge.

2. **Extract `python/orion-client/`.** Move `client.py`, `params.py` (`CKKSParams`, `CompilerConfig`), `compiled_model.py`, `ciphertext.py`, tensor-to-slot mapping. Pure Python, depends on `lattigo`.

3. **Rename remaining to `python/orion-compiler/`.** `compiler.py`, `nn/`, `core/`, `models/`, `core/compiler_backend.py`. Depends on `orion-client` + `lattigo` + torch.

4. **Clean up dead code.** Strip FHE forward paths from nn modules (the `if self.he_mode` branches). Remove context propagation from `Ciphertext`. Simplify `Module` base class.

5. **Move Go evaluator.** `evaluator/` at repo root, importable as `github.com/butvinm/orion/evaluator`. Root `go.mod` for the evaluator module. Depends on Phase 2.

### Phase 4: JS/WASM bindings

Starts after Phases 1–3 are stable.

1. **`js/lattigo/`** — Go compiled to WASM (~8 MB uncompressed, ~3 MB gzipped). Same Lattigo ops as the Python bridge: keygen, encrypt, decrypt, encode, decode, serialization.

2. **`js/orion-client/`** — Pure TypeScript. Tensor-to-slot mapping, mirrors Python `orion-client`.

3. **Browser demo** — WASM keygen + encrypt in browser, Go server evaluates, browser decrypts.

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
