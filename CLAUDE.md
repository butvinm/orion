# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Orion

An opinionated fork of [baahl-nyu/orion](https://github.com/baahl-nyu/orion), a research-grade FHE framework for deep learning inference. The fork refactors Orion for practical usage: instance-based API (no global state), explicit context passing, and full access to underlying Lattigo primitives.

Orion takes PyTorch neural networks, analyzes them, and produces artifacts that enable encrypted inference using the CKKS scheme. The core pipeline is: **fit** (collect value range statistics) → **compile** (assign FHE levels, place bootstraps, pack data) → **encrypt & infer** (run on ciphertexts).

## Repository Structure

Three Python packages (`python/lattigo/`, `python/orion-compiler/`, `python/orion-evaluator/`), a Go evaluator (`evaluator/`), a JS/WASM package (`js/lattigo/`), and a browser demo (`examples/wasm-demo/`). Model examples under `examples/models/` (`{mlp,lenet,lola,alexnet,vgg,resnet}.py`) with unified `run.py` and `train.py`. The C3AE age-verification demo at `examples/c3ae-demo/` ships with a self-contained experiments harness flattened directly into the demo dir (`examples/c3ae-demo/{models,bench,scripts,results}/`, with `scripts/{run_cleartext,run_fhe}.sh` orchestrators and `scripts/{verify_fhe,build_results}.py` post-run analysis tools) — cleartext FPR/FNR + FHE timing/RSS benchmarks for `logn15`/`logn16` via a Go-only `bench` binary.

**Dependency graph:** `lattigo` ← `orion-compiler` (+ torch, networkx). `orion-evaluator` is independent. `js/lattigo` depends only on Lattigo (no Orion-specific code).

## Build & Development

### PyPI install (Linux only, no build tools needed)

```bash
pip install orion-v2-lattigo orion-v2-compiler orion-v2-evaluator
```

### From source

**System prerequisites:** Go 1.24+, C compiler (CGO), libgmp-dev, libssl-dev, Python 3.11–3.12, Node.js 18+.

```bash
# Build the Python CGO shared library (required before installing Python packages)
python tools/build_lattigo.py

# Install all Python packages (uv workspace)
uv sync

# Run all Python tests
pytest python/tests/

# Run a single Python test
pytest python/tests/test_v2_api.py::TestCompiler::test_compiler_produces_compiled_model

# Go evaluator tests
go test ./evaluator/...
go vet ./...

# Build JS/WASM binary (requires Go with js/wasm support)
python tools/build_lattigo_wasm.py

# JS/WASM package — install deps, build TypeScript, run tests
cd js/lattigo && npm install && npm run build && npm test

# JS/WASM — build only WASM (Go bridge to lattigo.wasm)
cd js/lattigo && npm run build:wasm

# JS/WASM — build only TypeScript wrappers
cd js/lattigo && npm run build:ts

# JS/WASM — type-check without emitting
cd js/lattigo && npm run typecheck

# JS/WASM — lint TypeScript
cd js/lattigo && npm run lint

# Python linting and formatting (ruff)
ruff check python/
ruff check --fix python/
ruff format python/

# Python type checking (mypy)
mypy python/lattigo/ python/orion-compiler/ python/orion-evaluator/
```

## Design Principles

Orion provides **compilation**, **encoding**, and **evaluation** — never constrain the user's access to Lattigo primitives. No `Client` class, no `orion-client` package. Compiled model stores raw float64 data, not Lattigo artifacts. No backward compatibility with legacy code.

## End-to-end Usage

```python
import orion_compiler.nn as on
from orion_compiler import Compiler, CKKSParams
from lattigo.ckks import Parameters, Encoder
from lattigo.rlwe import KeyGenerator, Encryptor, Decryptor, MemEvaluationKeySet
from orion_evaluator import Model, Evaluator

# 1. Define model using orion_compiler.nn layers
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

# 2. Compile — writes blobs directly to file, no intermediate storage
net = MLP()
compiler = Compiler(net, CKKSParams(logn=14, logq=[...], logp=[...], logscale=40))
compiler.fit(dataloader)
compiler.compile_to_file("model.orion")

# 3. Load model and get client params
with open("model.orion", "rb") as f:
    model = Model.load(f.read())
params_dict, manifest, input_level = model.client_params()

# 4. Client — keygen + encrypt using Lattigo primitives directly
params = Parameters.from_dict(params_dict)
kg = KeyGenerator(params)
sk = kg.gen_secret_key()
pk = kg.gen_public_key(sk)
encoder = Encoder(params)
encryptor = Encryptor(params, pk)
pt = encoder.encode(input_values, level=input_level, scale=params.default_scale())
ct = encryptor.encrypt_new(pt)
ct_bytes = ct.marshal_binary()

# 5. Server — Go evaluator via orion-evaluator
keys_bytes = evk.marshal_binary()  # MemEvaluationKeySet
evaluator = Evaluator(params_dict, keys_bytes)
result_bytes_list = evaluator.forward(model, [ct_bytes])  # list in, list out

# 6. Client — decrypt
from lattigo.rlwe import Ciphertext as RLWECiphertext
result_ct = RLWECiphertext.unmarshal_binary(result_bytes_list[0])
decryptor = Decryptor(params, sk)
result_pt = decryptor.decrypt_new(result_ct)
output = encoder.decode(result_pt, params.max_slots())
```

## Package Details

### Python packages

- `lattigo.ckks` — `Parameters`, `Encoder`
- `lattigo.rlwe` — `SecretKey`, `PublicKey`, `RelinearizationKey`, `GaloisKey`, `Ciphertext`, `Plaintext`, `KeyGenerator`, `Encryptor`, `Decryptor`, `MemEvaluationKeySet`
- `lattigo.gohandle` — `GoHandle` RAII wrapper for cgo.Handle values
- `orion_compiler` — `Compiler`, `CKKSParams`, `CompiledModel`, `Graph`, `GraphNode`, `GraphEdge`, `KeyManifest`, `CompilerConfig`, `CostProfile`
- `orion_compiler.nn` — FHE-compatible layers (cleartext-only forward)
- `orion_compiler.core` — Compilation algorithms (tracer, packing, level assignment, auto-bootstrap, galois)
- `orion_evaluator.Model` — `load()`, `client_params()`, `close()`
- `orion_evaluator.Evaluator` — `__init__(params, keys_bytes, btp_keys_bytes=None)`, `forward(model, ct_bytes_list) → list[bytes]`, `close()`

### Go evaluator (`evaluator/`)

- `evaluator/format.go` — Binary format parser
- `evaluator/graph.go` — Computation graph with topological ordering
- `evaluator/model.go` — `Model` (immutable, shareable): `LoadModel`, `ClientParams()`
- `evaluator/evaluator.go` — `Evaluator` (per-client): `NewEvaluatorFromKeySet`, `Forward`

### GoHandle — Go object lifecycle management

`GoHandle` (`lattigo/gohandle.py`) is an RAII wrapper for `cgo.Handle` values (`uintptr_t`). Rules:

1. **GoHandle wraps every Go object.** Tagged with descriptive strings (`"CKKSParams"`, `"RLWECiphertext"`, etc.).
2. **Bridge functions borrow, never consume.** Only `DeleteHandle` (called by `GoHandle.close()`) frees the handle slot.
3. **Canonical `__del__`.** Every handle-owning class uses `def __del__(self): try: self.close() except Exception: pass`.
4. **Context manager support.** All handle-owning classes support `with` statements (`__enter__`/`__exit__`), which call `close()` on block exit.

### Exception Hierarchies

Each package defines its own exception hierarchy. Use these instead of generic `RuntimeError`/`ValueError`:

- `lattigo.errors` — `LatticeError` (base), `HandleClosedError`, `FFIError`
- `orion_compiler.errors` — `CompilerError` (base), `CompilationError`, `ValidationError`
- `orion_evaluator.errors` — `EvaluatorError` (base), `ModelLoadError`

### Serialization — Lattigo native, no custom formats

- Keys: `MemEvaluationKeySet.MarshalBinary()` / `UnmarshalBinary()`
- Ciphertexts: `rlwe.Ciphertext.MarshalBinary()` / `UnmarshalBinary()`
- Models: `.orion` v2 format (Go parser in `evaluator/format.go`)

## Conventions

- Three separate packages: `orion-v2-lattigo`, `orion-v2-compiler`, `orion-v2-evaluator` (import names: `lattigo`, `orion_compiler`, `orion_evaluator`)
- No `Client` class — users use Lattigo primitives directly
- Go evaluator is a subpackage of the root module (`github.com/butvinm/orion/v2/evaluator`)
- Tests in `python/tests/`, run with `pytest python/tests/`

## FHE Inference Performance Notes

- **`lintrans.Encode` is the dominant allocator on convolution-heavy ops.** For C3AE conv2 at logn=15, a single op allocates ~85 GB transient (mostly GC'd). Top hot spots inside `embedDouble`: `lattigo/ring.Ring.BRedConstants` (51%, fresh slice per call), `lattigo/ring.Ring.ModuliChain` (26%, fresh slice per call), `lattigo/ring.NewPoly` (22%). Both `BRedConstants` and `ModuliChain` are `func (r Ring) ...` value-receiver methods that `make` a fresh slice of immutable per-prime constants on every call — caching them on the Ring would eliminate ~77% of the churn (open upstream PR opportunity). See [issue #21](https://github.com/butvinm/orion/issues/21) for the per-line breakdown.
- **Go-only inference vs Python wrapper: ~50% less peak RSS** for the same model. Measured at logn=15: 54 GB (Go bench) vs 103 GB (Python wrapper, now removed). Cause: the Python wrapper loads two CGO `.so` files (one for keygen via `lattigo`, one for inference via `orion_evaluator`) which can't share Go heap and forces serialize/deserialize round-trips of the eval keys. Prefer the Go bench path (`examples/c3ae-demo/bench/`) for any benchmarking.
- **`GOMEMLIMIT < observed_peak` causes allocator deadlock on the _lazy_ encode path** (baseline). Tested at GOMEMLIMIT=20GiB on the C3AE logn=15 forward (which has a transient 85 GB allocation peak in conv2): process throttled to 0% CPU mid-conv2 and made no progress. Lattigo's keyswitch buffers stay live until the op returns — Go GC can't free them, so the allocator backpressure perma-stalls. Don't go below ~110% of the unconstrained peak on baseline code.
- **`GOMEMLIMIT=100GiB` is safe and beneficial on the _eager pre-encoded_ path at logn=16** (commit `89bd3e7`+). Measured 2026-05-16: feature×logn16 unconstrained peak 127.6 GB → with GOMEMLIMIT=100GiB peak 99.1 GB (~28 GB reduction), no deadlock, forward time _slightly faster_ (52 s vs 60 s). The eager path's working set is steady (no big in-flight transients), so GC pressure at the limit just reclaims a bit more aggressively without stalling. Recommend `GOMEMLIMIT=100GiB` as the default env for logn=16 server deployments.
- **Working-set CT memory is tiny.** The evaluator's `results map[string][]*rlwe.Ciphertext` rarely holds >100 MB total at C3AE scale. The big RSS numbers are from Lattigo internal buffers + the resident evk (~7 GB at logn=15, ~13 GB at logn=16) + transient allocator churn.
- **Intermediate `results` are never freed during `Forward`** (`evaluator/evaluator.go:79`). For C3AE at end of forward, 19 ciphertexts are still alive (~80 MB). Real win would matter more for ResNet-class networks. Reverse-topological discard after last consumer is an open optimization.
- **Eager LT pre-encoding at LoadModel (optimize-server-rss branch, measured 2026-05-16, cpu.16.128.240).** Pre-encoding `lintrans.LinearTransformation` data once in `LoadModel` (instead of every `Forward`) trades load-time for forward-time RSS. logn=15 deltas (3 boundary-band samples avg): `rss_post_load` 15.7 → 43.6 GB (+27.9 GB resident), `rss_post_forward1` 27.3 → 45.7 GB (Forward no longer spikes), `VmHWM` 55.9 → 49.2 GB (**~6.7 GB / 12% peak savings**), load wall-clock 6.6 → 136.3 s (~20× slower; encode cost moves from per-Forward into LoadModel once). logn=16 deltas (re-run with per-diagonal GC + `ParseClientParams`): `rss_post_load` 28.0 → 89.6 GB, `rss_post_forward1` **115.6 → 93.9 GB** (the headline win — per-request transient drops ~22 GB), `VmHWM` 120.1 → 127.6 GB (regresses ~7.5 GB; the eager resident LT cache + Lattigo's evaluator working set together push past baseline's just-in-time pattern). Run completes inside the 128 GB ceiling; correctness preserved (`verify_fhe.py --tol 0.05` PASS, max_diff = 0.0000 to 6 decimals).
- **Per-diagonal `runtime.GC()` is required for logn=16 eager LT encoding to fit in 128 GB** (`evaluator/model.go` `loadLinearTransformMetadata`, commit 89bd3e7). With per-LT-node GC only, the inner encode loop iterating ~4379 diagonals on conv2 accumulated ~4379 × ~300 MB transients in a single function call and OOM-killed during `bench keygen` at 130 GB. Per-diagonal GC keeps load-time RSS at ~89.6 GB (well under baseline forward peak of 115.6 GB). Trade: thousands of small GC pauses during `LoadModel` (load_s 13.6 → 264.7 s at logn=16, ~20× slower). Worth it: the run completes and the per-request transient at inference drops from ~88 GB to ~4 GB.
- **`bench keygen`/`encrypt`/`decrypt` must use `evaluator.ParseClientParams`, not `LoadModel`** (commit 3a0f2f6). The header-only parse returns `(params, manifest, inputLevel)` without allocating biases, polynomials, or encoded LTs. After Task 1 of the optimize-server-rss work made `LoadModel` eagerly encode LTs, calling LoadModel from keygen alone OOM'd at 128 GB. Keep this split — the eager path is only paid by the inference server.
- **Heap composition at logn=16 (eager-encoded, GOMEMLIMIT=100GiB, post_load):** Go heap shows ~61 GB encoded LT diagonals (`lattigo/ring.NewPoly`, 80% of tracked heap), ~13 GB evk (`MemEvaluationKeySet.ReadFrom`), ~1.7 GB model.orion bytes still in RAM (`os.ReadFile` result kept in `LoadModel` frame), and ~14 GB untracked (CGO / runtime / fragmentation). The encoded LT cache is ~5× the prior CLAUDE.md estimate of "13 GB resident at logn=16" — that estimate was wrong. Cheap follow-ups: release the model bytes buffer after parsing (~1.7 GB), investigate NTT-only vs NTT+Montgomery double-allocation in `ring.NewPoly` (~potentially 30 GB).

## VPS / Benchmarking

- Canonical benchmarking flavor: `cpu.16.128.240` on immers.cloud. Fits both `logn15` (54 GB peak) and `logn16` (114.37 GB peak, ~14 GB margin). Larger CPU flavors (256 GB) not currently available.
- Provisioning script: `docs/plans/2026-05-09-c3ae-vps-runs/setup-fhe.sh` (and `setup-train.sh` for GPU training). Takes ~5 min on a fresh `cpu.16.128.240`. Handles: apt build deps, `python3.12` from deadsnakes PPA (NOT in default Ubuntu 22.04 repos), Go 1.24 from upstream tarball (Ubuntu ships 1.18), uv install, repo clone + `git checkout experiments`, build_lattigo CGO, `uv sync` (kagglehub is now a workspace dev-dep), kagglehub UTKFace download, symlink `data/UTKFace` under `examples/c3ae-demo/`. The double-symlink hack for the legacy nested experiments subdir is obsolete after the 2026-05-10 consolidation.
- **Pattern for SSH-resilient long-running work**: `nohup bash work.sh > log 2>&1 < /dev/null &` then poll the log every 30-60s in a `timeout 600` loop. SSH disconnects don't kill detached work. Don't use shell `set -euxo pipefail` together with `ls | head` — SIGPIPE on `ls` triggers pipefail and aborts the script silently.
- **Branch must be on `origin` for VPS provisioning to work** — `setup-fhe.sh` does `git checkout experiments` from the cloned-from-origin repo. Local-only branches require an explicit `git push -u origin experiments` first.
- **Ubuntu 22.04 immers.cloud image needs `vm.overcommit_memory=1` + swap before bench at logn=16.** Default policy (`vm.overcommit_memory=0`) rejects the bench process's large transient allocations _before_ the 128 GB hardware ceiling, producing a misleading early kill. Run `sudo sysctl -w vm.overcommit_memory=1` and add a 64 GB swapfile (`fallocate -l 64G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile`) after `setup-fhe.sh`. Without this the run fails before reaching real OOM.
- **`go` is at `/usr/local/go/bin/go` on the VPS image but non-login ssh shells don't pick it up.** `ssh user@host 'go build'` errors with `go: command not found`. Use `ssh user@host 'bash -lc "go build"'` or set `PATH=/usr/local/go/bin:$PATH` explicitly in non-interactive commands.
- **`examples/c3ae-demo/scripts/run_fhe.sh` isn't multi-instance safe.** Two concurrent invocations race on `out/<cfg>/model.orion`, producing a truncated ~33 MB file instead of the expected ~1.75 GB. If you see a tiny model.orion mid-debug, you have parallel runners (kill them, clear `out/<cfg>/`, rerun once).
- **`examples/c3ae-demo/bench/bench` is statically linked against the evaluator package.** When comparing baseline vs feature bench numbers, rebuild `bench` after each `git checkout` or the comparison is invalid — the running binary keeps whichever evaluator code was linked at its build time, regardless of source-tree state.
- **C3AE empirical numbers (UTKFace test split, n=3557 / boundary 16-20 n=161, seed 42 split):** ReLU 95.6% overall / 64.6% boundary; Quad (FHE-compatible) 94.2% / 61.5%. Quad costs ~3pp accuracy. Boundary band FPR is ~65-71% — model strongly biases "adult" due to 18%/82% class imbalance + asymmetric loss.
- **FHE vs cleartext correctness**: at C3AE scale with confident-saturated sigmoid outputs on the boundary samples, `|fhe_prob - cleartext_prob|` measures 0.000000 to 6 decimals across both logn=15 and logn=16. Use `scripts/verify_fhe.py --tol 0.05` as the pipeline-correctness gate.
