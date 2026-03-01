# Phase 1: Compiled Model v2 Format

## Overview

Rewrite the `.orion` compiled model format from v1 (Lattigo-serialized blobs, flat topology) to v2 (raw float64 diagonals, computation graph with edges). This is the contract between compiler and evaluator — everything in Phase 2+ builds on it.

Key changes:

- `CompiledModel` stores a `Graph` (nodes + edges) instead of `topology` + `module_metadata`
- Linear transform blobs store raw float64 packed diagonals (~6x smaller) instead of Lattigo `MarshalBinary` data
- Bootstrap operations become explicit graph nodes with edges (not forward hook metadata)
- New `CostProfile` with bootstrap/key counts
- No `module.compile()` calls — compiler reads module state directly (set by `fit()` + `generate_diagonals()`)
- Python evaluator deleted (Go evaluator replaces it in Phase 2)
- Cleartext graph validator confirms format correctness without CKKS

## Context

**Files involved:**

- `orion/core/galois.py` — **NEW**: pure Python BSGS Galois element computation (from `experiments/07_galois_elements_python/galois.py`)
- `orion/compiled_model.py` — CompiledModel, container format, serialization
- `orion/compiler.py` — compile() loop, metadata extraction, blob serialization
- `orion/evaluator.py` — Python evaluator (to be deleted)
- `orion/__init__.py` — public API exports
- `orion/params.py` — CKKSParams, CompilerConfig (add CostProfile)
- `orion/core/auto_bootstrap.py` — bootstrap placement (modify to insert DAG nodes)
- `orion/core/network_dag.py` — DAG structure, fork/join nodes
- `tests/test_galois.py` — **NEW**: unit tests for Galois element computation
- `tests/test_v2_api.py` — integration tests
- `tests/test_gohandle.py` — handle lifecycle tests
- `tests/models/` — model-specific tests

**Design decisions from feasibility analysis:**

1. **Galois elements:** Pure Python BSGS computation — no Go/Lattigo dependency. Reimplements Lattigo's `lintrans.GaloisElements()` algorithm: `GaloisGen^k mod NthRoot` where `GaloisGen=5`, `NthRoot=2^(logn+2)` for ConjugateInvariant or `2^(logn+1)` for Standard ring. Validated in `experiments/07_galois_elements_python/` — produces identical results to Lattigo on a real SimpleMLP model (FULL MATCH on per-LT BSGS, power-of-2 rotations, and output rotation elements).
2. **No compile() calls, no Go dependency in compile():** The compiler does NOT call `module.compile()` on any module. It reads attributes directly (set by `fit()` + `generate_diagonals()`). Galois elements computed in pure Python from diagonal indices + BSGS ratio. The `compile()` step has **zero Go/Lattigo dependency** — only `fit()` still needs Go (for minimax polynomial generation in ReLU). The `compile()` methods on modules become dead code alongside the HE forward paths — stripped in Phase 3.
3. **ReLU:** Keep sub-components as separate graph nodes (`mult`, `polynomial`, `polynomial`, `polynomial`, `mult`). No single `relu` op type. Coefficients stored inline on each Chebyshev node. Deviates from ARCH.md section 1.2 but preserves the current DAG structure.
4. **Fork/join nodes:** Filter out during edge extraction. Re-link edges directly (A -> A_fork -> B becomes A -> B). Fork/join are internal to the bootstrap solver.
5. **Bias:** Needs explicit zero-padding to `max_slots` in the new flow (currently padded by CKKS encoder internally, then decoded back to original length).
6. **Fused BatchNorms:** Excluded from the graph entirely. Already removed from DAG by `remove_fused_batchnorms()`. Their weights are folded into preceding LinearTransform.
7. **CostProfile:** No `rotation_count` for now (requires counting actual eval-time rotations, deferred). Only `bootstrap_count`, `galois_key_count`, `bootstrap_key_count`.

## Development Approach

- **Testing approach**: Regular (implement, then test)
- Complete each task fully before moving to the next
- Make small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes in that task
  - tests are not optional — they are a required part of the checklist
  - write unit tests for new functions/methods
  - write unit tests for modified functions/methods
  - add new test cases for new code paths
  - update existing test cases if behavior changes
  - tests cover both success and error scenarios
- **CRITICAL: all tests must pass before starting next task** — no exceptions
- **CRITICAL: update this plan file when scope changes during implementation**
- Run tests after each change
- No backward compatibility with v1 format

## Testing Strategy

- **Unit tests**: required for every task (see Development Approach above)
- Python tests in `tests/` with pytest. All tests require the Go shared lib.
- E2E validation: compile MLP -> walk graph in cleartext numpy -> compare to PyTorch

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with + prefix
- Document issues/blockers with ! prefix
- Update plan if implementation deviates from original scope
- Keep plan in sync with actual work done

## Implementation Steps

### Task 1: Add Galois element computation module

Move the validated pure Python BSGS algorithm from `experiments/07_galois_elements_python/galois.py` into the main codebase.

- [x] Create `orion/core/galois.py` with functions from experiment: `galois_element()`, `galois_elements()`, `bsgs_index()`, `find_best_bsgs_ratio()`, `compute_galois_elements()`, `nth_root_for_ring()`, `compute_galois_elements_for_linear_transform()`
- [x] Write unit tests in `tests/test_galois.py`:
  - [x] `galois_element()` matches known values: `galois_element(1, 2^15) = 5`, `galois_element(0, ...) = 1`
  - [x] `nth_root_for_ring()`: standard = `2^(logn+1)`, conjugate_invariant = `2^(logn+2)`
  - [x] `bsgs_index()`: verify baby-step/giant-step decomposition for known inputs
  - [x] `compute_galois_elements()` with BSGS enabled vs disabled
  - [x] Integration test: compare against Lattigo via `TransformEncoder` for a real LinearTransform (same approach as `experiments/07_galois_elements_python/compare.py`)
- [x] Run tests — must pass before task 2

### Task 2: Add new data types

Add the v2 data structures alongside the existing v1 code (don't break anything yet).

- [x] Add `CostProfile` frozen dataclass to `orion/params.py`: `bootstrap_count`, `galois_key_count`, `bootstrap_key_count`
- [x] Add `GraphNode` dataclass to `orion/compiled_model.py`: `name`, `op`, `level`, `depth`, `shape` (optional dict), `config` (dict), `blob_refs` (optional dict)
- [x] Add `GraphEdge` dataclass to `orion/compiled_model.py`: `src`, `dst`
- [x] Add `Graph` dataclass to `orion/compiled_model.py`: `input`, `output`, `nodes` (list of GraphNode), `edges` (list of GraphEdge)
- [x] Add `to_dict()` and `from_dict()` methods on each new type for JSON round-tripping
- [x] Write tests for `CostProfile` creation and `to_dict()`/`from_dict()` roundtrip
- [x] Write tests for `GraphNode`, `GraphEdge`, `Graph` creation and `to_dict()`/`from_dict()` roundtrip
- [x] Run tests — must pass before task 3

### Task 3: Add raw diagonal blob helpers

Implement `pack_raw_diagonals()` and `unpack_raw_diagonals()` for the v2 blob format.

- [x] Add `pack_raw_diagonals(diags: dict[int, list[float]], max_slots: int) -> bytes` to `orion/compiled_model.py`
  - Format: `[4B NUM_DIAGS uint32 LE][NUM_DIAGS x 4B DIAG_INDICES int32 LE sorted ascending][NUM_DIAGS x max_slots x 8B VALUES float64 LE]`
- [x] Add `unpack_raw_diagonals(data: bytes, max_slots: int) -> dict[int, list[float]]`
  - Parse fixed-stride format, return `{diag_index: [float64_values]}`
- [x] Add `pack_raw_bias(bias, max_slots: int) -> bytes`
  - Zero-pad to `max_slots`, write as raw float64 LE
- [x] Add `unpack_raw_bias(data: bytes, max_slots: int) -> list[float]`
- [x] Write tests for `pack_raw_diagonals` -> `unpack_raw_diagonals` roundtrip (various diagonal counts, indices, values)
- [x] Write tests: diagonal indices sorted ascending in packed output
- [x] Write tests: each diagonal has exactly `max_slots` values
- [x] Write tests for `pack_raw_bias` -> `unpack_raw_bias` roundtrip
- [x] Write tests: bias blob length = `max_slots * 8` bytes
- [x] Write edge case tests: empty diagonals, single diagonal, max diagonal index
- [x] Run tests — must pass before task 4

### Task 4: Insert bootstrap nodes into DAG

Change bootstrap from forward hooks to explicit DAG nodes. This must happen before the compiler rewrite (Task 5) because the compiler needs bootstrap nodes in the DAG to emit them as graph nodes.

- [x] Modify `BootstrapPlacer.place_bootstraps()` to insert explicit bootstrap nodes into the `NetworkDAG`:
  - For each node marked with `bootstrap=True`: create a `boot_{idx}` node, remove edges from that node to all its children, add edges `node -> boot_{idx}` and `boot_{idx} -> child` for each child
  - Set node attributes: `op="bootstrap"`, `module=bootstrapper_instance`
  - The bootstrapper instance is still created by `_create_bootstrapper()` (still calls `fit()` for prescale/postscale computation — but NOT `compile()`)
- [x] Set bootstrap node level: the input level to the bootstrap = `predecessor.level - predecessor.depth`
- [x] Verify that `network_dag.topological_sort()` still works correctly after insertion (new node sits strictly between parent and children — acyclicity preserved)
- [x] Write unit tests for bootstrap DAG insertion: create a small DAG manually, insert a bootstrap node, verify edges are correct
- [x] Write integration test: compile a model that requires bootstraps (need logq chain short enough to trigger bootstrap), verify bootstrap nodes appear in `network_dag`
- [x] Run tests — must pass before task 5

### Task 5: Rewrite CompiledModel + compiler for v2 format

This is the core task. Both `CompiledModel` and `Compiler.compile()` change together (can't change one without the other). Also updates existing compiler tests to match v2 structure.

**CompiledModel changes:**

- [ ] Change `CompiledModel` fields: remove `module_metadata`, `topology`; add `cost` (CostProfile), `graph` (Graph)
- [ ] Update magic from `ORMDL\x00\x01\x00` to `ORION\x00\x02\x00`
- [ ] Update `to_bytes()`: serialize JSON header with `version: 2`, `graph` (nodes, edges, input, output), `cost`, `params`, `config`, `manifest`, `input_level` — matching ARCH.md JSON structure
- [ ] Update `from_bytes()`: parse new magic, deserialize v2 JSON header, reconstruct `Graph`, `CostProfile`, etc.
- [ ] Keep `_pack_container` / `_unpack_container` helpers (same binary container pattern, new magic and JSON)
- [ ] `EvalKeys` and `KeyManifest` are unchanged

**Compiler changes:**

- [ ] Add helper to extract edges from `network_dag`, filtering out fork/join auxiliary nodes (re-link A -> fork -> B as A -> B, and A -> join -> B as A -> B)
- [ ] Rewrite the main compilation loop in `Compiler.compile()`:
  - Do NOT call `module.compile()` on any module
  - For LinearTransforms: serialize raw diagonals via `pack_raw_diagonals()` and raw bias via `pack_raw_bias(packing.construct_linear_bias(module), max_slots)`. Galois elements computed via `orion.core.galois.compute_galois_elements_for_linear_transform()` (pure Python, no Go calls)
  - For all other modules: read attributes directly from module state (set by `fit()`)
- [ ] Compute Galois elements for `KeyManifest` entirely in Python:
  - Per-LinearTransform: `compute_galois_elements_for_linear_transform(diag_indices_per_block, slots, bsgs_ratio, logn, ring_type)` from `orion.core.galois`
  - Power-of-2 rotations: `galois_element(2^i, nth_root)` for `i` in `[0, log2(slots))`
  - Output rotations: `galois_element(slots // 2^i, nth_root)` for each LT with `output_rotations > 0`
  - No `TransformEncoder.generate_transforms()` or `_lt_evaluator` needed for Galois elements
- [ ] Build `GraphNode` list from DAG nodes:
  - Map module classes to op strings: `LinearTransform` -> `"linear_transform"`, `Quad` -> `"quad"`, `Chebyshev`/`Sigmoid`/`SiLU`/`GELU` -> `"polynomial"`, `Activation` (monomial) -> `"polynomial"`, `Add` -> `"add"`, `Mult` -> `"mult"`, `Flatten` -> `"flatten"`, `Bootstrap` -> `"bootstrap"`
  - Build op-specific `config` dicts per ARCH.md section 1.2 (with ReLU deviation: sub-components as individual nodes)
  - Populate `shape` dict for `linear_transform` and `bootstrap` nodes
  - Store polynomial coefficients inline: `{"coeffs": [...], "basis": "chebyshev"|"monomial", "prescale": ..., "postscale": ..., "constant": ...}`
- [ ] Build `GraphEdge` list from filtered DAG edges (fork/join removed)
- [ ] Determine `graph.input` (first node with no predecessors) and `graph.output` (last node with no successors)
- [ ] Remove `_extract_module_metadata()` — replaced by GraphNode construction
- [ ] Remove `_extract_bootstrap_metadata()` — bootstraps are now regular DAG nodes (from Task 4)
- [ ] Compute `CostProfile`: bootstrap_count from solver, galois_key_count = `len(manifest.galois_elements)`, bootstrap_key_count = `len(manifest.bootstrap_slots)`
- [ ] Construct `CompiledModel` with new fields: `cost`, `graph`, `blobs`

**Test updates:**

- [ ] Update `test_compiler_produces_compiled_model`: assert `compiled.graph` has nodes/edges, remove assertions on `compiled.topology` and `compiled.module_metadata`
- [ ] Write test: `CompiledModel.to_bytes()` -> `from_bytes()` roundtrip with real compiled data
- [ ] Write test: magic bytes are `ORION\x00\x02\x00`
- [ ] Write test: JSON header has `graph` key, no `topology` or `modules` keys
- [ ] Write test: all `blob_refs` point to valid blob indices
- [ ] Write test: all edge src/dst reference existing node names
- [ ] Write test: all `linear_transform` blobs are raw float64 (unpack with `unpack_raw_diagonals`, verify structure)
- [ ] Write test: bias blobs are raw float64 of length `max_slots * 8`
- [ ] Write test: polynomial coefficients are inline in node config (not in blobs)
- [ ] Write test: edges form a valid DAG (acyclic, connected)
- [ ] Write test: CostProfile fields are populated and reasonable
- [ ] Run tests — must pass before task 6

### Task 6: Delete Python evaluator

- [ ] Move `test_compiled_model_serialization_roundtrip` from `TestEvaluator` to `TestCompiler` in `tests/test_v2_api.py` (this test only tests CompiledModel serialization, doesn't actually use Evaluator — would be wrongly skipped)
- [ ] Delete `orion/evaluator.py`
- [ ] Remove `from orion.evaluator import Evaluator as Evaluator` from `orion/__init__.py`
- [ ] Remove `from orion.evaluator import Evaluator` imports from test files: `tests/test_v2_api.py`, `tests/test_gohandle.py`, `tests/models/test_chebyshev_model.py`, `tests/models/test_mlp.py`
- [ ] Add `@pytest.mark.skip(reason="Python evaluator removed — Phase 2 provides Go evaluator")` to all test classes/functions that construct or use `Evaluator`:
  - `tests/test_v2_api.py`: `TestEvaluator` class (now only has `test_evaluator_modules_have_levels` after moving the serialization test out)
  - `tests/test_gohandle.py`: `TestEvaluatorLifecycle`, handle tracking tests, error cleanup tests
  - `tests/models/test_chebyshev_model.py`: `test_chebyshev_reconstruct`, `test_chebyshev_full_roundtrip`
  - `tests/models/test_mlp.py`: any test function using `orion.Evaluator`
- [ ] Verify non-evaluator tests still pass: `TestCompiler`, `TestClient`, `TestClientSecretKey`
- [ ] Run full test suite — must pass before task 7

### Task 7: Cleartext graph validator

New test file that validates the `.orion` v2 format without any CKKS operations.

- [ ] Create `tests/test_compiled_format.py`
- [ ] **Structural tests:**
  - [ ] `CompiledModel.to_bytes()` -> `from_bytes()` roundtrip preserves all fields
  - [ ] All `blob_refs` point to valid blob indices (0 <= idx < len(blobs))
  - [ ] All edge `src`/`dst` reference existing node names
  - [ ] `graph.input` and `graph.output` exist in node list
  - [ ] Topological sort of edges is acyclic
  - [ ] Every non-input node has at least one incoming edge
  - [ ] `add`/`mult` nodes have exactly two incoming edges
- [ ] **Numerical tests** (compile an MLP, walk graph in cleartext):
  - [ ] Implement `reconstruct_dense_matrix(diags, max_slots, block_height)` helper — this is the inverse of `packing.diagonalize()` and is non-trivial: it reconstructs a dense matrix from packed diagonals, accounting for hybrid/square embedding and block structure. It needs `fhe_input_shape` and `fhe_output_shape` from the node's `shape` field to determine dimensions.
  - [ ] For each `linear_transform` node: unpack raw diags from blob, reconstruct dense weight matrix, do `numpy.matmul(W, x) + bias`
  - [ ] For each `quad` node: `x = x * x`
  - [ ] For each `polynomial` node: evaluate polynomial using numpy (Horner for monomial, Clenshaw for Chebyshev)
  - [ ] Compare final output against `net(x).detach().numpy()` — tolerance <= 1e-10
- [ ] **Blob format tests:**
  - [ ] `pack_raw_diagonals()` -> `unpack_raw_diagonals()` roundtrip with real compiled data
  - [ ] Diagonal indices sorted ascending
  - [ ] Each diagonal has exactly `max_slots` values
  - [ ] Bias blob length = `max_slots * 8` bytes
- [ ] Run tests — must pass before task 8

### Task 8: Verify acceptance criteria

- [ ] Verify `CompiledModel` uses magic `ORION\x00\x02\x00` and version 2
- [ ] Verify JSON header contains `graph` with `nodes`, `edges`, `input`, `output` (no `topology` or `modules` keys)
- [ ] Verify JSON header contains `cost` profile
- [ ] Verify all `linear_transform` blobs contain raw float64 diagonals, not Lattigo-serialized data
- [ ] Verify bias blobs are raw float64 arrays
- [ ] Verify polynomial coefficients are inline in node `config` (no blobs)
- [ ] Verify bootstrap nodes appear as explicit graph nodes with edges (not hook metadata)
- [ ] Verify fused batch norms absent from graph
- [ ] Verify `orion/evaluator.py` deleted, `Evaluator` removed from `orion/__init__.py`
- [ ] Run full test suite — all non-skipped tests pass
- [ ] Run linter — all issues must be fixed

### Task 9: [Final] Update documentation

- [ ] Update ARCH.md: document ReLU deviation (sub-components as separate nodes instead of single `relu` node), note fork/join filtering, note no `rotation_count` yet
- [ ] Update CLAUDE.md if new patterns discovered during implementation

_Note: ralphex automatically moves completed plans to `docs/plans/completed/`_

## Technical Details

### v2 Binary Layout

```
[8]   MAGIC ("ORION\x00\x02\x00")
[4]   HEADER_LEN (uint32 LE)
[N]   HEADER_JSON (utf-8)
[4]   BLOB_COUNT (uint32 LE)
for each blob:
  [8]  BLOB_LEN (uint64 LE)
  [N]  BLOB_DATA
```

### Raw Diagonal Blob Format

```
[4]                          NUM_DIAGS (uint32 LE)
[NUM_DIAGS x 4]              DIAG_INDICES (int32 LE, sorted ascending)
[NUM_DIAGS x max_slots x 8]  VALUES (float64 LE, IEEE 754)
```

### Op Types (deviation from ARCH.md)

| `op`               | Source module(s)                                     | Notes                                                                                         |
| ------------------ | ---------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `flatten`          | `Flatten`                                            | No-op, shape only                                                                             |
| `linear_transform` | `Linear`, `Conv2d`, `AvgPool2d`                      | Config: `bsgs_ratio`, `output_rotations`. Blob refs: `diag_{row}_{col}`, `bias`               |
| `quad`             | `Quad`                                               | x^2, depth=1                                                                                  |
| `polynomial`       | `Chebyshev`, `Sigmoid`, `SiLU`, `GELU`, `Activation` | Config: `coeffs`, `basis`, `prescale`, `postscale`, `constant`                                |
| `bootstrap`        | `Bootstrap`                                          | Config: `input_level`, `input_min`, `input_max`, `prescale`, `postscale`, `constant`, `slots` |
| `add`              | `Add`                                                | Two incoming edges                                                                            |
| `mult`             | `Mult`                                               | Two incoming edges                                                                            |

**Not in graph:** Fused `BatchNorm` nodes (removed from DAG by `remove_fused_batchnorms()`, weights folded into preceding LinearTransform).

**ReLU deviation:** No single `relu` op. ReLU traces to sub-components: `mult` -> `polynomial` x N -> `mult`. Each sub-node is a separate graph node with its own level/depth. Chebyshev coefficients stored inline on each `polynomial` node's config.

### Edge extraction (fork/join filtering)

When extracting edges from `NetworkDAG`, skip nodes where `op == "fork"` or `op == "join"`:

- For `A -> fork -> B, C`: emit `A -> B` and `A -> C`
- For `A, B -> join -> C`: emit `A -> C` and `B -> C`

### No module.compile() in new flow — compile() is Go-free

The compiler reads module state directly. **No Go/Lattigo calls during compile()** — only `fit()` still needs Go (for minimax polynomial generation in ReLU).

- **LinearTransform**: `module.diagonals` (from `generate_diagonals()`), `module.bsgs_ratio`, `module.output_rotations`, `module.level`, `module.depth`, shapes. Bias from `packing.construct_linear_bias(module)`. Galois elements computed in pure Python via `orion.core.galois.compute_galois_elements_for_linear_transform()` — reimplements Lattigo's BSGS decomposition (`GaloisGen^k mod NthRoot`). Validated against Lattigo in `experiments/07_galois_elements_python/`.
- **Chebyshev/Activation**: `module.coeffs` (from `fit()`), `module.prescale`, `module.postscale`, `module.constant`, `module.level`, `module.depth`.
- **Bootstrap**: `module.prescale`, `module.postscale`, `module.constant`, `module.input_level`, `module.input_min`, `module.input_max`, `module.fhe_input_shape` (from `_create_bootstrapper()` + `fit()`).
- **Quad/Add/Mult/Flatten**: `module.level`, `module.depth`.

## Post-Completion

**Manual verification:**

- Compare `.orion` file size: v2 should be ~6x smaller than v1 for the same model (raw vs CKKS-encoded diagonals)
- Verify the compiled model can be inspected with standard JSON tools

**Phase 2 prerequisites:**

- The v2 format is the input contract for the Go evaluator
- Go `evaluator/format.go` will parse this exact binary layout
- Go `evaluator/model.go` will CKKS-encode raw diagonals at load time
