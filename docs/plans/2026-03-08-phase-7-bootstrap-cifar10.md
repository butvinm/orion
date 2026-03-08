# Phase 7: Bootstrapping and CIFAR-10 Examples

## Overview

Implement bootstrapping support in the Go evaluator, add bootstrap key transport to Python evaluator and WASM demo, create CIFAR-10 examples (AlexNet, VGG, ResNet), and clean up old model definitions. This unlocks deeper networks that require modulus chain refresh via Lattigo's bootstrap circuit.

Key deliverables:

- `btp_logn` parameter plumbing through the full stack (Python → Go bridge → .orion format → Go evaluator)
- `evalBootstrap` handler in Go evaluator with lazy bootstrapper initialization
- Bootstrap key transport in `orion_evaluator` (Python) and WASM demo server
- Three CIFAR-10 examples exercising Conv2d, high-degree polynomials, and bootstrap
- Deletion of legacy `models/` directory

## Context (from discovery)

- **Compiler side ready:** `Bootstrap` class (nn/operations.py), `BootstrapSolver`/`BootstrapPlacer` (core/auto_bootstrap.py), `BootstrapConfig` in format.go all exist
- **Go evaluator stub:** evaluator.go line ~119 returns "op not yet implemented" for bootstrap
- **Missing:** `btp_logn` field in `CKKSParams`, `KeyManifest`, `HeaderParams`, `HeaderManifest`; Go bridge `LogNthRoot` enforcement; full `evalBootstrap` implementation; bootstrap key parameters in Python evaluator and WASM demo
- **Reference implementation:** `~/Dev/3rd-party/orion/orion/backend/lattigo/bootstrapper.go` (Go bootstrap), `orion/nn/operations.py` (Python Bootstrap class)
- **CIFAR-10 models exist in old API:** `models/alexnet.py`, `models/vgg.py`, `models/resnet.py` — need porting to `orion_compiler.nn`
- **Phase 5 template:** `examples/mlp/` (model.py, train.py, run.py, README.md)

## Development Approach

- **Testing approach**: Regular (code first, then tests)
- Complete each task fully before moving to the next
- Make small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes in that task
- **CRITICAL: all tests must pass before starting next task**
- **CRITICAL: update this plan file when scope changes during implementation**
- Run tests after each change
- Maintain backward compatibility

## Testing Strategy

- **Unit tests**: required for every task
- **Go tests**: `go test ./evaluator/...` after each Go change
- **Python tests**: `pytest python/tests/` after each Python change
- **JS/WASM tests**: `cd js/lattigo && npm test` after WASM changes
- **Tolerance calibration**: bootstrap adds significant noise — use `max_error_stats_test.go` pattern (N=30 runs, tolerance = max_observed x 1.5)

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with + prefix
- Document issues/blockers with ! prefix
- Update plan if implementation deviates from original scope

## Implementation Steps

### Task 1: Add `btp_logn` to CKKSParams and KeyManifest (Python)

- [x] Add `btp_logn: int | None = None` field to `CKKSParams` in `params.py` (after `boot_logp`)
- [x] In `CKKSParams.__post_init__`: default `btp_logn` to `logn` when `boot_logp` is set and `btp_logn` is None. **Note:** `CKKSParams` is `frozen=True` — must use `object.__setattr__(self, "btp_logn", self.logn)`
- [x] Add `btp_logn` to `CKKSParams.to_bridge_json()` (params.py:76-88). Only include when not None (same pattern as `boot_logp`), otherwise Go gets `"btp_logn": null` which won't unmarshal to `int`
- [x] Add `btp_logn: int | None` field to `KeyManifest` in `compiled_model.py`
- [x] Add `btp_logn` to `KeyManifest.to_dict()` (compiled_model.py:181)
- [x] Parse `btp_logn` in `KeyManifest.from_dict()` (compiled_model.py:189)
- [x] Add `btp_logn` to `CompiledModel.to_bytes()` params dict (compiled_model.py:329)
- [x] Parse `btp_logn` in `CompiledModel.from_bytes()` CKKSParams construction (compiled_model.py:360)
- [x] Populate `btp_logn` in `KeyManifest` during compilation (wherever manifest is constructed)
- [x] Write tests for `CKKSParams` btp_logn defaulting behavior (None→logn when boot_logp set, None stays None when no boot_logp)
- [x] Write tests for round-trip serialization of btp_logn in CompiledModel (both with and without bootstrap)
- [x] Run `pytest python/tests/` — must pass before next task

### Task 2: Add `BtpLogN` to Go structs (params.go, keys.go, format.go, model.go)

All 5 runtime callers of `NewCKKSParameters()` go through `orion.Params` (params.go),
so the `LogNthRoot` fix is centralized there.

- [x] Add `BtpLogN int` field to `orion.Params` in `params.go` (json tag: `"btp_logn,omitempty"`)
- [x] In `Params.NewCKKSParameters()` (params.go:34): set `LogNthRoot: p.BtpLogN + 1` when `p.BtpLogN > 0`. This affects all 5 callers: `python/lattigo/bridge/lattigo.go:33`, `evaluator/model.go:41`, `python/orion-evaluator/bridge/evaluator.go:95`, `examples/wasm-demo/server/main.go:79`, and tests.
- [x] Add `BtpLogN int` field to `orion.Manifest` in `keys.go`
- [x] Add `BtpLogN int` field to `HeaderParams` in `evaluator/format.go`
- [x] Add `BtpLogN int` field to `HeaderManifest` in `evaluator/format.go`
- [x] Update `headerToParams()` in `model.go:220` to copy `BtpLogN` from `HeaderParams`
- [x] Update `ClientParams()` in `model.go:209` to copy `BtpLogN` from `HeaderManifest`
- [x] Write Go test for LoadModel with btp_logn field present in binary
- [x] Write Go test verifying `LogNthRoot` is set correctly for bootstrap-enabled params (construct params with `BtpLogN=14`, verify generated primes satisfy constraint)
- [x] Run `go test ./evaluator/...` and `go vet ./...` — must pass before next task

### Task 3: Implement `evalBootstrap` in Go evaluator

- [x] Add `bootstrappers map[int]*bootstrapping.Evaluator` field to `Evaluator` struct in `evaluator/evaluator.go`
- [x] Change `NewEvaluatorFromKeySet` signature to accept `btpKeys *bootstrapping.EvaluationKeys` (nil when no bootstrap needed)
- [x] Store btpKeys in Evaluator for lazy initialization
- [x] Implement lazy bootstrapper initialization: on first `Forward()` call, reconstruct `bootstrapping.Parameters` from model manifest (`btp_logn`, `boot_logp`, `bootstrap_slots`) and create `bootstrapping.Evaluator` per unique slot count
- [x] Implement `evalBootstrap(model *Model, node *Node, ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error)`:
  - Step 1: Parse `BootstrapConfig` from node
  - Step 2: If `constant != 0`, `AddNew(ct, constant)` (shift into symmetric range)
  - Step 3: Prescale — encode prescale plaintext at ct.Level() with scale = modulus at that level, `MulNew` + `Rescale`
  - Step 4: Set `ct.LogDimensions.Cols = bootstrapper.LogMaxSlots()` (sparse bootstrap)
  - Step 5: `bootstrapper.Bootstrap(ct)` (refresh modulus chain)
  - Step 6: Sparse-slot postscale — `1 << (params.LogMaxSlots() - bootstrapper.LogMaxSlots())`, integer multiply, restore `LogDimensions.Cols`
  - Step 7: If `postscale != 1`, integer multiply for range-mapping postscale (no rescale)
  - Step 8: If `constant != 0`, `AddNew(ct, -constant)` (un-shift)
- [x] Wire `evalBootstrap` into the op dispatch in `Forward()` (replace the "not yet implemented" error)
- [x] Update `Close()` to nil out `bootstrappers` map
- [x] Update all existing callers of `NewEvaluatorFromKeySet` to pass nil for btpKeys: `evaluator_test.go:62`, `max_error_stats_test.go:74` (the bridge callers in `python/orion-evaluator/bridge/evaluator.go:109` and `wasm-demo/server/main.go:329` are handled in Tasks 5 and 6)
- [x] Run `go test ./evaluator/...` — must pass before next task (existing tests still green)

### Task 4: Go evaluator bootstrap unit tests

Validated with original Orion: MLP at logn=14, `LogQ=[55,40,40,40]`, `LogP=[61,61]`,
`boot_logp=[61x6]`, `RingType=Standard`, `H=192` triggers exactly 1 bootstrap
(logslots=7). Peak RSS: 3 GB. Bootstrap time: 1.3s. Precision: 22.9 bits.

- [x] Write `TestEvalBootstrap` — synthetic model with one bootstrap node at logn=14, verify ciphertext level is refreshed and values preserved within tolerance (~22 bits precision expected)
- [x] Write `TestForwardWithBootstrap` — compile MLP with short `logq=[55,40,40,40]` chain forcing 1 bootstrap, run E2E: keygen → encrypt → evaluate → decrypt → compare. Use params: `logn=14, logq=[55,40,40,40], logp=[61,61], logscale=40, boot_logp=[61x6], ring_type=standard, h=192`
- [x] Calibrate bootstrap tolerance using `max_error_stats_test.go` pattern (N=30 runs, tolerance = max_observed x 1.5)
- [x] Run `go test ./evaluator/...` — must pass before next task

### Task 5: Python `orion_evaluator` bootstrap key support

Changes span 3 files: `evaluator.py` (Python API), `ffi.py` (ctypes prototypes),
`bridge/evaluator.go` (C export). Must rebuild shared library after Go changes.

- [x] Update `EvalNewEvaluator` in `python/orion-evaluator/bridge/evaluator.go:86` to accept additional `btpKeysData *C.char, btpKeysDataLen C.ulong` parameters. When non-null, unmarshal as `bootstrapping.EvaluationKeys` and pass to `NewEvaluatorFromKeySet`
- [x] Update ctypes prototype in `python/orion-evaluator/orion_evaluator/ffi.py:77-82` to match new C signature (add two args: `ctypes.c_void_p, ctypes.c_ulong` for btp keys)
- [x] Update `new_evaluator()` in `ffi.py:184-195` to accept and pass `btp_keys_bytes`
- [x] Add `btp_keys_bytes: bytes | None = None` parameter to `Evaluator.__init__` in `evaluator.py:24`, pass to `ffi.new_evaluator()`
- [x] **Rebuild shared library:** run `python python/orion-evaluator/build_bridge.py` (produces `orion-evaluator-linux.so`)
- [x] Write Python test: create evaluator with bootstrap keys, run forward on bootstrap-enabled model
- [x] Write Python test: evaluator without bootstrap keys on non-bootstrap model still works
- [x] Run `pytest python/tests/` — must pass before next task

### Task 6: WASM demo server bootstrap key endpoint

Server: `examples/wasm-demo/server/main.go`. Client: `examples/wasm-demo/client/client.ts`.

- [x] Add `import "github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"` to server
- [x] Add `btpKeys *bootstrapping.EvaluationKeys` field to `session` struct (main.go:50)
- [x] Add `maxBootstrapKeyBytes` constant (~6 GB — bootstrap keys are much larger than individual Galois keys)
- [x] Add `POST /session/{id}/keys/bootstrap` handler: read body, unmarshal as `bootstrapping.EvaluationKeys`, store in `sess.btpKeys`
- [x] Register route in server setup
- [x] Update `HandleFinalize` (main.go:280): validate `sess.btpKeys != nil` when `s.manifest.BootstrapSlots` is non-empty; pass `sess.btpKeys` to `evaluator.NewEvaluatorFromKeySet`; nil out `sess.btpKeys` after evaluator creation
- [x] Update client (`client.ts`): after Galois key upload and before finalize, if `manifest.bootstrap_slots.length > 0`, generate bootstrap keys via `btpParamsGenEvaluationKeys`, marshal, and POST to `/session/{id}/keys/bootstrap`
- [x] Run `cd js/lattigo && npm run build` to rebuild TypeScript
- [x] Write test for bootstrap key upload endpoint
- [x] Write test for finalize validation rejecting missing bootstrap keys
- [x] Run JS/WASM tests (`cd js/lattigo && npm test`) — must pass before next task

### Task 7: AlexNet CIFAR-10 example

- [x] Create `examples/alexnet/model.py` — port from `models/alexnet.py` using `orion_compiler.nn` imports (Conv2d, SiLU via Chebyshev degree-127, AvgPool2d, AdaptiveAvgPool2d, Linear)
- [x] Create `examples/alexnet/train.py` — CIFAR-10 training script with torchvision data loading
- [x] Create `examples/alexnet/run.py` — FHE compile → keygen → encrypt → evaluate → decrypt pipeline, starting with `logn=15`
- [x] Determine CKKS parameters empirically (adjust logq chain, check if bootstrap is triggered)
- [x] Verify `run.py` compiles and produces correct compilation graph (bootstrap placement, level assignment)
- [x] Create `examples/alexnet/README.md` — document CKKS params, bootstrap count (if any), key size, expected precision
- [x] Run cleartext forward pass to verify model correctness (no FHE — see Note below)

### Task 8: VGG CIFAR-10 example

> **MEMORY WARNING:** VGG16 compilation at logn=16 OOMs on 38GB machines (diagonal
> packing of 13 conv layers with up to 512 channels produces multi-GB weight matrices).
> Do NOT attempt compilation on this machine. Only verify cleartext forward pass.
> Compilation and FHE E2E are deferred to Post-Completion (64+ GB machine).

- [x] Create `examples/vgg/model.py` — port from `models/vgg.py` using `orion_compiler.nn` (ReLU via minimax sign approximation, ~15 levels/activation)
- [x] Create `examples/vgg/train.py` — CIFAR-10 training
- [x] Create `examples/vgg/run.py` — FHE pipeline with bootstrap (cleartext-only path verified, FHE path present but untested)
- [x] Create `examples/vgg/README.md` — document expected CKKS params (logn=16), note that compilation/FHE requires 64+ GB RAM
- [x] Run cleartext forward pass to verify model correctness (`python run.py --cleartext-only`)

### Task 9: ResNet CIFAR-10 example

> **MEMORY WARNING:** ResNet20 at logn=16 OOMs on 38GB machines — both compilation
> (large diagonal matrices) and bootstrapper generation (38 bootstraps). Do NOT attempt
> compilation on this machine. Only verify cleartext forward pass.

- [ ] Create `examples/resnet/model.py` — ResNet20 (BasicBlock, [3,3,3], [16,32,64]) using `orion_compiler.nn`, port from `models/resnet.py`
- [ ] Create `examples/resnet/train.py` — CIFAR-10 training
- [ ] Create `examples/resnet/run.py` — FHE pipeline with bootstrap across residual (branching) DAG paths (cleartext-only path verified, FHE path present but untested). Use reference CKKS params: `logn=16`, `logq=[55,40x10]`, `logp=[61x3]`, `boot_logp=[61x8]`
- [ ] Create `examples/resnet/README.md` — document expected CKKS params (logn=16), 38 bootstraps, note that compilation/FHE requires 64+ GB RAM
- [ ] Run cleartext forward pass to verify model correctness (`python run.py --cleartext-only`)

> **Note: CIFAR-10 compilation and FHE E2E are infeasible on <64GB machines.**
> VGG16 compilation at logn=16 OOMed on 38GB (diagonal packing of conv weight matrices).
> ResNet20 at logn=16 OOMs during bootstrapper generation (38 bootstraps, logslots=14/13/12).
> Bootstrap keys alone are ~5.8 GB at logn=16. Full compilation and FHE E2E verification
> are deferred to Post-Completion (requires 64+ GB machine).
>
> Bootstrap correctness is validated by Go evaluator unit tests at logn=14 (Task 4):
> MLP with 1 bootstrap, full E2E, 22.9 bits precision, 3 GB RSS.
> Compilation graph correctness is validated by AlexNet at logn=15 (Task 7):
> 3 bootstraps, input level 20, 4.5 GB model.

### Task 10: Cleanup legacy models

- [ ] Delete `models/` directory entirely (alexnet.py, vgg.py, resnet.py, yolo.py)
- [ ] Grep codebase for any remaining imports of `orion.nn` or `orion_compiler.models` — remove them
- [ ] Run `pytest python/tests/` — must pass
- [ ] Run `go test ./evaluator/...` — must pass

### Task 11: Verify acceptance criteria

- [ ] Verify all requirements from ARCH.md Phase 7 acceptance checklist are met
- [ ] Verify bootstrap parameter plumbing: `btp_logn` flows through CKKSParams → bridge → .orion → Go structs
- [ ] Verify Go evaluator: `evalBootstrap` handles full sequence, lazy bootstrapper init works
- [ ] Verify key transport: Python evaluator accepts btp_keys_bytes, WASM demo has bootstrap endpoint
- [ ] Verify all three CIFAR-10 examples pass cleartext forward (`python run.py --cleartext-only`). AlexNet compilation already verified (Task 7). VGG/ResNet compilation deferred to Post-Completion (OOMs at logn=16 on 38GB)
- [ ] **Bootstrap E2E acceptance test:** MLP at logn=14 with 1 bootstrap runs through full v2 pipeline (compile → serialize .orion → load in Go evaluator → keygen → encrypt → forward with bootstrap → decrypt → compare). Params: `logn=14, logq=[55,40,40,40], logp=[61,61], logscale=40, boot_logp=[61x6], ring_type=standard, h=192`. Must achieve >=20 bits precision. Peak RSS must stay under 4 GB.
- [ ] Run full test suite: `pytest python/tests/` + `go test ./evaluator/...` + `cd js/lattigo && npm test`
- [ ] Run linter: `go vet ./...`
- [ ] No imports of `orion.nn` or `orion_compiler.models` anywhere

### Task 12: [Final] Update documentation

- [ ] Update `ARCH.md` to mark Phase 7 as complete (if convention exists)
- [ ] Update project knowledge docs if new patterns discovered during implementation

## Technical Details

### Bootstrap execution sequence (evalBootstrap)

1. Parse `BootstrapConfig` from graph node (InputLevel, InputMin, InputMax, Prescale, Postscale, Constant, Slots)
2. Constant shift: `AddNew(ct, constant)` — centers values around 0
3. Prescale: encode prescale at ct.Level() with scale = q_L (modulus), `MulNew` + `Rescale` — maps to [-1, 1]
4. Sparse dims: set `ct.LogDimensions.Cols = bootstrapper.LogMaxSlots()`
5. `bootstrapper.Bootstrap(ct)` — refresh modulus chain
6. Sparse postscale: `1 << (params.LogMaxSlots() - bootstrapper.LogMaxSlots())`, integer multiply, restore dims
7. Range postscale: `int(config.Postscale)`, integer multiply (no rescale needed)
8. Un-shift: `AddNew(ct, -constant)`

Steps 6 and 7 are both integer multiplies — neither consumes a level.

### Bootstrapper lazy initialization

During first `Forward()`, for each unique slot count in `bootstrap_slots`:

```go
btpLit := bootstrapping.ParametersLiteral{
    LogN:     utils.Pointy(manifest.BtpLogN),
    LogP:     manifest.BootLogP,
    Xs:       ckksParams.Xs(),
    LogSlots: utils.Pointy(int(math.Log2(float64(slots)))),
}
btpParams, _ := bootstrapping.NewParametersFromLiteral(ckksParams, btpLit)
bootstrapper, _ := bootstrapping.NewEvaluator(btpParams, btpKeys)
```

### LogNthRoot constraint

Lattigo requires all CKKS primes to satisfy `q = 1 mod 2^(btp_logn+1)`. The Go bridge must set `LogNthRoot = btp_logn + 1` when constructing parameters for bootstrap-enabled models. Without this, `bootstrapping.NewParametersFromLiteral()` fails.

### Bootstrap key size reference

| btp_logn | Bootstrap Keys | Per Galois Key | Galois Count |
| -------- | -------------- | -------------- | ------------ |
| 14       | 1.0 GB         | 27 MB          | 35           |
| 15       | 2.6 GB         | 55 MB          | 43           |
| 16       | 5.8 GB         | 109 MB         | 48           |

Default `btp_logn = logn` (residual ring dimension) avoids ring-switching keys.

### Memory requirements (measured with original Orion)

ResNet20 at logn=16 with 38 bootstraps: OOMs on 38GB machine during
`bootstrapping.NewEvaluator()` construction (before keygen). The scheme init alone
takes ~1.7 GB RSS, compilation pushes past multi-GB, and bootstrapper generation
(for logslots=14, 13, 12) exceeds available memory. **Minimum 64 GB RAM required
for logn=16 bootstrap models.** logn=14 bootstrap tests (Task 4) should be feasible
on 38 GB (~1 GB bootstrap keys).

### CIFAR-10 example parameters

| Example | logn | logq            | logp   | boot_logp | Bootstrap |
| ------- | ---- | --------------- | ------ | --------- | --------- |
| AlexNet | 15   | TBD empirically | TBD    | TBD       | Maybe     |
| VGG     | 16   | TBD empirically | TBD    | TBD       | Yes       |
| ResNet  | 16   | [55, 40x10]     | [61x3] | [61x8]    | Yes       |

## Post-Completion

**FHE E2E verification (requires 64+ GB machine):**

- ResNet20 at logn=16 needs 38 bootstraps (logslots=14, 13, 12) — OOMs on 38GB during compile
- VGG at logn=16 would be heavier (13 ReLU x ~15 levels each)
- AlexNet at logn=15 may be feasible on 38GB if bootstrap is not triggered
- Run each `run.py` on a machine with 64+ GB RAM, verify FHE output matches cleartext
- Measure bootstrap execution time per invocation and total inference time
- Document timing and precision results in respective README.md files

**WASM demo:**

- Verify WASM demo works with bootstrap-enabled models in a real browser
- Bootstrap keys are single-blob upload (not streamable) — verify upload UX for large keys
