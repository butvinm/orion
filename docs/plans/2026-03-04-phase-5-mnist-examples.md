# Phase 5: MNIST Examples (MLP, LeNet, LoLA)

## Overview

Move the three MNIST model architectures (MLP, LeNet, LoLA) out of `orion_compiler/models/` into self-contained `examples/` directories. Each example demonstrates the full FHE pipeline: define model → train → compile → encrypt → evaluate → decrypt.

This is a partial Phase 5 — CIFAR-10 models (AlexNet, VGG, ResNet) and YOLO require bootstrapping support and are deferred to Phase 6 (documented in ARCH.md).

## Context (from discovery)

- **Source models**: `python/orion-compiler/orion_compiler/models/{mlp,lenet,lola}.py` — all use `orion_compiler.nn` layers with `Quad` activations only
- **Training utils**: `orion_compiler.core.utils` — `get_mnist_datasets()`, `train_on_mnist()` stay here (ARCH.md decision)
- **Working E2E pattern**: `python/tests/test_orion_evaluator.py::TestE2EForward::test_forward_mlp_from_compiler` — full compile→encrypt→forward→decrypt pipeline
- **CKKS params**: MLP uses `logn=13, logq=[29,26,26,26,26,26], logp=[29,29], logscale=26, h=8192, ring_type="conjugate_invariant"`. LeNet/LoLA may need different depth.
- **Test importing models**: `python/tests/models/test_mlp.py` imports `orion_compiler.models` — already skipped, needs cleanup
- **CLAUDE.md reference**: `from orion_compiler.models import MLP` — needs update after models move

## Development Approach

- **Testing approach**: Full E2E execution — each `run.py` must produce correct FHE output with MAE within tolerance
- Complete each task fully before moving to the next
- Make small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes in that task
- **CRITICAL: all tests must pass before starting next task**
- **CRITICAL: update this plan file when scope changes during implementation**
- Run tests after each change

## Testing Strategy

- **E2E tests**: Each `run.py` doubles as an E2E test — must produce MAE < 0.1 when run
- **Unit tests**: Existing `pytest python/tests/` must continue to pass
- **Import tests**: No remaining imports of `orion_compiler.models.{mlp,lenet,lola}` in library code

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix
- Update plan if implementation deviates from original scope

## Implementation Steps

### Task 1: Create `examples/mlp/` — first working example

- [x] Create `examples/mlp/model.py` — copy `MLP` class from `orion_compiler/models/mlp.py`, strip `__main__` block, keep only the model definition with `orion_compiler.nn` imports
- [x] Create `examples/mlp/train.py` — MNIST training script using `orion_compiler.core.utils.train_on_mnist()`, saves `weights.pt`
- [x] Create `examples/mlp/run.py` — full FHE pipeline following ARCH.md template: compile → keygen → encrypt → evaluate → decrypt → print MAE. Use `Parameters.from_logn(**params_dict)` pattern from working E2E test. Must be run from `examples/mlp/` directory (`from model import MLP` is a local import)
- [x] Create `examples/mlp/README.md` — architecture description, CKKS params rationale (logn=13, 9 levels, Quad, no bootstrap), expected MAE. Include run instructions: `cd examples/mlp && python run.py`
- [x] Run `cd examples/mlp && python run.py` end-to-end — must complete and print MAE < 0.1 (actual MAE: 0.001111)
- [x] Run `pytest python/tests/` — all existing tests pass (206 passed, 1 skipped)

### Task 2: Create `examples/lenet/`

- [x] Create `examples/lenet/model.py` — copy `LeNet` class, strip `__main__` block
- [x] Create `examples/lenet/train.py` — MNIST training script
- [x] Create `examples/lenet/run.py` — full FHE pipeline. Same CKKS params as MLP (9 levels sufficient, input_level=7, 0 bootstraps). 178 galois elements from Conv2d rotations produce ~1.3 GB key material.
- [x] Create `examples/lenet/README.md` — architecture, CKKS params, expected MAE. Include run instructions: `cd examples/lenet && python run.py`
- [x] Run `cd examples/lenet && python run.py` end-to-end — must complete and print MAE < 0.1 (actual MAE: 0.015088)
- [x] Run `pytest python/tests/` — all existing tests pass (206 passed, 1 skipped)

### Task 3: Create `examples/lola/`

- [ ] Create `examples/lola/model.py` — copy `LoLA` class, strip `__main__` block
- [ ] Create `examples/lola/train.py` — MNIST training script
- [ ] Create `examples/lola/run.py` — full FHE pipeline. LoLA is simpler than MLP (1 Conv + 1 FC). Start with MLP params, adjust if needed
- [ ] Create `examples/lola/README.md` — architecture, CKKS params, expected MAE. Include run instructions: `cd examples/lola && python run.py`
- [ ] Run `cd examples/lola && python run.py` end-to-end — must complete and print MAE < 0.1
- [ ] Run `pytest python/tests/` — all existing tests pass

### Task 4: Remove MNIST models from `orion_compiler/models/`

- [ ] Remove MLP, LeNet, LoLA imports from `orion_compiler/models/__init__.py` (keep AlexNet, VGG, ResNet, YOLO)
- [ ] Delete `orion_compiler/models/mlp.py`, `lenet.py`, `lola.py`
- [ ] Update `python/tests/models/test_mlp.py` — remove `import orion_compiler.models`, delete the skipped test (it's been replaced by the examples). Check if `python/tests/models/` directory has other test files; clean up if empty
- [ ] Update `CLAUDE.md` — fix the `from orion_compiler.models import MLP` example to use inline model definition pattern instead
- [ ] Grep for any remaining `orion_compiler.models.mlp`, `orion_compiler.models.lenet`, `orion_compiler.models.lola` imports — fix or remove
- [ ] Run `pytest python/tests/` — all tests pass

### Task 5: Add Phase 6 to ARCH.md

- [ ] Add Phase 6 section after Phase 5 in ARCH.md: "Phase 6: Bootstrapping and CIFAR-10 examples"
- [ ] Document scope: implement bootstrapping support in Go evaluator, then create `examples/alexnet/`, `examples/vgg/`, `examples/resnet/`, `examples/yolo/`
- [ ] Include acceptance checklist for Phase 6
- [ ] Update Phase 5 description text (lines 1216-1226) to reflect the split into Phase 5 (MNIST) and Phase 6 (CIFAR/YOLO)
- [ ] Update Phase 5 acceptance checklist to reflect partial scope (MNIST models only, `models/` directory kept for remaining models)
- [ ] Format with `npx prettier --write ARCH.md`

### Task 6: Verify acceptance criteria

- [ ] Verify all 3 examples have `model.py`, `train.py`, `run.py`, `README.md`
- [ ] Verify each `run.py` produces correct FHE output (MAE < 0.1)
- [ ] Verify `orion_compiler/models/` no longer contains `mlp.py`, `lenet.py`, `lola.py`
- [ ] Verify no imports of `orion_compiler.models.{mlp,lenet,lola}` anywhere
- [ ] Run full test suite: `pytest python/tests/`

### Task 7: Update documentation

- [ ] Update CLAUDE.md if new patterns discovered (e.g., example-specific CKKS params)
- [ ] Format all new Markdown files with `npx prettier --write`

## Technical Details

**CKKS Parameters (MNIST models, no bootstrap):**

| Model | logn | logq                                 | logp    | logscale | h    | ring_type           |
| ----- | ---- | ------------------------------------ | ------- | -------- | ---- | ------------------- |
| MLP   | 13   | [29,26,26,26,26,26]                  | [29,29] | 26       | 8192 | conjugate_invariant |
| LeNet | 13   | [29,26,26,26,26,26,26,26,26,26]      | [29,29] | 26       | 8192 | conjugate_invariant |
| LoLA  | 13   | TBD (similar to MLP — 1 Conv + 1 FC) | [29,29] | 26       | 8192 | conjugate_invariant |

**`run.py` pipeline flow:**

1. Instantiate model, optionally load `weights.pt`
2. Compute cleartext baseline on sample input
3. `Compiler(net, params).fit(dataloader).compile()` → `CompiledModel`
4. `Model.load(compiled.to_bytes())` → get `params_dict`, `manifest`, `input_level`
5. `Parameters.from_logn(**params_dict)` → Lattigo keygen → encrypt input
6. `Evaluator(params_dict, keys_bytes).forward(model, ct_bytes)` → result bytes
7. Decrypt → compare with cleartext → print MAE

**Key decision:** Use `Parameters.from_logn(**params_dict)` pattern (from working tests) rather than `ckks.Parameters.from_json()` (ARCH.md template). Both work, but `from_logn` is the established test pattern.

## Post-Completion

**Manual verification:**

- Train each model for a few epochs and verify FHE inference on trained weights produces classification-quality output
- Performance benchmarking: measure FHE inference time for each model (target < 10s for MNIST models)
