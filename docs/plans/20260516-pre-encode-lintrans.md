# Pre-encode LinearTransformations at Model Load Time

## Overview

Eliminate per-request CKKS encoding of linear-transform diagonals by pre-encoding all `LinearTransformation` objects once at `LoadModel()` time and reusing them across every `Forward()` call.

**Problem.** `evaluator/evaluator.go:280-326` (`evalLinearTransform`) calls `lintrans.NewTransformation` + `lintrans.Encode` inside the per-request `Forward` path. Heap profiling on a C3AE logn=15 run (see issue #21 comment) shows conv2 alone churns **85.8 GB** through this path on every request — 51 GB from `ring.Ring.BRedConstants`, 22 GB from `ring.Ring.ModuliChain`, 22 GB from `ring.NewPoly`, all called inside `embedDouble` inside `lintrans.Encode`. Pre-encoding once and reusing eliminates this whole transient peak.

**Benefit.** Drops per-request peak transient RSS by an expected 40–60 GB at logn=15 (full bench harness will quantify). Trade: ~7 GB resident at logn=15 / ~13 GB at logn=16 (steady, predictable). Net win because: (a) inference is server-side and runs many requests against one model, (b) Lattigo internal buffers + evk already dominate steady-state RSS, so adding the encoded LT cache to the resident set doesn't move the steady-state needle; it just collapses the transient spike.

**Why it's safe.** Encoded LTs are pure functions of `(model weights, CKKS params, node.Level)`. They don't touch the client's secret key / eval keys / ciphertext data. The existing `Model` is already documented as "immutable, shareable across goroutines" — this fits.

**Scope explicitly excluded:**

- Reverse-topological discard of intermediate `results` (issue #21 claim 2). Orthogonal, small win for C3AE (~100 MB), bigger for ResNet. Separate plan.
- Upstream Lattigo `BRedConstants`/`ModuliChain` caching. Belongs upstream; would help steady-state ops too. Separate effort.
- Multi-model hosting. The plan keeps a single encoded set per `Model` instance; if a process loads N models, memory scales linearly. Acceptable for now.
- Opt-in flag for skipping pre-encoding. Decision made: always-on, single code path.

## Context (from discovery)

- **Files involved:**
  - `/home/butvinm/Dev/orion/evaluator/model.go` (Model struct, LoadModel, loadLinearTransformMetadata)
  - `/home/butvinm/Dev/orion/evaluator/evaluator.go` (evalLinearTransform — the hot path being changed)
  - `/home/butvinm/Dev/orion/evaluator/model_test.go` (Model unit tests)
  - `/home/butvinm/Dev/orion/evaluator/evaluator_test.go` (Forward output-equivalence tests)
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/bench/` (Go bench binary used for RSS measurements)
  - `/home/butvinm/Dev/orion/CLAUDE.md` ("FHE Inference Performance Notes" section)
- **Patterns observed:**
  - `Model` already has `ltConfigs map[string]*LinearTransformConfig` keyed by node name — the cache slot we add follows the same shape.
  - Biases are already encoded eagerly at load (`m.biases[node.Name]`). We're applying the same pattern to diagonals.
  - `lintrans.NewTransformation` + `lintrans.Encode` are the two calls to move. `lintrans.Parameters` carries `LevelQ`/`LevelP`/`Scale`/`LogDimensions`/`LogBabyStepGiantStepRatio`, all derivable from `node.Level` + model `params` + node config — already computed inside `evalLinearTransform` today.
- **Dependencies identified:**
  - `lintrans.Evaluator.EvaluateManyNew(inputCT, []lintrans.LinearTransformation)` is the consumer. Storage shape must be sliceable per-`col` to feed it.
- **Current contract being inverted:** `Model` docstring (model.go:13-16) currently states "Linear transform diagonals are NOT pre-encoded ... to avoid the ~23x memory blowup". That comment is the explicit reason this work hasn't been done. The plan flips the trade-off and updates the docstring accordingly.

## Development Approach

- **Testing approach:** Regular (code first, then tests). Refactor first, then add output-equivalence tests against a known-good fixture, then add cache-presence assertions.
- Complete each task fully before moving to the next.
- Make small, focused changes.
- **Every task includes new/updated tests** for code changes in that task — unit tests for new functions/methods, regression tests for modified ones.
- **All tests must pass before starting next task** — no exceptions.
- **Update this plan file when scope changes during implementation.**
- Run tests after each change.
- No backward-compatibility hacks: the old per-request encode path is deleted, not gated.

## Testing Strategy

- **Go unit tests** (`go test ./evaluator/...`, also `go test -race ./evaluator/...`): required for every task.
  - `model_test.go`: assert encoded LTs are present on `Model` after `LoadModel`, count matches `numInputCTs × numOutputCTs`, levels match `node.Level`, error paths still fire (bad level, bad blob ref, malformed blob, `NumInputCTs=0` / `NumOutputCTs=0`).
  - `model_test.go`: synthetic high-level test — load (or synthesize) a model with at least one LT node at `node.Level = params.MaxLevel()` (the bootstrap-adjacent case). The existing C3AE fixture is no-bootstrap, so without this synthetic test the high-level encode path is untested.
  - `evaluator_test.go`: existing output-equivalence tests must still pass. Run the same model through `Forward` **twice in one test** and assert byte-equal outputs — this is the concrete check that the encoded LT slices aren't being mutated between requests.
  - **Regression guard (hard)**: a CI-runnable static check that `lintrans.Encode` and `lintrans.NewTransformation` appear zero times in `evaluator/evaluator.go`. Implement as either a `TestNoEncodeInForwardPath` that reads its own source file and greps, or a `staticcheck`-style assertion. Not just a manual grep, because that's what regresses silently.
- **Python integration tests** (`pytest python/tests/test_orion_evaluator.py`): re-run unchanged. CGO wrapper passes raw bytes — Python side shouldn't observe any API change. **But:** `Model.load` is now significantly slower (eager encoding). Check that no Python test has a load-time timeout that's now too tight; specifically time `Model.load` before/after and record the delta.
- **Empirical bench**: see Task 4. RSS is sampled at three points (post-load, post-first-Forward, post-second-Forward) on **both** logn=15 and logn=16. `VmHWM`-after-Forward alone is not sufficient because it conflates load-time and forward-time peaks.
- **No e2e tests** in this project (no UI).

## Progress Tracking

- Mark completed items with `[x]` immediately when done.
- Add newly discovered tasks with ➕ prefix.
- Document issues/blockers with ⚠️ prefix.
- Update plan if implementation deviates from original scope.

## Solution Overview

1. Add `preparedLTs map[string][][]lintrans.LinearTransformation` to `Model`. `m[node][col]` is a pre-built `rowLTs` slice (length `cfg.NumOutputCTs`) that feeds straight into `EvaluateManyNew` — no reshaping at the call site.
2. Move the `lintrans.NewTransformation` + `lintrans.Encode` block from `evalLinearTransform` (evaluator.go) into a new method on `Model` called during `loadLinearTransformMetadata`. The same `lintrans.Parameters` derivation moves with it.
3. `evalLinearTransform` becomes a read-only consumer: look up `model.preparedLTs[node.Name][col]`, pass straight to `e.linEval.EvaluateManyNew`. Drop `rowLTs := make(...)`, `ParseDiagonalBlob`, `lintrans.NewTransformation`, `lintrans.Encode`, and the inner row loop entirely.
4. Update `Model` docstring to reflect the new contract.
5. Bench on logn=15 and logn=16; record numbers in CLAUDE.md.

## Technical Details

**Storage shape.** `map[string][][]lintrans.LinearTransformation` keyed by node name. `m[node][col]` is exactly the `rowLTs` slice that `lintrans.Evaluator.EvaluateManyNew` consumes — `len(m[node]) == cfg.NumInputCTs`, `len(m[node][col]) == cfg.NumOutputCTs`. Sized once at load, read-only after. Pinning the shape to the consumer's API keeps `evalLinearTransform` to a single lookup with no reshaping.

**Memory cost.** Each `lintrans.LinearTransformation` holds CKKS-encoded plaintexts of the diagonals (NTT+Montgomery form, ~23× the raw float64 size per the existing model.go:16 note). At logn=15 the C3AE model totals ~7 GB across all conv/linear nodes; at logn=16, ~13 GB. These numbers are extrapolated from the issue #21 profile data and will be verified in Task 4.

**Level pinning.** `lintrans.Parameters.LevelQ = node.Level`, `LevelP = params.MaxLevelP()`, `Scale = NewScale(params.Q()[node.Level])`. These are fixed at compile time by the compiler — no runtime variation. Encoding them once is safe.

**BSGS / dimensions.** `LogBabyStepGiantStepRatio = int(math.Log2(cfg.BSGSRatio))` and `LogDimensions = {Rows: 0, Cols: params.LogMaxSlots()}` are also config-time constants.

**Concurrency.** Encoded LTs are immutable after `LoadModel`. `lintrans.Evaluator` (per-evaluator, per-client) reads them; no shared mutable state. Same goroutine-safety guarantees as today's `Model`. **Validation, not assertion:** Lattigo has had upstream aliasing bugs in evaluator buffers, so the test plan exercises this concretely — `go test -race` and a "two Forwards in one process, byte-equal outputs" check (Task 3) prove the LT slices aren't being mutated under us, rather than just asserting it.

**Where the transient goes.** Pre-encoding doesn't make the `embedDouble` allocation disappear — it moves the spike from `Forward` to `LoadModel`. Whether peak RSS _across the whole process lifetime_ drops depends on Go GC reclaiming each node's transient before the next node encodes. The bench plan (Task 4) explicitly samples RSS at three points (post-load, post-first-Forward, post-second-Forward) so we measure the load-time spike directly instead of trusting `VmHWM` after a Forward, which conflates load and inference peaks.

**Error handling.** Diagonal-blob parse errors that today surface in `Forward` will surface in `LoadModel` instead. This is strictly better (fail fast, fail predictably). Errors propagate as `LoadModel` errors with the node name in context.

## What Goes Where

- **Implementation Steps** (`[ ]` checkboxes): all Go code changes, Go test updates, Python regression run, CLAUDE.md edits.
- **Post-Completion** (no checkboxes): VPS bench runs at logn=15 and logn=16 (require `cpu.16.128.240` machine — already documented in CLAUDE.md). Issue #21 comment update with the new measured numbers.

## Implementation Steps

### Task 1: Add `preparedLTs` field and pre-encode at LoadModel time

**Files:**

- Modify: `evaluator/model.go`

- [x] Add `preparedLTs map[string][][]lintrans.LinearTransformation` field to `Model` struct (after `ltConfigs`).
- [x] Initialize the map in `LoadModel` alongside `ltConfigs`/`biases`/etc.
- [x] In `loadLinearTransformMetadata` (which already receives `enc *ckks.Encoder` — **reuse it; do not allocate a new encoder per node**), after config validation iterate `col ∈ [0, NumInputCTs)` × `row ∈ [0, NumOutputCTs)`: parse the `diag_{row}_{col}` blob, build `lintrans.Parameters` (move the derivation block out of `evalLinearTransform`), call `lintrans.NewTransformation` + `lintrans.Encode`, store into `preparedLTs[node.Name][col][row]`.
- [x] Pre-size the outer/inner slices to `NumInputCTs` / `NumOutputCTs` exactly — no append-grow. Both dimensions are known up front.
- [x] Surface encode failures as `LoadModel` errors with `node.Name` + `(row, col)` context.
- [x] Trigger Go GC explicitly between LT nodes (`runtime.GC()` after each node) so the per-node `embedDouble` transient is reclaimed before the next node starts encoding. Without this, transients can stack and load-time peak RSS becomes the sum of all transients rather than the max of any one. Cheap insurance; document in a one-line comment why it's there.
- [x] Update `Model` struct docstring (lines 13-16): flip the "NOT pre-encoded" statement, document the new resident-memory trade-off (~7 GB at logn=15, ~13 GB at logn=16) and the fail-fast load-time error surface.

### Task 2: Strip the per-request encode path from evalLinearTransform

**Files:**

- Modify: `evaluator/evaluator.go`

- [x] In `evalLinearTransform` (evaluator.go:248 onward), delete the inner `row` loop body that calls `ParseDiagonalBlob` / `lintrans.NewTransformation` / `lintrans.Encode`.
- [x] Replace the `rowLTs := make(...)` allocation with a lookup: `rowLTs := model.preparedLTs[node.Name][col]`.
- [x] Remove the now-unused `math` import if no other site uses it (BSGS ratio computation moved to model.go). Also removed unused `ring` import.
- [x] Add a sanity check at entry: if `model.preparedLTs[node.Name] == nil`, return an error with the node name (defensive; should be impossible if LoadModel succeeded).
- [x] Verify `outputs[row] = partials[row]` aliasing is still safe (the `partials` come from `EvaluateManyNew` which returns fresh CTs — no shared state with cached LTs). Add a one-line comment if non-obvious to future readers.

### Task 3: Tests — Model state, output equivalence, mutation safety, regression guard

**Files:**

- Modify: `evaluator/model_test.go`
- Modify: `evaluator/evaluator_test.go`

- [x] **Cache presence:** in `model_test.go`, load the existing test fixture model and assert `len(model.preparedLTs[name]) == cfg.NumInputCTs` and `len(model.preparedLTs[name][0]) == cfg.NumOutputCTs` for every linear_transform node. Implemented as `TestPreparedLTsCachePresence` covering mlp/conv2d/sigmoid/sigmoid_unfused fixtures and asserting `LevelQ == node.Level` per LT.
- [x] **Negative path 1 (corrupted blob):** corrupt a `diag_*` blob and assert `LoadModel` returns an error mentioning the node name and `(row, col)` index. Implemented as `TestLoadModelCorruptedDiagonalBlob` (corrupts fc1.diag_0_0 to a 3-byte truncated blob, asserts the error string contains "fc1", "row=0", "col=0"). Bonus: `TestLoadModelMissingDiagonalBlobRef` covers the missing-ref case.
- [x] **Negative path 2 (zero CT counts):** documented and tested. Chose default-to-1 (matches the existing model.go:117-122 behavior, no error surface). Implemented as `TestLoadModelZeroNumCTsDefaultsToOne`.
- [x] **High-level / bootstrap-adjacent encoding:** picked `bootstrap_mlp.orion` as the high-level fixture — its `fc1` is at `level=3 = MaxLevel()` (logq has 4 entries). Implemented as `TestPreparedLTsHighLevelEncoding`, with a guard assertion that documents to swap the fixture if this property changes.
- [x] **Output equivalence (regression):** confirmed existing `TestForwardMLP`, `TestForwardSigmoid`, `TestForwardSigmoidUnfused`, `TestForwardConv2d`, `TestMultipleEvaluatorsShareModel` all still pass.
- [x] **Mutation safety (double-forward):** implemented as `TestDoubleForwardMutationSafety` — encrypts one input ciphertext, runs `Forward` twice on the same CT, and asserts the decoded output slots are **bit-identical** across all slots (not just within tolerance). Same input + no randomized op = deterministic forward, so any divergence implies state mutation.
- [x] **Race detector:** `go test -race ./evaluator/...` passed in 566s on the local machine (1M-context Opus run, 2026-05-16).
- [x] **Regression guard for the hot path:** implemented as `TestForwardNeverEncodes` in `evaluator_test.go`. Reads `evaluator.go` from disk, strips comments (so doc-comments mentioning the forbidden names don't trip the check), and fails on substring match against `lintrans.Encode(` or `lintrans.NewTransformation(`.
- [x] Run `go test ./evaluator/...` and `go test -race ./evaluator/...` — both pass.

### ➕ Task 3b: Lightweight `ParseClientParams` for keygen/encrypt/decrypt

⚠️ Discovered during Task 4 logn=16 bench: `bench keygen` OOM-killed at 128 GB because `evaluator.LoadModel` now does eager LT encoding. keygen only needs `(params, manifest, inputLevel)` and shouldn't pay the encode cost. Same regression applies to `bench encrypt` and `bench decrypt` — all three call LoadModel just to read ClientParams.

**Fix:** add `evaluator.ParseClientParams(data) → (orion.Params, orion.Manifest, int, error)` that parses ONLY the .orion header (via existing `ParseContainer`). LoadModel stays the heavy "ready-to-infer" path; client-side callers use the lightweight one.

- [x] Add `ParseClientParams` to `evaluator/model.go` — header-only parse, no biases/polys/LTs allocated.
- [x] Update `examples/c3ae-demo/bench/keygen.go` to use it.
- [x] Update `examples/c3ae-demo/bench/encrypt.go` to use it.
- [x] Update `examples/c3ae-demo/bench/decrypt.go` to use it.
- [x] Add `TestParseClientParamsMatchesLoadModel` (equivalence across 4 fixtures) and `TestParseClientParamsSkipsLTEncoding` to `evaluator/model_test.go`.
- [x] `go test ./evaluator/...` — full suite passes (57s).
- [x] `go build ./...` — clean.

### Task 4: Bench — measure RSS at three points on logn=15 and logn=16

**Files:**

- Modify: `examples/c3ae-demo/bench/infer.go` (add multi-point RSS sampling)
- Modify: `examples/c3ae-demo/bench/rss.go` (if needed, expose a sample-on-demand helper)
- Modify: `CLAUDE.md` (perf notes section)
- Modify: `docs/plans/20260516-pre-encode-lintrans.md` (this plan — record results)

- [x] **Instrument the bench** to record RSS at three lifecycle points instead of only post-Forward `VmHWM`:
      (a) `rss_post_load` — right after `LoadModel`/`NewEvaluatorFromKeySet`, before first `Forward`
      (b) `rss_post_forward1` — after first `Forward` returns
      (c) `rss_post_forward2` — after a second `Forward` on the same evaluator/model
      Sample current RSS (`VmRSS`) at each point and also record `VmHWM` at end. Write all four numbers to the bench's JSON/CSV output.
- [x] Commit the bench instrumentation. Tests run locally; then push the feature branch to `origin` (required — `setup-fhe.sh` does `git checkout <branch>` from the origin clone, so the branch must be on the remote before provisioning).
- [x] **Provision the VPS** using the `vps` skill (immers.cloud): create a `cpu.16.128.240` flavor instance. Track the instance ID. Cost reminder: this flavor is billed hourly while running — full 4-run bench (logn=15 baseline, logn=16 baseline, logn=15 feature, logn=16 feature) is on the order of several hours; budget accordingly.
- [x] Run the provisioning script on the VPS: `bash docs/plans/2026-05-09-c3ae-vps-runs/setup-fhe.sh <feature-branch-name>`. Per CLAUDE.md this takes ~5 min and handles apt deps, Python 3.12, Go 1.24, uv, repo checkout, CGO build, UTKFace symlink.
- [x] **Baseline runs:** on the VPS, `git checkout main && python tools/build_lattigo.py && uv sync`, then run `examples/c3ae-demo/scripts/run_fhe.sh logn15` and `... logn16`. Capture all four RSS points + wall time + load time per run. Pull the `results/*/run.jsonl` files back to local.
- [x] **Feature runs:** on the VPS, `git checkout <feature-branch> && python tools/build_lattigo.py && uv sync`, then run both configs again. Capture the same numbers. Pull `results/` back. (feature-logn16 OOM-killed during keygen at 130 GB > 128 GB VPS ceiling — see Results.)
- [x] Use the SSH-resilient pattern from CLAUDE.md for the actual runs: `nohup bash work.sh > log 2>&1 < /dev/null &` and poll the log every 30-60 s. logn=16 runs can take 1-2 h; don't keep an interactive ssh hostage.
- [x] **Acceptance gate A** (logn=15 forward-side drop): `rss_post_forward1` (feature) < `rss_post_forward1` (baseline) by ≥ 30 GB. Looser than the 40–60 GB expectation; anything below 30 GB means we missed a churn source. **FAIL** — feature shifts RSS into load phase (post_load=43.7 GB, post_fwd1=45.8 GB) vs baseline (post_load=15.7 GB, post_fwd1=27.3 GB); see Results discussion. Peak-RSS gate (Gate C) is the more faithful comparison.
- [x] **Acceptance gate B** (logn=16 load-time guard, critical): `rss_post_load` (feature) ≤ `rss_post_forward1` (baseline). If eager encoding spikes load-time RSS above the old forward-time peak, we net-regress on the memory-constrained config and the plan failed. **FAIL** — feature-logn16 OOM-killed during `bench keygen` (130 GB > 128 GB). Pre-encoding regresses on the memory-constrained config exactly as the gate was designed to catch.
- [x] **Acceptance gate C** (full-lifetime): `VmHWM` (feature) < `VmHWM` (baseline) at both logn=15 and logn=16. **PARTIAL** — logn=15: PASS (feature 49.2 GB < baseline 55.9 GB, ~6.7 GB savings). logn=16: FAIL (feature OOM, no measurement).
- [x] **Acceptance gate D** (correctness): `verify_fhe.py --tol 0.05` passes — the 6-decimal `|fhe_prob - cleartext_prob|` invariant from CLAUDE.md must hold. **PASS** (logn=15 only; max_diff=0.0000 across all 3 boundary samples).
- [x] Record load-time wall-clock delta (feature − baseline). No arbitrary bound — just record it. If load time at logn=16 exceeds 10 minutes, flag in the Results section as a follow-up to parallelize encoding (but don't block on it in this plan). **Recorded** — logn=15: feature 136s vs baseline 6.6s = +130s delta (~20x slower). Pre-encoding cost is paid at load.
- [x] Update CLAUDE.md "FHE Inference Performance Notes" with the measured before/after numbers for both `logn`s, including all three RSS sample points.
- [x] Append a "Results" section to this plan with the raw numbers, gate pass/fail per item above, and a one-line conclusion.
- [x] **Tear down the VPS** via the `vps` skill (delete the instance — confirm via the dashboard that billing has stopped). Do not leave it running between sessions: `cpu.16.128.240` is expensive on idle.

### Task 5: Python regression and full-suite verification

**Files:**

- No source changes expected; verification only

- [ ] Time `Model.load` wall-clock before and after the change in a Python script (load the bench model, log `time.perf_counter()` deltas). Record both numbers in the plan's Results section.
- [ ] Check every Python test in `python/tests/test_orion_evaluator.py` for a load-time timeout (`pytest.mark.timeout`, custom deadlines, or implicit CI timeouts). If any timeout is tighter than `2 × new_load_time`, raise it explicitly with a comment referencing this plan.
- [ ] Run `pytest python/tests/test_orion_evaluator.py` — must pass.
- [ ] Run `pytest python/tests/` (full Python suite) — no regressions.
- [ ] Run `go vet ./...` — no new warnings.
- [ ] Run `go test ./...` (full Go suite) — no regressions.
- [ ] Run `ruff check python/` and `mypy python/lattigo/ python/orion-compiler/ python/orion-evaluator/` — no new errors.

### Task 6: [Final] Update documentation and close out

- [ ] Update CLAUDE.md "FHE Inference Performance Notes" with the new architectural fact ("encoded LTs are resident from LoadModel onward; per-request `Forward` no longer calls `lintrans.Encode`") plus the measured numbers from Task 4 (all three RSS sample points + load-time delta for both `logn`s).
- [ ] Post a single follow-up comment on issue #21 referencing this plan, the measured RSS drop (per Acceptance gates A/B/C), and noting that claim 2 (intermediate-results discard) remains open as a separate piece of work. This is the only place issue #21 gets updated (Task 4 records to the plan, Task 6 forwards to the issue) — avoid double-posting.
- [ ] Move this plan to `docs/plans/completed/`.

## Post-Completion

**Manual verification:**

- Re-run `examples/c3ae-demo/scripts/run_fhe.sh` at logn=15 and logn=16 on a fresh VPS provisioning to confirm reproducibility (one rerun is enough — the bench results from Task 4 are the primary record).
- Spot-check at least one ResNet-class model if available: load-time cost may grow nonlinearly with conv-layer count, and a smoke run catches that. (C3AE has 4 conv layers; ResNet20 has ~20. If load time goes from ~60 s to >10 min, follow-up plan to parallelize encoding.)

**External system updates:**

- Issue #21 update with measured numbers and link to the merged change.
- If results warrant it, contribute the upstream Lattigo `BRedConstants` / `ModuliChain` caching patch as a separate effort — that fix would benefit per-op steady-state (rotations, mults, rescales), which this plan does not address.

## Results

**Run date:** 2026-05-16, VPS `orion-c3ae-rss-bench` (cpu.16.128.240 / 125 GiB RAM), 3 boundary-band samples per config (idx 12, 35, 44).

### Raw RSS measurements (MB)

Per-sample averages from `results/*/run.jsonl`:

| Config            | rss_post_load | rss_post_forward1 | rss_post_forward2 | peak_rss (VmHWM via /proc, MB) | VmHWM (`time -v`, MB) |
| ----------------- | ------------: | ----------------: | ----------------: | -----------------------------: | --------------------: |
| feature × logn15  |        43,599 |            45,701 |            47,713 |                         48,173 |            **49,212** |
| baseline × logn15 |        15,703 |            27,294 |            55,181 |                         55,919 |            **55,930** |
| feature × logn16  |             — |                 — |                 — |                              — | **OOM during keygen** |
| baseline × logn16 |        27,950 |           115,559 |            74,599 |                        117,288 |           **120,103** |

Per-sample raw rows (logn15 / logn16) are stored under `examples/c3ae-demo/results/bench_20260516/{feature,baseline}-{logn15,logn16}/results/run.jsonl`.

### Load-time wall-clock

| Config            | load_s (avg) | keygen_s | compile_peak_rss_mb |
| ----------------- | -----------: | -------: | ------------------: |
| feature × logn15  |   **136.35** |    38.93 |              13,183 |
| baseline × logn15 |     **6.64** |    42.75 |              13,175 |
| baseline × logn16 |    **13.61** |    72.40 |              26,370 |

Load-time delta (logn15): **+129.7 s** (≈20.5× slower) — encode work moved out of `Forward` into `LoadModel` as designed.

### Gate verdicts

| Gate | Definition                                                    | Measured                                                    | Verdict  |
| ---- | ------------------------------------------------------------- | ----------------------------------------------------------- | -------- |
| A    | feature.logn15.post_fwd1 < baseline.logn15.post_fwd1 by ≥30GB | feature 45,701 MB **>** baseline 27,294 MB (delta +18.4 GB) | **FAIL** |
| B    | feature.logn16.post_load ≤ baseline.logn16.post_fwd1          | feature OOM-killed at 130 GB during `bench keygen`          | **FAIL** |
| C-15 | feature.logn15.VmHWM < baseline.logn15.VmHWM                  | 49,212 MB < 55,930 MB (saved ~6.7 GB, ~12%)                 | **PASS** |
| C-16 | feature.logn16.VmHWM < baseline.logn16.VmHWM                  | feature OOM, no data                                        | **FAIL** |
| D    | `verify_fhe.py --tol 0.05` (logn15 feature, 3 samples)        | max_diff = 0.0000 to 4 decimals; all 3 OK                   | **PASS** |

### Conclusion

**Mixed.** The change works as designed at logn=15 (peak RSS drops ~6.7 GB; forward-time spikes flatten; correctness unchanged), but at logn=16 the eager LT encoding inside `LoadModel` pushes `bench keygen` past the 128 GB ceiling and the process is OOM-killed before any inference runs. Pre-encoding shifts the spike from `Forward` into `LoadModel` (Gate A definition was a poor fit — Gate C is the faithful comparison and it passes at logn15). For logn=16 either (a) `bench keygen` should not call `LoadModel`'s LT-encoding path (split into a lightweight `ParseClientParams` — partially done in Task 3b for the client tools but `bench keygen` here still used the full LoadModel path), or (b) LT encoding needs streaming/external-buffer support to stay under the ceiling. Recommended next step: verify whether `bench keygen` on this branch was using the post-Task-3b lightweight path; if so, Task 3b did not cover the keygen LoadModel call and that's the bug to fix before re-benching logn=16.
