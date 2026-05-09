# C3AE Experiments — Cleartext FNR/FPR + FHE Timing/RSS Benchmarks

## Overview

Build a self-contained experiment harness under `examples/c3ae-demo/experiments/` that produces two comparable measurements for the C3AE age-verification model:

1. **Experiment 1 — Cleartext quality.** Train a true-ReLU variant of C3AE, evaluate it side-by-side with the existing Quad (x²) variant on UTKFace test split. Report accuracy/FPR/FNR over (a) the full test split and (b) the 16–20 age boundary band.
2. **Experiment 2 — FHE cost.** Run the Quad variant under two CKKS configurations, both **without bootstrap**:
   - **`logn15`**: existing demo params from `/home/butvinm/Dev/orion/examples/c3ae-demo/generate_model.py:23-29` — `logn=15, logq=[51]+[40]*15, logp=[50]*4`. LogQP=851. Proven config from the current demo.
   - **`logn16`**: `logn=16, logq=[55]+[40]*15, logp=[55]*6`. LogQP=985. **Same multiplicative depth (15 levels) as the logn=15 baseline** — the experiment isolates the cost of doubling the ring degree at fixed network depth.
     Measure forward time and peak RSS for 3 fixed boundary samples per config. All FHE work runs in a **Go-only** binary (`bench`) to eliminate Python wrapper overhead from RSS accounting.

Final deliverable: `examples/c3ae-demo/experiments/results/results.md` with two tables (cleartext quality, FHE cost).

## Context (from discovery)

**Existing reference files:**

- `/home/butvinm/Dev/orion/examples/c3ae-demo/model.py` — current Quad C3AE (`on.Module`)
- `/home/butvinm/Dev/orion/examples/c3ae-demo/train.py` — training loop, `UTKFaceDataset`, `evaluate()`
- `/home/butvinm/Dev/orion/examples/c3ae-demo/run_fhe.py` — Python end-to-end FHE pipeline (reference for what `bench` replicates in Go)
- `/home/butvinm/Dev/orion/examples/c3ae-demo/server/main.go` + `go.mod` — Go HTTP server, model for sub-module layout
- `/home/butvinm/Dev/orion/examples/c3ae-demo/generate_model.py` — current compile script + the `logn15` params source
- `/home/butvinm/Dev/orion/evaluator/model.go` — `Model.LoadModel`, `ClientParams()` returning `params, manifest, input_level`
- `/home/butvinm/Dev/orion/evaluator/evaluator.go` — `Evaluator.Forward(ctsIn) → ctsOut`

**Existing weights:**

- `/home/butvinm/Dev/orion/examples/c3ae-demo/weights.pth` — Quad variant, already trained. **Will be reused as `out/weights_fhe.pth`.**

**Patterns observed:**

- Sub-modules use their own `go.mod` (`examples/c3ae-demo/server/go.mod` references root with `github.com/butvinm/orion/v2 v2.1.3`)
- Streaming results pattern is novel for this repo — most prior runs were one-shot.
- `.orion` files embed CKKS params; Go reads them via `evaluator.Model.LoadModel` + `ClientParams()`. **Go never reads Python config files.**

## Development Approach

- **No automated test files.** This is research/experiment code with a small surface area. Every task ends with a **manual verification** step listing the exact commands to run that prove the task works. No `_test.go` files, no `test_*.py` files. The agent runs the verification commands during implementation and reports the output.
- Complete each task fully (including manual verification) before moving to the next.
- **Existing repo tests must keep passing.** Run `pytest python/tests/` and `go test ./evaluator/...` after any change that touches imported code. We do **not** modify imported code, so this should stay green.
- Use `venv` for all Python work — never install to system Python.
- All file references in this plan use absolute `/full/path` or `/full/path:line` format.

## Progress Tracking

- Mark completed items with `[x]` immediately when done.
- Add newly discovered tasks with `➕` prefix.
- Document issues/blockers with `⚠️` prefix.
- Update this plan file if scope changes.

## What Goes Where

- **Implementation Steps**: writing code/scripts/configs in this repo.
- **Post-Completion**: actually running the experiments on the VPS (~hours of compute), writing the final report.

## Implementation Steps

### Task 1: Scaffold experiment directory + gitignore + README skeleton

**Files:**

- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/README.md`
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/.gitignore`
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/results/.gitkeep`

- [x] create `examples/c3ae-demo/experiments/` directory with subdirs `models/`, `bench/`, `scripts/`, `results/`
- [x] write `experiments/.gitignore` ignoring `out/`, `results/*` (except `results/results.md` and `results/.gitkeep`), `__pycache__/`, `*.pyc`, `bench/bench` (the compiled Go binary)
- [x] write `experiments/README.md` skeleton: goal, dataset, two experiments summary, how-to-run pointers (filled out fully in the final task)
- [x] **manual verify**: `git status` shows only the new directory and skeleton files; no spurious files

### Task 2: Add `c3ae.py` (true-ReLU) and `c3ae_fhe.py` (Quad)

**Files:**

- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/c3ae.py`
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/c3ae_fhe.py`
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/__init__.py`

- [x] write `c3ae.py` as a plain `torch.nn.Module` mirroring the architecture of `/home/butvinm/Dev/orion/examples/c3ae-demo/model.py:13-86` but using `torch.nn.{Conv2d, BatchNorm2d, AvgPool2d, Linear, ReLU(inplace=False), Flatten}`. Same constructor `C3AE(img_size=64, first_stride=2)` and same `forward(x)` shape behavior.
- [x] write `c3ae_fhe.py` by copying `/home/butvinm/Dev/orion/examples/c3ae-demo/model.py:13-86` verbatim — already correct (orion_compiler.nn + Quad).
- [x] empty `__init__.py` so `from models.c3ae import C3AE` and `from models.c3ae_fhe import C3AE as C3AE_FHE` work.
- [x] **manual verify**: from `experiments/`, run

  ```sh
  python -c "
  import torch
  from models.c3ae import C3AE as ReLUNet
  from models.c3ae_fhe import C3AE as QuadNet
  for net in [ReLUNet(), QuadNet()]:
      net.eval()
      out = net(torch.zeros(1,3,64,64))
      assert out.shape == (1,1), out.shape
  print('OK')
  "
  ```

  Must print `OK`.

### Task 3: Add `params.py` with both CKKS configurations

**Files:**

- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/params.py`

- [x] write `params.py` with `PARAMS: dict[str, CKKSParams] = {...}` containing exactly two entries:
  - `"logn15"` — copy from `/home/butvinm/Dev/orion/examples/c3ae-demo/generate_model.py:23-29`:
    ```python
    CKKSParams(
        logn=15,
        logq=[51, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
        logp=[50, 50, 50, 50],
        log_default_scale=40,
        ring_type="standard",
    )
    ```
    LogQ=651, LogP=200, LogQP=851 ≤ 881 (128-bit @ logn=15).
  - `"logn16"`:
    ```python
    CKKSParams(
        logn=16,
        logq=[55] + [40]*15,
        logp=[55]*6,
        log_default_scale=40,
        ring_type="standard",
    )
    ```
    LogQ=655, LogP=330, LogQP=985 ≤ 1770 (128-bit @ logn=16). **Same depth (15 levels) as `logn15`** for direct ring-degree comparison.
- [x] add a module-level docstring documenting the LogQ/LogP arithmetic and the security bound for each config.
- [x] **manual verify**: from `experiments/`, run

  ```sh
  python -c "
  from models.params import PARAMS
  assert set(PARAMS) == {'logn15', 'logn16'}
  a = PARAMS['logn15']
  b = PARAMS['logn16']
  assert sum(a.logq) == 651 and sum(a.logp) == 200, (sum(a.logq), sum(a.logp))
  assert sum(b.logq) == 655 and sum(b.logp) == 330, (sum(b.logq), sum(b.logp))
  assert a.boot_logp is None and b.boot_logp is None
  print('OK')
  "
  ```

  Must print `OK`.

### Task 4: Add `train.py` with `--variant {relu,fhe}` flag

**Files:**

- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/train.py`

- [x] write `train.py` adapted from `/home/butvinm/Dev/orion/examples/c3ae-demo/train.py:1-183`. Same `UTKFaceDataset`, same 70/15/15 split with `manual_seed(42)`, same training loop, same `asymmetric_loss`. Add `--variant {relu,fhe}` selecting `from models.c3ae import C3AE` vs `from models.c3ae_fhe import C3AE`. Default output: `out/weights_<variant>.pth`.
- [x] preserve all existing CLI flags (`--epochs`, `--batch-size`, `--lr`, `--fpr-weight`, `--max-grad-norm`, `--data-dir`, `--stride`).
- [x] **import-path discipline**: scripts under `models/` are run as `python -m models.train ...` from `experiments/` so `from models.c3ae import ...` resolves. Document this in the README.
- [x] **manual verify**: from `experiments/`, run

  ```sh
  python -m models.train --help
  python -c "from models import train"
  ```

  Both must succeed; `--help` should list `--variant {relu,fhe}`.

### Task 5: Add `prep_input.py` for sample preprocessing → .bin

**Files:**

- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/prep_input.py`

- [x] write `prep_input.py` that:
  - reproduces the 70/15/15 test split via the same `manual_seed(42)` + `UTKFaceDataset`
  - takes either `--idx N` (single sample) or `--boundary-band` (dump the first 3 test samples with `16 ≤ age ≤ 20` in iteration order)
  - writes `out/inputs/sample_<idx>.bin` as a raw little-endian `float64` blob of exactly 12288 values (3×64×64 image, normalized to `[-1, 1]` same as training)
  - writes/updates `out/inputs/ground_truth.csv` with columns `idx,age,is_adult`
- [x] **decision documented**: write `float64` (not `float32`) to match Lattigo's encoder input type and avoid casting in Go.
- [x] **manual verify**: from `experiments/` with the UTKFace dataset present at `./data/UTKFace`, run

  ```sh
  python -m models.prep_input --boundary-band
  ls -la out/inputs/
  # Must show: sample_<idx>.bin × 3, ground_truth.csv
  stat -c '%s' out/inputs/sample_*.bin
  # Each must be exactly 98304 bytes (12288 * 8)
  cat out/inputs/ground_truth.csv
  # Must show 3 rows, all with 16 <= age <= 20
  ```

  Verified with synthetic UTKFace fixture (real dataset only on VPS): 40 dummy 64x64 RGB jpg files spread across ages 5..25 with the boundary band populated. Output: 3 × `sample_<idx>.bin` files of exactly 98304 bytes, `ground_truth.csv` with 3 rows all in `[16, 20]`. `--idx` mode and rerun-idempotency also confirmed.

### Task 6: Add `compile.py` for `CKKSParams` → `.orion`

**Files:**

- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/compile.py`

- [x] write `compile.py` adapted from `/home/butvinm/Dev/orion/examples/c3ae-demo/generate_model.py:1-72`. Flags: `--variant fhe`, `--config <name>`, `--weights out/weights_fhe.pth`, `--output out/<config>/model.orion`. Looks up `PARAMS[args.config]`. Uses `tracemalloc` to record peak Python memory. Writes `out/<config>/compile.json` with `{"compile_s": ..., "compile_peak_rss_mb": ..., "model_bytes": ...}`.
- [x] **manual verify**: from `experiments/`, run

  ```sh
  python -m models.compile --help
  ```

  Must list flags `--variant`, `--config`, `--weights`, `--output`. Full compilation runs in Post-Completion (slow + needs trained weights).

### Task 7: Add `eval.py` for cleartext FPR/FNR/Acc → CSV

**Files:**

- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/eval.py`

- [x] write `eval.py` that:
  - loads both `out/weights_relu.pth` and `out/weights_fhe.pth` (skipping any missing variant with a printed warning, not erroring)
  - reproduces test split via `manual_seed(42)`
  - computes FPR/FNR/Accuracy with two scopes: `overall` (full test set) and `boundary` (`16 ≤ age ≤ 20`)
  - writes `results/cleartext.csv` with columns `variant,scope,n,fpr,fnr,accuracy`
  - decision rule: `sigmoid(logit) >= 0.5 → adult`
  - factor metric computation into a helper `compute_metrics(probs: np.ndarray, targets: np.ndarray) -> dict`
- [x] **manual verify**: from `experiments/`, run

  ```sh
  python -c "
  import numpy as np
  from models.eval import compute_metrics
  # all correct
  m = compute_metrics(np.array([0.1, 0.9]), np.array([0., 1.]))
  assert m['accuracy'] == 1.0 and m['fpr'] == 0.0 and m['fnr'] == 0.0, m
  # all wrong
  m = compute_metrics(np.array([0.9, 0.1]), np.array([0., 1.]))
  assert m['accuracy'] == 0.0 and m['fpr'] == 1.0 and m['fnr'] == 1.0, m
  # one false positive
  m = compute_metrics(np.array([0.6, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]), np.array([0., 0., 0., 0., 1., 1., 1., 1.]))
  assert m['fpr'] == 0.25 and m['fnr'] == 0.0 and m['accuracy'] == 0.875, m
  print('OK')
  "
  ```

  Must print `OK`. Full eval against real weights runs in Post-Completion.

### Task 8: Bench Go module scaffold + subcommand router

**Files:**

- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/bench/go.mod`
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/bench/main.go`

- [x] write `bench/go.mod` modeled on `/home/butvinm/Dev/orion/examples/c3ae-demo/server/go.mod`: module path `github.com/butvinm/orion/v2/examples/c3ae-demo/experiments/bench`, requires `github.com/butvinm/orion/v2` with `replace github.com/butvinm/orion/v2 => ../../../..` (point at repo root for in-tree development), and `github.com/tuneinsight/lattigo/v6 v6.2.0`.
- [x] write `bench/main.go` with `main()` dispatching on `os.Args[1]` to `cmdKeygen|cmdEncrypt|cmdInfer|cmdDecrypt`. Each handler uses `flag.NewFlagSet(name, flag.ExitOnError)`. Stub each handler with `panic("not implemented")` for now. Print a usage message on unknown/missing subcommand and exit nonzero.
- [x] **manual verify**:

  ```sh
  cd examples/c3ae-demo/experiments/bench
  go build ./...                # must succeed
  ./bench                       # must print usage and exit nonzero
  echo "exit: $?"               # must show nonzero
  ./bench unknown_cmd           # must print usage and exit nonzero
  ```

### Task 9: Bench `keygen` subcommand (no bootstrap)

**Files:**

- Modify: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/bench/main.go`
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/bench/keygen.go`

- [x] implement `cmdKeygen`. Flags: `--model <path>`, `--out <dir>`. Behavior:
  - load model with `evaluator.LoadModel`
  - extract params + manifest via `model.ClientParams()`
  - construct `ckks.Parameters` from the orion params (use the existing helper if `evaluator` exposes one; else inline `ckks.NewParametersFromLiteral` with the orion fields)
  - generate `sk` via `rlwe.NewKeyGenerator(params).GenSecretKeyNew()`
  - generate exactly the galois keys listed in `manifest.GaloisElements` and the relinearization key if `manifest.NeedsRLK`
  - **no bootstrap path** — both configs are no-bootstrap, so error out with a clear message if `manifest.BootstrapSlots` is non-empty (defensive; should never trigger)
  - serialize: `sk.MarshalBinary() → <out>/sk.bin`, build `MemEvaluationKeySet`, `evk.MarshalBinary() → <out>/evk.bin`
  - write `<out>/keygen.json` with `{"keygen_s": ..., "evk_bytes": ...}`
- [x] **manual verify** (deferred to Task 13 once we have a working `.orion` model end-to-end). For now, just verify build:

  ```sh
  cd examples/c3ae-demo/experiments/bench
  go build ./...                # must succeed
  ./bench keygen                # must print missing-flag error and exit nonzero
  ```

  Output: `go build ./...` exits 0; `./bench keygen` prints `bench keygen: --model is required` and exits with status 1.

### Task 10: Bench `encrypt` subcommand (SK-mode)

**Files:**

- Modify: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/bench/main.go`
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/bench/encrypt.go`

- [x] implement `cmdEncrypt`. Flags: `--model <path>`, `--sk <path>`, `--input <path>`, `--out <path>`. Behavior:
  - load model + params + manifest (same path as keygen)
  - read `sk`: `sk := &rlwe.SecretKey{}; sk.UnmarshalBinary(skBytes)`
  - read input file as `[]float64` of exactly 12288 values; pad with zeros to `params.MaxSlots()`
  - construct `encoder := ckks.NewEncoder(params)`, `encryptor := rlwe.NewEncryptor(params, sk)` (SK-mode)
  - encode at `manifest.InputLevel` with `params.DefaultScale()`; encrypt → write `ct.MarshalBinary()` to output
- [x] **manual verify**: build only (full E2E in Task 13):

  ```sh
  go build ./...                # must succeed
  ./bench encrypt               # must print missing-flag error
  ```

  Output: `go build ./...` exits 0; `go vet ./...` exits 0; `./bench encrypt` prints `bench encrypt: --model is required` and exits with status 1. Note: `input_level` is the third return value of `model.ClientParams()` — not a manifest field as the task description suggested (manifest only has `GaloisElements`, `BootstrapSlots`, `BootLogP`, `BtpLogN`, `NeedsRLK` — see `/home/butvinm/Dev/orion/keys.go`).

### Task 11: Bench `infer` subcommand (the measured one)

**Files:**

- Modify: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/bench/main.go`
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/bench/infer.go`
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/bench/rss.go`

- [x] implement `readVmHWM()` in `rss.go`: parses `/proc/self/status` for the `VmHWM:` line, returns kB as `int64`. Returns 0 with a one-shot warning if `/proc/self/status` is unavailable (non-Linux).
- [x] implement `cmdInfer`. Flags: `--model <path>`, `--evk <path>`, `--ct <path>`, `--out <path>`, `--metrics <path>` (jsonl, append mode), `--sample-idx <int>` (for metrics tagging). Behavior:
  - load model
  - unmarshal `evk` (`MemEvaluationKeySet.UnmarshalBinary`)
  - construct `evaluator.NewEvaluatorFromKeySet(params, evk, nil)` (no bootstrap keys; signature is `(ckks.Parameters, *rlwe.MemEvaluationKeySet, *bootstrapping.EvaluationKeys)` per `/home/butvinm/Dev/orion/evaluator/evaluator.go:33`)
  - read input ciphertext from `--ct`
  - capture `t0 := time.Now()`, run `result, err := eval.Forward(model, []*rlwe.Ciphertext{ct})`, capture `forward_s := time.Since(t0).Seconds()`
  - read `peak_rss_mb := readVmHWM() / 1024`
  - serialize `result[0].MarshalBinary() → <out>`
  - append JSONL line to `--metrics`: `{"sample_idx": N, "forward_s": ..., "peak_rss_mb": ..., "result_ct_bytes": ...}`. Open with `os.O_APPEND | os.O_CREATE | os.O_WRONLY`, write line + `\n`, **call `f.Sync()` before close** for crash resilience.
- [x] **manual verify** (full E2E deferred to Task 13). Build only:

  ```sh
  go build ./...                # must succeed
  go vet ./...                  # must succeed
  ./bench infer                 # must print missing-flag error and exit 1
  ```

  Output: `go build ./...` exits 0; `go vet ./...` exits 0; `./bench infer` prints `bench infer: --model is required` and exits with status 1. Note: the actual `NewEvaluatorFromKeySet` signature takes `(ckks.Parameters, *rlwe.MemEvaluationKeySet, *bootstrapping.EvaluationKeys)` — not `(model, evk)` as the task description suggested. We pass `nil` for bootstrap keys since both supported configs are no-bootstrap. `input_level` from `model.ClientParams()` is the third return value (consistent with Task 10's note).

### Task 12: Bench `decrypt` subcommand

**Files:**

- Modify: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/bench/main.go`
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/bench/decrypt.go`

- [x] implement `cmdDecrypt`. Flags: `--model <path>`, `--sk <path>`, `--ct <path>`. Behavior:
  - load model + params
  - read sk
  - construct decryptor (`rlwe.NewDecryptor(params, sk)`) and encoder
  - read result ciphertext, decrypt → plaintext, decode → `[]float64`
  - take `decoded[0]` as the binary-classifier logit, compute `prob := 1/(1+math.Exp(-logit))`
  - write JSON `{"logit": ..., "prob": ...}` to stdout
- [x] **manual verify**: build only:

  ```sh
  go build ./...                # must succeed
  ./bench decrypt               # must print missing-flag error
  ```

  Output: `go build ./...` exits 0; `go vet ./...` exits 0; `./bench decrypt` prints `bench decrypt: --model is required` and exits with status 1. `encoder.Decode(pt, values)` accepts `interface{}` per `/home/butvinm/go/pkg/mod/github.com/tuneinsight/lattigo/v6@v6.2.0/schemes/ckks/encoder.go:192` so passing `[]float64` is correct.

### Task 13: End-to-end pipeline smoke test (manual, against `logn15`)

**Files:** none (uses files produced by Tasks 1–12 + existing weights)

**Local-machine adaptation:** the local box has 38 GB RAM (~14 GB free) but `bench infer` peaks at ~103 GB RSS at logn=15 per `/home/butvinm/Dev/orion/examples/c3ae-demo/README.md`. We therefore (a) ran compile/keygen/encrypt locally, (b) skipped `bench infer` (deferred to VPS Post-Completion), and (c) substituted a `bench decrypt` round-trip on the **encrypted input ciphertext** as a sanity check that the encrypt/decrypt pair works. UTKFace dataset is not present locally; we use a synthetic input — 12288 random float64 values in `[-1, 1]` (seed=42) written via numpy to `out/inputs/sample_test.bin` (98304 bytes).

- [x] manual end-to-end run against the existing `logn15` config. **Prerequisites**: existing trained Quad weights `/home/butvinm/Dev/orion/examples/c3ae-demo/weights.pth` copied to `out/weights_fhe.pth`; synthetic input generated.
- [x] compile model with `logn15` — `compile.json`: `compile_s=121.75`, `compile_peak_rss_mb=3322.9`, `model_bytes=878078308`.
- [x] keygen produces `sk.bin` (5,243,056 B) + `evk.bin` (7,717,797,550 B = ~7.2 GB) + `keygen.json` (`keygen_s=30.13`, `evk_bytes=7717797550`, galois=183, rlk=true).
- [x] encrypt produces `ct.bin` (8,389,166 B) — input padded from 12288 to 16384 slots at `level=15`.
- [x] **`bench infer` skipped** — needs ~103 GB peak RSS at logn=15 vs 38 GB local RAM. Deferred to VPS Post-Completion (the existing demo numbers in `examples/c3ae-demo/README.md` already establish the logn=15 cost; this task only proves the pipeline plumbing is correct).
- [x] decrypt encrypt output round-trips to recover input — `bench decrypt out/logn15/ct_test.bin` decoded `decoded[0]=0.547912`, vs `sample[0]=0.547912`, diff `4.01e-11` (well under the 1e-3 tolerance).

  ```sh
  cd examples/c3ae-demo/experiments

  # one-time
  cp ../weights.pth out/weights_fhe.pth
  python -c "
  import numpy as np
  rng = np.random.default_rng(42)
  vals = rng.uniform(-1, 1, 12288).astype(np.float64)
  vals.tofile('out/inputs/sample_test.bin')
  "
  python -m models.compile --variant fhe --config logn15 \
      --weights out/weights_fhe.pth --output out/logn15/model.orion

  # bench
  cd bench && go build && cd ..
  ./bench/bench keygen --model out/logn15/model.orion --out out/logn15/keys/
  ./bench/bench encrypt \
      --model out/logn15/model.orion \
      --sk out/logn15/keys/sk.bin \
      --input out/inputs/sample_test.bin \
      --out out/logn15/ct_test.bin
  ./bench/bench decrypt \
      --model out/logn15/model.orion \
      --sk out/logn15/keys/sk.bin \
      --ct out/logn15/ct_test.bin
  ```

  Round-trip verification:

  ```sh
  python -c "
  import numpy as np, json
  sample = np.fromfile('out/inputs/sample_test.bin', dtype=np.float64)
  with open('out/logn15/roundtrip.json') as f:
      result = json.load(f)
  diff = abs(result['logit'] - sample[0])
  assert diff < 1e-3
  "
  ```

- [x] **`bench infer` E2E + cleartext-vs-FHE MAE check deferred to VPS Post-Completion** (see Section "Post-Completion → 2. Run FHE `logn15`"). The local smoke test only validates compile + keygen + encrypt + decrypt round-trip; full forward inference and the `|fhe_prob − clear_prob| < 0.05` check require a VPS run.

### Task 14: Orchestration scripts

**Files:**

- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/scripts/run_cleartext.sh`
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/scripts/run_fhe.sh`

- [x] write `run_cleartext.sh` (`#!/usr/bin/env bash`, `set -euo pipefail`):
  - assert venv is active (`python -c 'import sys; assert sys.prefix != sys.base_prefix' || { echo "activate venv"; exit 1; }`)
  - if `out/weights_relu.pth` missing: `python -m models.train --variant relu --data-dir ./data/UTKFace --epochs 60`
  - if `out/weights_fhe.pth` missing: `cp ../weights.pth out/weights_fhe.pth`
  - run `python -m models.eval` → `results/cleartext.csv`
- [x] write `run_fhe.sh` (`#!/usr/bin/env bash`, `set -euo pipefail`):
  - takes `$1 = config name` (one of `logn15`, `logn16`)
  - assert venv is active
  - `python -m models.compile --variant fhe --config "$CFG" --weights out/weights_fhe.pth --output "out/$CFG/model.orion"`
  - `python -m models.prep_input --boundary-band` (idempotent — only run if `out/inputs/ground_truth.csv` is missing or empty)
  - `(cd bench && go build)` if `bench/bench` is missing or stale
  - keygen once: `/usr/bin/time -v ./bench/bench keygen --model "out/$CFG/model.orion" --out "out/$CFG/keys/" 2> "results/$CFG/keygen_time.log"`
  - per sample-idx in `out/inputs/ground_truth.csv`: encrypt → `/usr/bin/time -v ./bench/bench infer ... 2>> results/$CFG/infer_time.log` → decrypt → append to `results/$CFG/run.jsonl`
  - script must be **idempotent**: skip a sample-idx if its line already exists in `run.jsonl` (grep for `"sample_idx": N`)
- [x] **manual verify**:

  ```sh
  bash -n scripts/run_cleartext.sh
  bash -n scripts/run_fhe.sh
  # both must return 0 (syntax OK)
  ```

  Output: both `bash -n` checks exit 0. Scripts marked executable (`chmod +x`). `shellcheck` not available locally (skipped per task instructions).

### Task 15: Add `build_results.py` aggregator

**Files:**

- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/build_results.py`

- [x] write `build_results.py` that:
  - reads `results/cleartext.csv` (if exists), `results/<cfg>/run.jsonl`, `results/<cfg>/keygen_time.log`, `out/<cfg>/keys/keygen.json`, and `out/<cfg>/compile.json` for each `<cfg>` directory under `results/`
  - emits `results/results.md` with two markdown tables:
    - **Cleartext quality**: rows = `variant × scope`, cols = `n, FPR, FNR, Accuracy`
    - **FHE cost**: rows = `config`, cols = `compile_s, keygen_s, evk_GB, mean_forward_s ± std, peak_rss_GB`
  - graceful degradation: if a config has only partial data, emit what's available; mark missing as `n/a`
- [x] **manual verify** with synthetic input:

  ```sh
  cd examples/c3ae-demo/experiments
  mkdir -p /tmp/c3ae_test/results/logn16 /tmp/c3ae_test/out/logn16
  echo 'variant,scope,n,fpr,fnr,accuracy
  relu,overall,3492,0.012,0.034,0.978
  relu,boundary,512,0.18,0.21,0.81' > /tmp/c3ae_test/results/cleartext.csv
  echo '{"sample_idx":0,"forward_s":300.0,"peak_rss_mb":50000,"result_ct_bytes":14000000}' > /tmp/c3ae_test/results/logn16/run.jsonl
  echo '{"keygen_s":80.0,"evk_bytes":7700000000}' > /tmp/c3ae_test/out/logn16/keys/keygen.json 2>/dev/null || mkdir -p /tmp/c3ae_test/out/logn16/keys && echo '{"keygen_s":80.0,"evk_bytes":7700000000}' > /tmp/c3ae_test/out/logn16/keys/keygen.json
  echo '{"compile_s":150.0,"compile_peak_rss_mb":3500,"model_bytes":900000000}' > /tmp/c3ae_test/out/logn16/compile.json
  python build_results.py --root /tmp/c3ae_test
  cat /tmp/c3ae_test/results/results.md
  ```

  Inspect: must show 2 tables with correct rows and cells; no traceback.

  Verified: scenario 1 (full synthetic data) produces `results.md` with the **Cleartext quality** table (4 rows: `fhe×{boundary,overall}`, `relu×{boundary,overall}`) and **FHE cost** table (1 row: `logn16` with `compile_s=150.0, keygen_s=80.0, evk_GB=7.17, mean_forward_s=305.0 ± 5.0, peak_rss_GB=49.32 ± 0.49`). Scenario 2 (empty `results/` only) produces `results.md` with `_No cleartext results yet..._` and `_No FHE results yet..._` notices — no traceback. Default `--root` resolves to `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments` via `--help`.

### Task 16: Verify acceptance criteria + flesh out experiments README

**Files:**

- Modify: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/README.md`

- [x] expand the README to cover:
  - prerequisites (venv, Go 1.22+, UTKFace data dir)
  - end-to-end commands: cleartext (`bash scripts/run_cleartext.sh`) and FHE (`bash scripts/run_fhe.sh logn15`, `bash scripts/run_fhe.sh logn16`)
  - bench binary build (`cd bench && go build`)
  - directory layout of generated `out/` and `results/` artifacts
  - explicit caveat: this experiment runs **without bootstrap** in either config. Bootstrap at the user's externally-fixed Q=415 budget at logn=15 was investigated and dropped due to security-bound violations.
- [x] run `npx prettier --write examples/c3ae-demo/experiments/README.md` (per global preference for markdown formatting)
- [x] **manual verify**:
  - `pytest python/tests/` — must pass (no regressions in core repo)
  - `go test ./evaluator/...` — must pass (no regressions)
  - `cd examples/c3ae-demo/experiments/bench && go vet ./... && go build ./...` — must pass
  - Tasks 1–15 all marked `[x]`

  Output: `pytest python/tests/` → 215 passed, 1 skipped, 1 warning in 16.46s. `go test ./evaluator/...` → ok in 29.430s. `go vet ./...` and `go build ./...` in `bench/` both exit 0. `grep -nE '^- \[ \]' docs/plans/2026-05-08-c3ae-experiments.md` matches only Task 16's own three lines (now closed by this edit) — Tasks 1–15 are all `[x]`. `npx prettier --write` reported the README unchanged (already well-formatted).

## Technical Details

### Data flow diagram

```
                 ┌────────────────────┐
                 │ UTKFace dataset    │
                 │ (~3.5k test imgs)  │
                 └──────┬─────────────┘
                        │ seed=42 split
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
   train.py         eval.py        prep_input.py
   --variant       (cleartext     (3 boundary
   {relu,fhe}       scopes)        samples → .bin)
        │               │               │
        ▼               ▼               ▼
  weights_*.pth   cleartext.csv   sample_N.bin
        │
        ▼
   compile.py ──► model.orion (per CKKS config)
                       │
                       ▼
            ┌─── bench keygen ────► sk.bin, evk.bin
            │
            ├─── bench encrypt (per sample) ─► ct.bin
            │
            ├─── bench infer  (measured) ─► result.bin + run.jsonl
            │
            └─── bench decrypt ────► stdout JSON
                                          │
                                          ▼
                                   build_results.py
                                          │
                                          ▼
                                    results.md
```

### Naming conventions

- Variants: `relu`, `fhe` (Quad)
- Config names: `logn15`, `logn16`
- Weight files: `out/weights_<variant>.pth`
- Compiled models: `out/<config>/model.orion`
- Keys: `out/<config>/keys/{sk,evk}.bin`
- Inputs: `out/inputs/sample_<idx>.bin`
- Results: `results/cleartext.csv`, `results/<config>/run.jsonl`, `results/results.md`

### Key parameter math

| Config   | LogN | LogQ             | sum | LogP     | sum | LogQP | Bound (128-bit dense) | Bootstrap |
| -------- | ---- | ---------------- | --- | -------- | --- | ----- | --------------------- | --------- |
| `logn15` | 15   | `[51] + [40]*15` | 651 | `[50]*4` | 200 | 851   | 881                   | no        |
| `logn16` | 16   | `[55] + [40]*15` | 655 | `[55]*6` | 330 | 985   | 1770                  | no        |

Bounds from HE Standard, 128-bit security, dense ternary secret. Both configs have headroom under the bound.

### Why SK-mode encryption?

Bench is a single-party local simulator — there is no client/server boundary inside the `bench` process. SK-mode (`rlwe.NewEncryptor(params, sk)`) avoids the need for a public key file, produces smaller ciphertexts, and is cryptographically equivalent for the FHE evaluation we measure. The `wasm-demo` uses PK-mode because it has a real client/server split; bench does not.

### Why Go-only inference?

Existing `run_fhe.py` uses two `.so` files (one for keygen via `lattigo`, one for inference via `orion_evaluator`), which can't share Go heap and forces a serialize → deserialize round-trip of the eval keys. That residual ~30 GB of Python-side overhead distorts RSS at logn=16. By moving keygen into the same Go binary as inference (each subcommand a fresh process, but the same binary), we eliminate this and get a clean RSS reading.

### Why no bootstrap?

An earlier draft included a `logn=15, Q=415` config with bootstrap. A bootstrap circuit at logn=15 with Q=415 expands to LogQP ≈ 1339 bits (415 residual + ~558 bootstrap inner Q + ~366 boot_LogP) — far above the 128-bit-dense bound of 881 and with no documented sparse-key bound at logn=15 high enough to cover it. The bootstrap variant was dropped. Both surviving configs are pure no-bootstrap with **identical multiplicative depth (15 levels)**, isolating the cost of doubling the ring degree from logn=15 to logn=16.

## Post-Completion

_Items requiring actual experiment runs on the VPS — informational, no checkboxes._

**1. Run cleartext experiment** (~30 min on VPS CPU)

- ssh into VPS, cd into checkout, activate venv
- `cd examples/c3ae-demo/experiments && bash scripts/run_cleartext.sh`
- Verify `results/cleartext.csv` has 4 rows (`relu × {overall, boundary}` + `fhe × {overall, boundary}`)
- Inspect: boundary-band FPR/FNR should be substantially worse than overall (5–10× expected)

**2. Run FHE `logn15`** (the existing demo config — already proven at ~139s/103 GB)

- `bash scripts/run_fhe.sh logn15`
- Verify `results/logn15/run.jsonl` has 3 lines (one per boundary sample)
- Sanity: `forward_s` should be in the ballpark of the demo's 139 s; `peak_rss_mb` should be substantially **lower** than 103 GB (the Python wrapper overhead is gone in Go-only inference)

**3. Run FHE `logn16`**

- `bash scripts/run_fhe.sh logn16`
- Watch for OOM. The 128 GB VPS may be tight; logn=16 ciphertexts are ~2× and key set ~2–3× larger than logn=15.
- If OOM: capture `dmesg`, document, and consider dropping to a smaller logn=16 Q budget or a larger machine.
- Verify `results/logn16/run.jsonl` has 3 lines.

**4. Build final report**

- `cd examples/c3ae-demo/experiments && python build_results.py`
- Manually review `results/results.md` for sanity: logn=16 should have larger eval keys than logn=15; forward time should be larger; peak RSS should be larger.
- Commit `results/results.md` (the only result file under git per `.gitignore`)
- Move this plan to `/home/butvinm/Dev/orion/docs/plans/completed/2026-05-08-c3ae-experiments.md`

**Manual verification scenarios:**

- Cleartext: run `eval.py` with a deliberately broken weight file — confirm graceful skip with warning, not a crash.
- FHE: confirm `decrypt_mae < 0.05` (`|sigmoid(fhe_logit) − sigmoid(cleartext_logit)|`) for both configs. Higher MAE indicates a pipeline bug.
