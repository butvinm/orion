# C3AE-demo + experiments consolidation

## Overview

Collapse the duplicated `/home/butvinm/Dev/orion/examples/c3ae-demo/` and `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/` trees into a single canonical layout. Eliminate file-pair drift risks (4 pairs of overlapping files; one pair is byte-identical). Update the demo README with the new measurements (Go-only inference reduced peak RSS by ~47% vs the Python-wrapped pipeline the README currently advertises) and add a benchmarking guide.

Net result: same code, half the surface area, accurate documentation.

## Context (from discovery)

**Files involved:**

- `/home/butvinm/Dev/orion/examples/c3ae-demo/model.py` — current Quad C3AE definition
- `/home/butvinm/Dev/orion/examples/c3ae-demo/train.py` — original training script
- `/home/butvinm/Dev/orion/examples/c3ae-demo/generate_model.py` — original compile script (one config hardcoded)
- `/home/butvinm/Dev/orion/examples/c3ae-demo/run_fhe.py` — legacy Python end-to-end FHE pipeline (superseded by Go bench)
- `/home/butvinm/Dev/orion/examples/c3ae-demo/README.md` — current demo docs (out of date measurements)
- `/home/butvinm/Dev/orion/examples/c3ae-demo/.gitignore`
- `/home/butvinm/Dev/orion/examples/c3ae-demo/requirements.txt` — vestigial pip-install file (4 lines: 3 orion packages + Pillow). Already incomplete (missing `kagglehub` which the post-consolidation demo uses). Going away — superseded by the workspace `uv sync` since the demo requires the repo checkout anyway (the model defs, bench binary, scripts, server, client all live in the repo, not on PyPI).
- `/home/butvinm/Dev/orion/pyproject.toml` — workspace pyproject. Will gain `kagglehub` as a dependency (currently pip-installed ad-hoc by `setup-fhe.sh`).
- `/home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-fhe.sh` — currently does `pip install kagglehub`. Will be patched to drop that line once kagglehub is in the workspace.
- `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/` (whole subtree — to be flattened up one level):
  - `models/{c3ae,c3ae_fhe,params,train,compile,prep_input,eval,utkface,metrics}.py`
  - `bench/` (Go single-binary)
  - `scripts/{run_cleartext,run_fhe}.sh`
  - `verify_fhe.py`, `build_results.py`
  - `results/` (gitignored; some files allow-listed)
  - `out/` (gitignored)
- `/home/butvinm/Dev/orion/examples/c3ae-demo/server/main.go` — Go HTTP server (browser demo); takes paths via argv → path-agnostic
- `/home/butvinm/Dev/orion/examples/c3ae-demo/client/` — WASM client (browser demo)
- `/home/butvinm/Dev/orion/CLAUDE.md` — project docs; references `examples/c3ae-demo/experiments/` paths

**Patterns observed:**

- `models/c3ae_fhe.py` is a verbatim byte-identical copy of `model.py` — silent-divergence risk.
- `experiments/models/train.py` is a strict superset of original `train.py` (adds `--variant {relu,fhe}`).
- `experiments/models/compile.py` is a parameterized superset of `generate_model.py` (adds `--config <name>`).
- `UTKFaceDataset` exists in 3 places: original train.py, experiments/models/utkface.py, and indirectly via prep_input.py imports.
- The Go bench binary at `experiments/bench/bench` measured ~54 GB peak RSS at logn=15; the README still cites the legacy Python wrapper's 103 GB number.
- `experiments/.gitignore` allowlists `results/results.md` and `results/cleartext.csv`; the original `c3ae-demo/.gitignore` only excludes `weights.pth` and `model.orion` at the root.

**Dependencies:**

- `bench/go.mod` declares `replace github.com/butvinm/orion/v2 => ../../../..` (3 levels up). After move, the bench dir is at `examples/c3ae-demo/bench/` instead of `examples/c3ae-demo/experiments/bench/` — the replace path becomes `../../..` (one fewer level).
- `server/go.mod` lives at `examples/c3ae-demo/server/go.mod` and is unchanged.
- `bench/main.go` and friends do not hardcode any paths; they take everything via argv.
- `scripts/run_cleartext.sh` and `scripts/run_fhe.sh` use script-relative `cd` (`SCRIPT_DIR/..`) — they re-resolve relative to the new location automatically; no path edits needed.
- `verify_fhe.py` and `build_results.py` use script-relative path resolution — same.
- `run_fhe.sh` invokes `cp ../weights.pth out/weights_fhe.pth` as a fallback. After the move, `../weights.pth` is `examples/c3ae-demo/../weights.pth` which doesn't exist (and `examples/c3ae-demo/weights.pth` is being deleted). Need to remove the fallback (or replace with a clearer error) since training is now the only way to get fhe weights from a fresh checkout.

## Development Approach

- **Testing approach: NO automated test files.** Established project rule for the c3ae-demo work area. Each task ends with manual-verify commands the agent runs. No `_test.go`, no `test_*.py` files added by this plan.
- **Override of the standard plan template's "every task MUST include tests" rule** — that rule does not apply here. The project-level decision (documented in `/home/butvinm/Dev/orion/docs/plans/completed/2026-05-08-c3ae-experiments.md`) is that c3ae-demo + bench code is research/demo code with manual-verify gates instead of unit tests.
- **Existing repo tests must continue to pass**: `pytest python/tests/` and `go test ./evaluator/...`. These cover the imported library code we are NOT touching, so they should stay green automatically.
- Atomic commits, specific staging only (`git mv` per file, `git rm` per file, never `git add .`).
- Use the staging-and-commit script: `bash /home/butvinm/.claude/plugins/cache/umputun-cc-thingz/planning/3.6.0/skills/exec/scripts/stage-and-commit.sh "<message>" <files…>`
- All file references use absolute `/full/path` format.
- Make small, focused commits — one per logical migration step.

## Progress Tracking

- Mark completed items `[x]` immediately when done.
- Add discovered tasks with `➕` prefix.
- Document blockers with `⚠️` prefix.

## Solution Overview

Three sequential atomic commits:

1. **Move** the experiments tree up one level using `git mv` so history is preserved. Adjust `bench/go.mod`'s `replace` directive (one fewer `../`). No content changes.
2. **Delete** the obsolete originals (`model.py`, `train.py`, `generate_model.py`, `run_fhe.py`). Patch `scripts/run_fhe.sh` to drop the `cp ../weights.pth` fallback now that the source file no longer exists.
3. **Rewrite** the README with the new layout, updated commands, current measurements (Go bench numbers), and a new benchmarking guide section. Update `CLAUDE.md` to point at the new paths instead of `experiments/`.

The bench binary, server, client, and Python module structure stay the same — we're only re-rooting paths.

## Technical Details

### bench/go.mod replace path

Current at `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/bench/go.mod`:

```
replace github.com/butvinm/orion/v2 => ../../../..
```

Path arithmetic: `experiments/bench` → `experiments` → `c3ae-demo` → `examples` → repo root. That's 4 levels up.

After move to `/home/butvinm/Dev/orion/examples/c3ae-demo/bench/go.mod`:

```
replace github.com/butvinm/orion/v2 => ../../..
```

Path arithmetic: `bench` → `c3ae-demo` → `examples` → repo root. 3 levels up.

### .gitignore consolidation

Current `/home/butvinm/Dev/orion/examples/c3ae-demo/.gitignore`:

```
weights.pth
model.orion
```

Current `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/.gitignore`:

```
out/
bench/bench
__pycache__/
*.pyc
results/*
!results/.gitkeep
!results/results.md
!results/cleartext.csv
```

Merged at `/home/butvinm/Dev/orion/examples/c3ae-demo/.gitignore`:

```
out/
bench/bench
__pycache__/
*.pyc
results/*
!results/.gitkeep
!results/results.md
!results/cleartext.csv

# Legacy paths from the pre-consolidation layout (kept until any
# user-local files are migrated):
weights.pth
model.orion
```

The legacy `weights.pth` and `model.orion` lines are kept (one commit) so existing local checkouts don't accidentally start tracking those files mid-migration. Can be deleted in a follow-up after enough time has passed.

### kagglehub: workspace dependency, drop ad-hoc pip install

Currently `kagglehub` is **not** listed in any pyproject.toml in the repo (verified via grep). The setup-fhe.sh provisioning script does `pip install kagglehub` as a separate step after `uv sync`. This is fragile — anyone trying to run the demo locally without the provisioning script hits an `ImportError` from `models/prep_input.py`.

Add `kagglehub` to `/home/butvinm/Dev/orion/pyproject.toml` `[tool.uv]` `dev-dependencies` (alongside `pytest`, `ruff`, `mypy`, `torchvision`). Rationale: `kagglehub` is needed by demo-running scripts (`prep_input.py`, training data download), not by anyone using Orion as a library, so it's a dev-/example-time dep rather than a core dep.

Edit:

```toml
[tool.uv]
dev-dependencies = [
    "pytest>=7.0",
    "ruff>=0.4",
    "mypy>=1.0",
    "torchvision>=0.17.0",
    "kagglehub>=0.3.0",
]
```

Then patch `/home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-fhe.sh` (and `setup-train.sh` for symmetry) to drop the `pip install kagglehub` line — `uv sync` will now handle it.

After this, `requirements.txt` is fully redundant: every dependency the demo needs is in the workspace. Anyone who wants to run the demo clones the repo and runs `uv sync`. Anyone who wants Orion as a library does `pip install orion-v2-{lattigo,compiler,evaluator}` (already documented in the demo README's "PyPI install" section, unchanged by this plan).

### scripts/run_fhe.sh patch (Task 2)

The script currently has (around line 60-70 after the move):

```bash
if [ ! -f out/weights_fhe.pth ]; then
    if [ -f ../weights.pth ]; then
        echo "[run_cleartext] copying ../weights.pth -> out/weights_fhe.pth"
        cp ../weights.pth out/weights_fhe.pth
    else
        echo "[run_cleartext] training FHE (Quad) variant (60 epochs)..."
        python -m models.train --variant fhe --data-dir ./data/UTKFace --epochs 60
    fi
fi
```

Wait — that's `run_cleartext.sh`, not `run_fhe.sh`. Re-check both during implementation. The `cp ../weights.pth` fallback is in `run_cleartext.sh`. After the move:

- New cwd is `examples/c3ae-demo/`, so `../weights.pth` resolves to `examples/weights.pth` — which doesn't exist now and never did.
- The pre-move semantics relied on `examples/c3ae-demo/weights.pth` (which was the original demo's checked-in-but-gitignored weights file). After the move, that file is gone too.

Patch: drop the `cp ../weights.pth` fallback entirely. Always train if `out/weights_fhe.pth` is missing. The fallback was a hack to avoid retraining when the demo's checked-in weights were available; now that the checked-in weights file is also gone, the fallback is dead code.

### Updated `bench infer` paths

`bench infer` is invoked from `scripts/run_fhe.sh` with `out/<cfg>/...` paths. These are script-relative and unchanged by the move (script is in the same place relative to its `out/` sibling).

### server quick-start command change

Current `examples/c3ae-demo/README.md`:

```
go run . ../model.orion ../client :8080
```

Source at `/home/butvinm/Dev/orion/examples/c3ae-demo/server/main.go` takes the model path as argv[1]. After consolidation, the canonical model path becomes:

```
go run . ../out/logn15/model.orion ../client :8080
```

(The `out/` directory is gitignored; users compile via `python -m models.compile --config logn15` first.)

### README replacement: Measurements section

Current numbers (line ~127 of `/home/butvinm/Dev/orion/examples/c3ae-demo/README.md`):

| Metric          | Value           |
| --------------- | --------------- |
| Inference time  | 139s per sample |
| Peak server RSS | 103 GB          |

These are from the Python-wrapper-era `run_fhe.py` benchmark on the same VPS (`cpu.16.128.240`). The Go bench measurements taken on 2026-05-10 are 47% lower on RSS:

New table sourced from `/home/butvinm/Dev/orion/examples/c3ae-demo/results/results.md` (post-move):

**Cleartext quality (UTKFace test split, n=3557 / boundary 16-20 n=161):**

| variant | scope    | n    | FPR    | FNR    | Accuracy |
| ------- | -------- | ---- | ------ | ------ | -------- |
| relu    | overall  | 3557 | 0.1675 | 0.0190 | 0.9556   |
| relu    | boundary | 161  | 0.6515 | 0.1474 | 0.6460   |
| fhe     | overall  | 3557 | 0.2085 | 0.0268 | 0.9421   |
| fhe     | boundary | 161  | 0.7121 | 0.1579 | 0.6149   |

**FHE inference cost (cpu.16.128.240, Go-only bench, 3 boundary samples):**

| config | compile_s | compile_peak_GB | keygen_s | evk_GB | mean_forward_s | peak_rss_GB |
| ------ | --------- | --------------- | -------- | ------ | -------------- | ----------- |
| logn15 | 160.2     | 12.87           | 44.1     | 7.19   | 157.0 ± 2.9    | 54.19       |
| logn16 | 386.7     | 25.77           | 68.1     | 12.70  | 543.7 ± 219.4  | 114.37      |

Compared to the pre-Go-bench measurements (Python wrapper at logn=15): forward time ~+13% (155s vs 139s), peak RSS **−47% (54 GB vs 103 GB)**. The peak RSS reduction is the headline result — confirms the Python wrapper added ~50 GB of overhead at logn=15.

Add a one-liner pointer to the full table at `examples/c3ae-demo/results/results.md`.

### Benchmarking guide section content

Lives in README between "Architecture" and "Model" sections. Outline:

````markdown
## Benchmarking guide

Reproduce the cleartext quality + FHE inference cost measurements
yourself.

### Prerequisites

- Project venv (`uv sync` from repo root) with `kagglehub` installed
- Go 1.24+ (note: above the demo's stated 1.22+ minimum, due to the
  bench module's go directive)
- UTKFace dataset (downloaded via kagglehub, see Quick Start)
- For FHE: at least 64 GB RAM for logn=15, 128 GB for logn=16
  (smaller boxes will OOM mid-inference)

### Cleartext

```sh
bash scripts/run_cleartext.sh
```
````

Trains the ReLU variant if missing (~30 min on CPU, ~3 min on GPU),
trains the Quad variant if missing (same), then evaluates both on the
test split. Writes `results/cleartext.csv`.

### FHE inference

```sh
# Build the bench binary once
cd bench && go build && cd ..

# Run for a config
bash scripts/run_fhe.sh logn15  # or logn16
```

Compiles the model, generates keys, runs encrypt + infer + decrypt for
the 3 boundary samples (idx 12, 35, 44). Streams metrics to
`results/<cfg>/run.jsonl` (one line per sample) so a crashed/OOM-killed
run still preserves what completed. Idempotent — re-runs skip
already-completed samples.

### Verify FHE correctness

```sh
python verify_fhe.py --config logn15
```

Compares each FHE-decrypted probability against the cleartext PyTorch
forward of the same input. Writes `results/<cfg>/cleartext_vs_fhe.csv`
and exits non-zero if any sample exceeds `--tol` (default 0.05).

### Aggregate the report

```sh
python build_results.py
```

Reads `results/cleartext.csv` and `results/<cfg>/run.jsonl` files,
emits `results/results.md` with two markdown tables.

### Provisioning a fresh VPS for benchmarking

We provisioned three VPSes on immers.cloud during the 2026-05-09 run.
The provisioning script at
`/home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-fhe.sh`
captures the apt deps + Go 1.24 + uv + UTKFace download in one shot;
total provisioning takes ~5 min on a fresh `cpu.16.128.240`. Use it as
a reference rather than copy-pasting commands. The plan at
`/home/butvinm/Dev/orion/docs/plans/completed/2026-05-09-c3ae-vps-runs.md`
documents the full sequence including measured timings and cost.

````

## What Goes Where

- **Implementation Steps** — code/file/doc changes in this repo (the 4 commits).
- **Post-Completion** — push + verify + plan-move (after all 4 commits land cleanly).

## Implementation Steps

### Task 1: Move experiments tree into c3ae-demo via git mv

**Files:**

- Move (via `git mv`):
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/__init__.py` → `/home/butvinm/Dev/orion/examples/c3ae-demo/models/__init__.py`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/c3ae.py` → `/home/butvinm/Dev/orion/examples/c3ae-demo/models/c3ae.py`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/c3ae_fhe.py` → `/home/butvinm/Dev/orion/examples/c3ae-demo/models/c3ae_fhe.py`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/params.py` → `/home/butvinm/Dev/orion/examples/c3ae-demo/models/params.py`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/utkface.py` → `/home/butvinm/Dev/orion/examples/c3ae-demo/models/utkface.py`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/metrics.py` → `/home/butvinm/Dev/orion/examples/c3ae-demo/models/metrics.py`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/train.py` → `/home/butvinm/Dev/orion/examples/c3ae-demo/models/train.py`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/compile.py` → `/home/butvinm/Dev/orion/examples/c3ae-demo/models/compile.py`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/prep_input.py` → `/home/butvinm/Dev/orion/examples/c3ae-demo/models/prep_input.py`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/models/eval.py` → `/home/butvinm/Dev/orion/examples/c3ae-demo/models/eval.py`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/bench/` (whole dir) → `/home/butvinm/Dev/orion/examples/c3ae-demo/bench/`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/scripts/` (whole dir) → `/home/butvinm/Dev/orion/examples/c3ae-demo/scripts/`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/results/` (whole dir, including `.gitkeep` and `results.md` and `cleartext.csv`) → `/home/butvinm/Dev/orion/examples/c3ae-demo/results/`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/verify_fhe.py` → `/home/butvinm/Dev/orion/examples/c3ae-demo/verify_fhe.py`
  - `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/build_results.py` → `/home/butvinm/Dev/orion/examples/c3ae-demo/build_results.py`
- Modify: `/home/butvinm/Dev/orion/examples/c3ae-demo/bench/go.mod` (the `replace` line: `../../../..` → `../../..`)
- Modify: `/home/butvinm/Dev/orion/examples/c3ae-demo/.gitignore` (merge in the experiments/.gitignore rules; keep legacy `weights.pth`/`model.orion` lines for now)
- Delete: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/.gitignore` (consolidated into parent)
- Delete: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/README.md` (the experiments-internal one — its content was redundant with the c3ae-demo README anyway; the new c3ae-demo README absorbs everything in Task 3)

Steps:

- [x] verify branch state — should be 3 commits ahead of `origin/experiments` (the revert commits) with a clean working tree.
- [x] `git mv` each models/* file individually (NOT a single bulk `mv`, so per-file history stays clean).
- [x] `git mv` the three subdirs (bench/, scripts/, results/) — these can be moved as directories since the per-file history follows.
- [x] `git mv` verify_fhe.py and build_results.py.
- [x] edit `/home/butvinm/Dev/orion/examples/c3ae-demo/bench/go.mod` to fix the `replace` path arithmetic (`../../../..` → `../../..`).
- [x] merge the gitignores: read both, write the consolidated content to `/home/butvinm/Dev/orion/examples/c3ae-demo/.gitignore` per the "Technical Details / .gitignore consolidation" section above, then `git rm /home/butvinm/Dev/orion/examples/c3ae-demo/experiments/.gitignore`.
- [x] `git rm /home/butvinm/Dev/orion/examples/c3ae-demo/experiments/README.md`.
- [x] verify `examples/c3ae-demo/experiments/` directory is now empty: `ls -la /home/butvinm/Dev/orion/examples/c3ae-demo/experiments/` should show nothing tracked. Use `find /home/butvinm/Dev/orion/examples/c3ae-demo/experiments -type f` to confirm.
- [x] `rmdir /home/butvinm/Dev/orion/examples/c3ae-demo/experiments` (untracked, just removes the empty directory). Migrated user-local untracked artifacts (`out/`, `profiles/`) to the new layout first, then removed the now-empty stale `__pycache__`/`.mypy_cache` and rmdir'd the parent.
- [x] **manual verify** (build/import sanity post-move):

  ```sh
  cd /home/butvinm/Dev/orion
  go vet ./evaluator/... && go test ./evaluator/...     # imported library still healthy
  cd /home/butvinm/Dev/orion/examples/c3ae-demo/bench
  go vet ./... && go build ./...                         # bench builds at new location
  ./bench                                                 # prints usage and exits nonzero
  cd /home/butvinm/Dev/orion/examples/c3ae-demo
  source /home/butvinm/Dev/orion/.venv/bin/activate
  python -m models.train --help | head -3                # imports resolve from new path
  python -m models.compile --help | head -3
  python -m models.eval --help | head -3 2>&1 | head -3 || true
  bash -n scripts/run_cleartext.sh && bash -n scripts/run_fhe.sh
````

All of these must succeed. The Python imports (`from models.utkface import ...` etc.) must resolve from the new `examples/c3ae-demo/models/` location.

- [x] commit:
  ```sh
  bash /home/butvinm/.claude/plugins/cache/umputun-cc-thingz/planning/3.6.0/skills/exec/scripts/stage-and-commit.sh \
      "refactor: move c3ae-demo/experiments tree up one level (no content changes)" \
      examples/c3ae-demo/models examples/c3ae-demo/bench examples/c3ae-demo/scripts \
      examples/c3ae-demo/results examples/c3ae-demo/verify_fhe.py \
      examples/c3ae-demo/build_results.py examples/c3ae-demo/.gitignore \
      examples/c3ae-demo/experiments
  ```
  (The script passes file paths to `git add` — for renames, both the deleted-source and added-target paths get included automatically by mentioning the directories.)

### Task 2: Delete obsolete originals + retire requirements.txt + add kagglehub to workspace

**Files:**

- Delete: `/home/butvinm/Dev/orion/examples/c3ae-demo/model.py`
- Delete: `/home/butvinm/Dev/orion/examples/c3ae-demo/train.py`
- Delete: `/home/butvinm/Dev/orion/examples/c3ae-demo/generate_model.py`
- Delete: `/home/butvinm/Dev/orion/examples/c3ae-demo/run_fhe.py`
- Delete: `/home/butvinm/Dev/orion/examples/c3ae-demo/requirements.txt` (vestigial; superseded by workspace `uv sync`)
- Modify: `/home/butvinm/Dev/orion/pyproject.toml` (add `kagglehub>=0.3.0` to `[tool.uv]` `dev-dependencies`)
- Modify: `/home/butvinm/Dev/orion/uv.lock` (regenerated by `uv sync`)
- Modify: `/home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-fhe.sh` (drop the `pip install kagglehub` line)
- Modify: `/home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-train.sh` (same — drop `pip install kagglehub` for symmetry)
- Modify: `/home/butvinm/Dev/orion/examples/c3ae-demo/scripts/run_cleartext.sh` (drop the `cp ../weights.pth` fallback — see Technical Details)
- Modify: `/home/butvinm/Dev/orion/examples/c3ae-demo/.gitignore` (drop the legacy `weights.pth` and `model.orion` lines now that those files are gone — optional; can defer to a later commit if want a grace period)

Steps:

- [x] verify the four originals are referenced ONLY by README content + the gitignore lines being dropped. Use `grep -rn 'model\.py\|generate_model\|run_fhe\.py\|c3ae-demo/train\.py' /home/butvinm/Dev/orion/ --include='*.go' --include='*.py' --include='*.sh' --include='*.md' 2>/dev/null` and verify that any matches are either (a) in `docs/plans/completed/...` (historical, leave alone) or (b) in the README (will be rewritten in Task 3).
- [x] verify `requirements.txt` is referenced ONLY by README/CLAUDE.md (also fine to leave — they get updated in Task 3/4): `grep -rn 'c3ae-demo/requirements\.txt\|requirements\.txt' /home/butvinm/Dev/orion/ --include='*.go' --include='*.py' --include='*.sh' --include='*.md' 2>/dev/null | grep -v 'docs/plans/completed/'`. The remaining matches MUST be only README/CLAUDE.md text — no script imports or build-system references.
- [x] `git rm` the five obsolete files (4 originals + requirements.txt).
- [x] edit `/home/butvinm/Dev/orion/pyproject.toml`: add `"kagglehub>=0.3.0"` to the `[tool.uv]` `dev-dependencies` list (alongside `pytest`, `ruff`, `mypy`, `torchvision`).
- [x] from repo root, run `uv sync` to regenerate `uv.lock`. The `.lock` file should pick up kagglehub + its transitive deps.
- [x] verify import works: `cd /home/butvinm/Dev/orion && source .venv/bin/activate && python -c "import kagglehub; print(kagglehub.__version__)"`.
- [x] edit `/home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-fhe.sh`: remove the `pip install kagglehub` line (keep everything else — `uv sync` will install it as part of the workspace setup the script already runs).
- [x] edit `/home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-train.sh`: same edit.
- [x] edit `/home/butvinm/Dev/orion/examples/c3ae-demo/scripts/run_cleartext.sh`: in the `if [ ! -f out/weights_fhe.pth ]` block, drop the `if [ -f ../weights.pth ]; then cp ...` branch and just always train. Keep the existing always-train branch as the only path. Verify with `bash -n` after.
- [x] update `/home/butvinm/Dev/orion/examples/c3ae-demo/.gitignore`: remove the `weights.pth` and `model.orion` lines (the files no longer exist; the `out/` rule covers any new artifacts). Optional — can leave for grace period.
- [x] **manual verify**:

  ```sh
  cd /home/butvinm/Dev/orion
  ls /home/butvinm/Dev/orion/examples/c3ae-demo/model.py 2>&1 | grep -E 'No such|cannot access'  # should fail
  ls /home/butvinm/Dev/orion/examples/c3ae-demo/run_fhe.py 2>&1 | grep -E 'No such|cannot access'
  ls /home/butvinm/Dev/orion/examples/c3ae-demo/requirements.txt 2>&1 | grep -E 'No such|cannot access'
  pytest python/tests/                                       # 215 pass
  go test ./evaluator/...                                     # ok
  cd /home/butvinm/Dev/orion/examples/c3ae-demo/bench
  go vet ./... && go build ./...                              # bench unaffected
  cd /home/butvinm/Dev/orion/examples/c3ae-demo
  bash -n scripts/run_cleartext.sh                            # syntax check after edit
  bash -n scripts/run_fhe.sh
  source /home/butvinm/Dev/orion/.venv/bin/activate
  python -c "import kagglehub; print('kagglehub ok')"          # confirm workspace installed it
  python -m models.train --help | head -3
  python -m models.compile --help | head -3
  python -m models.prep_input --help | head -3                 # confirm kagglehub usage path imports
  bash -n /home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-fhe.sh
  bash -n /home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-train.sh
  ```

- [x] commit. Stage explicitly — never `git add .`:
  ```sh
  bash /home/butvinm/.claude/plugins/cache/umputun-cc-thingz/planning/3.6.0/skills/exec/scripts/stage-and-commit.sh \
      "refactor: delete obsolete c3ae-demo originals + requirements.txt; add kagglehub to workspace" \
      examples/c3ae-demo/model.py examples/c3ae-demo/train.py \
      examples/c3ae-demo/generate_model.py examples/c3ae-demo/run_fhe.py \
      examples/c3ae-demo/requirements.txt \
      examples/c3ae-demo/scripts/run_cleartext.sh \
      examples/c3ae-demo/.gitignore \
      pyproject.toml uv.lock \
      docs/plans/2026-05-09-c3ae-vps-runs/setup-fhe.sh \
      docs/plans/2026-05-09-c3ae-vps-runs/setup-train.sh
  ```

### Task 3: Rewrite README with new layout, measurements, and benchmarking guide

**Files:**

- Modify: `/home/butvinm/Dev/orion/examples/c3ae-demo/README.md`

Steps:

- [x] read the current README to extract reusable text (architecture diagram, the model description block, the CKKS parameters table, etc.).
- [x] rewrite section by section. Order:
  1. Title + one-paragraph "what is this" intro (unchanged in spirit)
  2. Prerequisites:
     - Go 1.24+ (NOT 1.22+ — bench/go.mod requires 1.24)
     - Python 3.11+
     - **Python deps via `uv sync` from repo root** (the workspace handles all `orion-v2-*` packages + `kagglehub` + `torchvision` etc.). The old `pip install -r requirements.txt` path is GONE.
     - Node.js 18+ for the WASM client
     - UTKFace dataset (`python -c "import kagglehub; kagglehub.dataset_download('jangedoo/utkface-new')"` once `uv sync` has run)
     - For FHE benchmarks: at least 64 GB RAM (logn=15) or 128 GB (logn=16)
  3. Quick Start — browser demo path:
     - From repo root: `uv sync` (gets all Python deps including kagglehub)
     - `python tools/build_lattigo.py` (unchanged — CGO shared lib)
     - `cd examples/c3ae-demo`
     - `python -m models.train --variant fhe --data-dir ./data/UTKFace --epochs 60` (replaces `python train.py`)
     - `python -m models.compile --variant fhe --config logn15 --weights out/weights_fhe.pth --output out/logn15/model.orion` (replaces `python generate_model.py`)
     - `python tools/build_lattigo_wasm.py` from repo root (unchanged)
     - `cd client && npm install && npm run build` (unchanged)
     - `cd ../server && go run . ../out/logn15/model.orion ../client :8080` (path moved from `../model.orion`)
  4. **PyPI install (alternative for using Orion as a library)** — clarified: `pip install orion-v2-lattigo orion-v2-compiler orion-v2-evaluator` — this gives you the libraries to use in your own code, but does NOT provide the c3ae-demo files (model defs, bench, scripts, server, client) which live in this repo. Anyone running this demo must clone the repo and `uv sync`.
  5. **Benchmarking guide** — new section, content per "Technical Details / Benchmarking guide section content" above
  6. Architecture (browser ↔ server diagram — unchanged)
  7. Model (table — unchanged)
  8. CKKS Parameters (refer to `models/params.py`)
  9. **Measurements** — replaced with the new tables per "Technical Details / README replacement: Measurements section" above. Include the headline 47% RSS reduction prominently.
  10. Pointer at the bottom: "For the full experiment audit trail, see `/home/butvinm/Dev/orion/docs/plans/completed/2026-05-09-c3ae-vps-runs.md` and `examples/c3ae-demo/results/results.md`."
- [x] format with `npx prettier --write /home/butvinm/Dev/orion/examples/c3ae-demo/README.md` (per the user's global preference for markdown).
- [x] **manual verify**:

  ```sh
  # No broken links to deleted files
  grep -E 'model\.py|generate_model|run_fhe\.py|train\.py' /home/butvinm/Dev/orion/examples/c3ae-demo/README.md \
      | grep -v 'models/' || echo "no stale refs (expected)"

  # No leftover requirements.txt references
  grep -nE 'requirements\.txt' /home/butvinm/Dev/orion/examples/c3ae-demo/README.md \
      && echo "FAIL: requirements.txt still mentioned" || echo "no requirements.txt refs (expected)"

  # uv sync is documented as the canonical Python install path
  grep -nE 'uv sync' /home/butvinm/Dev/orion/examples/c3ae-demo/README.md \
      || echo "FAIL: README must mention uv sync"

  # All quick-start commands present
  grep -E 'python -m models\.(train|compile|eval|prep_input)|bash scripts/(run_cleartext|run_fhe)|cd bench|verify_fhe|build_results' \
      /home/butvinm/Dev/orion/examples/c3ae-demo/README.md | wc -l
  # Should be at least 8 matches

  # Measurements: confirm new numbers present
  grep -E '54\.\d+ GB|47%|cpu\.16\.128\.240|114\.\d+ GB' /home/butvinm/Dev/orion/examples/c3ae-demo/README.md
  ```

- [x] commit:
  ```sh
  bash /home/butvinm/.claude/plugins/cache/umputun-cc-thingz/planning/3.6.0/skills/exec/scripts/stage-and-commit.sh \
      "docs: rewrite c3ae-demo README for consolidated layout + Go-bench measurements + benchmarking guide" \
      examples/c3ae-demo/README.md
  ```

### Task 4: Update CLAUDE.md to reflect new layout (if needed)

**Files:**

- Modify (if stale references found): `/home/butvinm/Dev/orion/CLAUDE.md`

Steps:

- [x] grep for stale references:
  ```sh
  grep -nE 'c3ae-demo/experiments|c3ae-demo/(model|train|generate_model|run_fhe)\.py|c3ae-demo/requirements\.txt|examples/c3ae-demo/experiments' /home/butvinm/Dev/orion/CLAUDE.md
  ```
- [x] if no matches, this task is a no-op — skip the commit and mark this task `[x]` with note "no stale references found". (Three matches found at lines 13, 188, 196 — proceeded to update.)
- [x] if matches exist, update them in-place to point at the new layout (`examples/c3ae-demo/models/`, `examples/c3ae-demo/bench/`, etc.). Keep the existing "self-contained experiments harness" sentence but update the path it cites.
- [x] **manual verify**:
  ```sh
  grep -nE 'c3ae-demo/experiments|c3ae-demo/(model|train|generate_model|run_fhe)\.py|c3ae-demo/requirements\.txt' /home/butvinm/Dev/orion/CLAUDE.md \
      || echo "no stale refs remaining"
  ```
- [x] commit (only if changes made):
  ```sh
  bash /home/butvinm/.claude/plugins/cache/umputun-cc-thingz/planning/3.6.0/skills/exec/scripts/stage-and-commit.sh \
      "docs: update CLAUDE.md to reference consolidated c3ae-demo layout" \
      CLAUDE.md
  ```

### Task 5: Verify acceptance criteria

- [x] verify all requirements from Overview are implemented: experiments/ subdir gone; 4 obsolete files gone; requirements.txt gone; kagglehub installable via `uv sync`; README updated with new measurements + benchmarking guide + uv-only Python install story; CLAUDE.md consistent.
- [x] full repo test sweep:
  ```sh
  cd /home/butvinm/Dev/orion
  source .venv/bin/activate
  pytest python/tests/                            # 215 pass, 1 skip
  go test ./evaluator/...                          # ok
  go vet ./examples/c3ae-demo/bench/...            # clean (post-move)
  cd examples/c3ae-demo/bench && go build ./...    # builds
  cd /home/butvinm/Dev/orion/examples/c3ae-demo
  bash -n scripts/run_cleartext.sh
  bash -n scripts/run_fhe.sh
  python -c "import kagglehub; print('kagglehub from uv workspace ok')"
  python -m models.train --help                    # works
  python -m models.compile --help                  # works
  python -m models.eval --help 2>&1 | head -2 || true
  python -m models.prep_input --help 2>&1 | head -2 || true
  python verify_fhe.py --help 2>&1 | head -2 || true
  python build_results.py --help 2>&1 | head -2 || true
  ```
- [x] verify the deleted files really are gone and not just renamed:
  ```sh
  for f in model.py train.py generate_model.py run_fhe.py requirements.txt; do
      [ ! -f /home/butvinm/Dev/orion/examples/c3ae-demo/$f ] && echo "$f: gone" || echo "$f: STILL THERE — FAIL"
  done
  ```
- [x] verify the c3ae-demo HTTP server still vets and builds:
  ```sh
  cd /home/butvinm/Dev/orion/examples/c3ae-demo/server
  go vet ./... && go build ./...
  ```
- [x] verify the README's Quick Start commands are all syntactically valid (don't actually run training — that's hours of compute):
  ```sh
  # The README mentions specific module paths and script paths — confirm they exist
  ls /home/butvinm/Dev/orion/examples/c3ae-demo/models/{c3ae,c3ae_fhe,params,utkface,metrics,train,compile,prep_input,eval}.py
  ls /home/butvinm/Dev/orion/examples/c3ae-demo/scripts/{run_cleartext,run_fhe}.sh
  ls /home/butvinm/Dev/orion/examples/c3ae-demo/bench/{go.mod,main.go,keygen.go,encrypt.go,infer.go,decrypt.go,rss.go}
  ls /home/butvinm/Dev/orion/examples/c3ae-demo/{verify_fhe,build_results}.py
  ```
- [x] git log: 3 or 4 new commits on `experiments` branch (Task 1, 2, 3, optional Task 4), each with the expected message. (4 task commits + 2 plan-mark commits = 6 total ahead of origin/experiments — exec-script artifact, expected.)

### Task 6: [Final] Move plan to completed/

- [x] move this plan: `git mv /home/butvinm/Dev/orion/docs/plans/2026-05-10-c3ae-demo-consolidation.md /home/butvinm/Dev/orion/docs/plans/completed/2026-05-10-c3ae-demo-consolidation.md`
- [x] commit:
  ```sh
  cd /home/butvinm/Dev/orion
  git add docs/plans
  git commit -m "docs: move c3ae-demo consolidation plan to completed/"
  ```

## Post-Completion

_Items requiring manual or external action — informational, no checkboxes._

**External verification (after merge to main):**

- Anyone with a local checkout that has `examples/c3ae-demo/weights.pth` (the old gitignored file) should `mkdir -p examples/c3ae-demo/out && mv examples/c3ae-demo/weights.pth examples/c3ae-demo/out/weights_fhe.pth` to preserve their trained weights without retraining. Document this in the commit body of Task 2.
- The browser demo's full end-to-end flow (kagglehub UTKFace download → train → compile → server → browser client → encrypted inference) is not exercised by the manual-verify steps in this plan. A real run requires UTKFace + ~30 min of GPU/CPU time. Do this on a VPS or local GPU box if you want to confirm the demo still works post-consolidation.
- Plan files in `/home/butvinm/Dev/orion/docs/plans/completed/2026-05-08-c3ae-experiments.md` and `/home/butvinm/Dev/orion/docs/plans/completed/2026-05-09-c3ae-vps-runs.md` reference `examples/c3ae-demo/experiments/...` paths. **Do not update these** — they are historical records of what existed at the time. The new plan (this one) supersedes the layout for future reference.

**Push:**

- After all 4 commits land green, `git push origin experiments`. The branch will be 7 commits ahead of `origin/experiments` (3 reverts + 4 consolidation commits).
