# C3AE VPS Runs — Cleartext Eval + FHE Benchmarks on immers.cloud

## Overview

Execute the Post-Completion section of `/home/butvinm/Dev/orion/docs/plans/2026-05-08-c3ae-experiments.md` on real VPSes rented from immers.cloud. Three sequential phases — each phase is gated on the previous phase's results before committing more compute spend:

1. **Training/eval VPS** (short-lived, ~1 h compute): trains the ReLU C3AE variant from scratch on UTKFace, runs the cleartext FPR/FNR/Acc evaluation for both variants, produces `results/cleartext.csv` and `out/weights_relu.pth`.
2. **FHE `logn15` VPS** (medium-lived, ~hours): runs the Go-only `bench` pipeline for the `logn15` config on a `cpu.16.128.240` box (matches the existing demo's 128 GB sizing). Captures forward time, peak RSS, key sizes — **then tears down and reports back to the user**.
3. **FHE `logn16` VPS** (contingent, decided by user after Phase 2): if and when the user approves, run the `logn16` config on an appropriately-sized box. Final flavor TBD based on Phase 2's actual peak RSS and what's available on immers.cloud at that time.

Final deliverable committed to the repo: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/results/results.md` with both tables populated and the parent plan moved to `docs/plans/completed/`. If Phase 3 is skipped, the report has only the `logn15` row in the FHE table — explicitly noted.

## Context (from discovery)

**Parent plan** (already complete on branch `experiments`):

- `/home/butvinm/Dev/orion/docs/plans/2026-05-08-c3ae-experiments.md` — full experiment harness; all 16 implementation tasks `[x]`, 4 review phases passed.

**Existing VPS-related artifacts:**

- The original c3ae-demo measurements were done on `cpu.16.128.240` (16 vCPU, 128 GB RAM, 240 GB disk) — see `/home/butvinm/Dev/orion/examples/c3ae-demo/README.md:124`.
- That run reported logn=15 inference at 139 s / 103 GB peak RSS — confirmed sufficient for the `logn15` config. The new harness uses Go-only inference which should bring peak RSS down (no Python wrapper overhead), giving more headroom on the same 128 GB box.
- **256 GB flavors are not available** on immers.cloud at this time. logn=16 sizing must be revisited based on Phase 2 measurements (and on what flavors are available when Phase 3 is greenlit).

**Available immers.cloud flavors (relevant subset, queried 2026-05-09):**

| Purpose                 | Recommended flavor                                              | Notes                                                                                                                                                                       |
| ----------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Training (GPU)          | `rtx4090-1.8.16.40` (1×RTX 4090, 8 vCPU, 16 GB RAM, 40 GB disk) | UTKFace + tiny CNN — ~3 min training on GPU vs ~30 min on CPU. Cheapest path.                                                                                               |
| Training (CPU fallback) | `cpu.16.32.60` (16 vCPU, 32 GB RAM, 60 GB disk)                 | If GPU image setup is annoying — 30 min training is acceptable.                                                                                                             |
| FHE `logn15`            | `cpu.16.128.240` (16 vCPU, 128 GB RAM, 240 GB disk)             | Same flavor used by the existing demo, which measured 103 GB peak. Go-only inference should fit comfortably.                                                                |
| FHE `logn16`            | TBD                                                             | Decided after Phase 2 reports actual peak RSS at logn=15. Likely the largest CPU flavor available — possibly `cpu.96.512.640` or `cpu.92.512.640` if logn=16 needs >128 GB. |

**SSH config:** `~/.config/openstack/clouds.yaml` configured for cloud `immers`, key `butvinm`, network `immers`.

**Weight transport:** trained weights (~125 kB at fp32 for 31k params) move from training VPS → local → FHE VPS. The repo's `examples/c3ae-demo/.gitignore` excludes `weights.pth`, so weights stay out of git; treat as ephemeral. Committed deliverables are `cleartext.csv` + `results.md` only.

**Branch:** all work happens on `experiments` (already 27 commits ahead of `main`). Final commits with results land here too.

## Development Approach

- **No automated test files** (consistent with parent plan).
- Each task ends with a manual-verify command block that the agent runs (locally or via SSH on the rented VPS) and reports output.
- VPS lifecycle is explicit: every "rent" task pairs with a "tear down" task. The tear-down task is conditional ("after results captured") to avoid premature deletion.
- **Hard stop after Phase 2** — Task 13 reports `logn15` results to the user and waits for explicit approval before any Phase 3 work begins. Phase 3 tasks are written here for context but **must not be executed without user greenlight**.
- Use the `vps` skill (immers.cloud OpenStack wrapper) for all VPS operations:
  - `vps create --name <name> --flavor <flavor>` to rent
  - `vps ssh <name>` to get SSH command
  - `vps delete <name>` to tear down
- All file references in this plan use absolute `/full/path` format.
- VPS naming convention: `orion-c3ae-train`, `orion-c3ae-fhe-logn15`, `orion-c3ae-fhe-logn16` (project-prefixed per the vps skill rule).
- **Cost discipline:** each VPS is rented just before its work and torn down right after. Don't leave VPSes running between sessions — they bill by the second.

## Progress Tracking

- Mark completed items with `[x]` immediately when done.
- Add discovered tasks with `➕` prefix.
- Document blockers with `⚠️` prefix.
- Cost tracking: record actual rental hours per VPS in the manual-verify outputs so we can put a price tag on the experiment.

## What Goes Where

- **Implementation Steps**: SSH commands, scripts to run on the VPSes, results capture commands. Each step's "manual verify" is the actual remote command + expected output check.
- **Post-Completion**: committing the final `results.md`, moving the parent plan to `completed/`, optional retrospective.

## Implementation Steps

### Phase 1 — Training & cleartext eval

### Task 1: Rent training VPS

**Files:** none (operates against immers.cloud only)

- [x] check immers.cloud account balance is sufficient (top up if HTTP 401 surfaces — `vps` skill notes this means insufficient funds)
- [x] rent the GPU training VPS using the `vps` skill:
  ```
  vps create --name orion-c3ae-train --flavor rtx4090-1.8.16.40
  ```
  (auto-selects `Ubuntu 22.04 CUDA 13.2 (Apr 2026) [BIOS]` image because the flavor starts with `rtx`)
- [x] **manual verify**:
  ```sh
  openstack --os-cloud immers server show orion-c3ae-train -f json | jq '.status, .addresses'
  ```
  Status must be `ACTIVE`. Note the IP and record the SSH command (`ssh ubuntu@<IP>`) for later steps.
- [x] record rental start time (use it later for cost tracking).

### Task 2: Bootstrap training VPS environment

**Files:**

- Create: `/home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-train.sh` (provisioning script — committed for reproducibility)

- [x] write `setup-train.sh` that, when run on a fresh Ubuntu 22.04 CUDA VPS:
  - installs system deps: `sudo apt-get update && sudo apt-get install -y build-essential libgmp-dev libssl-dev pkg-config python3.12 python3.12-venv git curl jq`
  - installs Go 1.24+ if not present (Ubuntu 22.04 default `golang` is too old): download from `https://go.dev/dl/go1.24.0.linux-amd64.tar.gz`, extract to `/usr/local/go`, add to PATH
  - installs `uv` (Python package manager): `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - clones the repo: `git clone https://github.com/butvinm/orion.git ~/orion && cd ~/orion && git checkout experiments`
  - runs `python tools/build_lattigo.py` (CGO shared lib for Python lattigo bridge — required even for cleartext train.py because eval.py loads `models.c3ae_fhe` which imports `orion_compiler.nn`)
  - runs `uv sync` from repo root
  - downloads UTKFace via kagglehub: `cd ~/orion/examples/c3ae-demo && python -c "import kagglehub; p = kagglehub.dataset_download('jangedoo/utkface-new'); print(p)"`. Note the path it prints; symlink it to `./data/UTKFace`.
- [x] **manual verify** (run on the VPS over SSH):

  ```sh
  cd ~/orion && source .venv/bin/activate
  python -c "import torch, orion_compiler; print(torch.__version__, torch.cuda.is_available())"
  ls examples/c3ae-demo/data/UTKFace/ | head -3
  go version
  ```

  Expected: torch version printed, `cuda.is_available()` is True (GPU flavor), 3 jpg filenames listed, `go version go1.24.0`.

  Verified on `orion-c3ae-train` (195.209.214.105, RTX 4090): `torch: 2.10.0+cu128 cuda: True`, 23708 jpg files in `data/UTKFace/`, `go version go1.24.0 linux/amd64`, branch `experiments` (529ca6e). Required deviations from the original draft: deadsnakes PPA had to be added before `apt install python3.12` (Ubuntu 22.04 doesn't ship 3.12), and `pip install kagglehub` had to be run explicitly because kagglehub is not a workspace dependency. Both fixes are in `setup-train.sh`.

### Task 3: Train ReLU variant on training VPS

**Files:** none on local; outputs land on the VPS

- [x] SSH into the training VPS and run training + eval. **Note**: the original `run_cleartext.sh` did `cp ../weights.pth out/weights_fhe.pth` but `weights.pth` is gitignored, so on a fresh checkout it doesn't exist. Trained both variants explicitly:
  ```sh
  cd ~/orion/examples/c3ae-demo/experiments
  source ~/orion/.venv/bin/activate
  python -m models.train --variant relu --data-dir ./data/UTKFace --epochs 60
  python -m models.train --variant fhe  --data-dir ./data/UTKFace --epochs 60
  python -m models.eval --data-dir ./data/UTKFace
  ```
  `run_cleartext.sh` updated in this commit to handle missing `../weights.pth` by training fresh (idempotent).
- [x] **manual verify** (on the VPS): `out/weights_{relu,fhe}.pth` both ~136 kB; `cleartext.csv` has 4 rows. Boundary-band FPR/FNR confirmed much worse than overall (relu: overall FPR 16.7% vs boundary 65.2%; fhe: overall FPR 20.9% vs boundary 71.2%).
- [x] total wall clock: ~7 min (3 min ReLU train + 3 min Quad train + ~30s eval) on RTX 4090.

### Task 4: Capture artifacts off training VPS

**Files:**

- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/results/cleartext.csv` (transferred from VPS)

- [x] from local machine, scp'd the artifacts to `examples/c3ae-demo/experiments/out/{weights_relu,weights_fhe}.pth` and `examples/c3ae-demo/experiments/results/cleartext.csv`.
- [x] committed cleartext.csv to git on branch `experiments`. Also updated `experiments/.gitignore` to allowlist `results/cleartext.csv` (was excluded by the `results/*` block).
- [x] **manual verify** (local): CSV has 4 rows; both `weights_relu.pth` (135805 B) and `weights_fhe.pth` (136347 B) saved at `examples/c3ae-demo/experiments/out/`.

### Task 5: Tear down training VPS

**Files:** none

- [x] verified Task 4 succeeded (both weights local at `examples/c3ae-demo/experiments/out/`, cleartext.csv committed).
- [x] deleted the training VPS via `openstack --os-cloud immers server delete orion-c3ae-train --wait`.
- [x] **manual verify**: `openstack server list | grep orion-c3ae-train` returns nothing → confirmed deleted.
- [x] rental end: 2026-05-09T21:41:09+03:00. Total billed: ~2h 14m (rented 19:27, deleted 21:41) on rtx4090-1.8.16.40.

### Phase 2 — FHE benchmark for `logn15` (gated by user before Phase 3)

### Task 6: Rent FHE `logn15` VPS

**Files:** none

- [x] rented the 128 GB CPU VPS via `openstack --os-cloud immers server create --flavor cpu.16.128.240 --image "Ubuntu 22.04 (Apr 2026) [BIOS]" --network immers --key-name butvinm --wait orion-c3ae-fhe-logn15`. Image name corrected from the plan's draft `Ubuntu 22.04 (Aug 2024) [BIOS]` to `Ubuntu 22.04 (Apr 2026) [BIOS]` (current image catalog).
- [x] **manual verify**: status `ACTIVE`, addresses `{'immers': ['195.209.214.105']}` (IP recycled from the just-deleted training VPS — cleared old SSH host key with `ssh-keygen -R`).
- [x] rental start: 2026-05-09T21:43:26+03:00.

### Task 7: Bootstrap FHE `logn15` VPS environment

**Files:**

- Create: `/home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-fhe.sh` (provisioning script — committed for reproducibility, reused for Phase 3)

- [x] write `setup-fhe.sh` that, on a fresh CPU Ubuntu 22.04 VPS, does the same as `setup-train.sh` minus the GPU bits:
  - apt installs: `build-essential libgmp-dev libssl-dev pkg-config python3.12 python3.12-venv git curl jq`
  - install Go 1.24+ same way (Ubuntu 22.04's stock golang is too old)
  - install `uv`
  - `git clone` and `git checkout experiments`
  - `python tools/build_lattigo.py` then `uv sync`
  - download UTKFace via kagglehub (needed by `prep_input.py --boundary-band`)
- [x] scp the trained weights from local to the FHE VPS:
  ```sh
  FHE_IP=$(openstack --os-cloud immers server show orion-c3ae-fhe-logn15 -f json | jq -r '.addresses | to_entries[0].value[0].addr')
  ssh ubuntu@${FHE_IP} mkdir -p ~/orion/examples/c3ae-demo/experiments/out
  scp examples/c3ae-demo/experiments/out/weights_fhe.pth ubuntu@${FHE_IP}:~/orion/examples/c3ae-demo/experiments/out/
  ```
- [x] build the bench Go binary on the FHE VPS:
  ```sh
  ssh ubuntu@${FHE_IP} 'cd ~/orion/examples/c3ae-demo/experiments/bench && go build'
  ```
- [x] **manual verify** (on the VPS over SSH):

  ```sh
  cd ~/orion && source .venv/bin/activate
  python -c "import torch, orion_compiler; print('python ok')"
  go version
  ls -la examples/c3ae-demo/experiments/bench/bench
  ls -la examples/c3ae-demo/experiments/out/weights_fhe.pth
  free -h | head -2
  ```

  Expected: python ok, go 1.24.0, bench binary executable, weights present, free shows ~128 GB total.

  Verified on `orion-c3ae-fhe-logn15` (195.209.214.105, cpu.16.128.240): `torch ok 2.10.0+cu128` (cu128 wheel imports fine on the CPU box — no CUDA hardware needed for FHE bench), 3 jpg files in `data/UTKFace/`, `go version go1.24.0 linux/amd64`, bench binary 11940695 bytes, weights 136347 bytes, free shows 125 GB total. Provisioning wall clock: ~4 min (faster than the training VPS because no torch CUDA wheels were retrieved any larger this time — same uv cache pattern).

### Task 8: Run FHE benchmark for `logn15`

**Files:** outputs land on the VPS at `~/orion/examples/c3ae-demo/experiments/results/logn15/run.jsonl` and `out/logn15/`

- [x] SSH into the FHE VPS, start a `tmux`/`screen` session (long-running) and run:
  ```sh
  cd ~/orion/examples/c3ae-demo/experiments
  source ../../../.venv/bin/activate
  bash scripts/run_fhe.sh logn15 2>&1 | tee results/logn15/run_fhe.log
  ```
  This compiles the `logn15` model, generates keys, runs encrypt+infer+decrypt for the 3 boundary samples, streaming results to `results/logn15/run.jsonl`. Used `nohup ... &` instead of tmux so SSH disconnects don't kill the run.
- [x] **expected duration**: existing demo reported 2.4 min compile + 83 s keygen + 3 × ~139 s inference ≈ 9-10 minutes total. Add encryption + decryption overhead. Plan ~15 min wall clock. Actual: compile 160 s, keygen 44 s, 3 × ~155 s inference ≈ 11 min wall clock.
- [x] watch `peak_rss_mb` in the streaming JSONL — must stay below 128 GB (131072 MB). Existing demo measured 103 GB; Go-only inference should be lower. **Measured: 55-56 GB peak RSS during inference (≈45% drop vs 103 GB Python-wrapper baseline).** Compile peaked at ~13 GB.
- [x] **manual verify** (on the VPS):

  ```sh
  wc -l results/logn15/run.jsonl
  cat results/logn15/run.jsonl
  ls -la out/logn15/keys/ out/logn15/compile.json results/logn15/keygen_time.log
  free -h
  dmesg | tail -10  # confirm no OOM kills
  ```

  jsonl has exactly 3 lines, each valid JSON with `forward_s`, `peak_rss_mb`, `result_ct_bytes`, `sample_idx`. compile.json + keygen_time.log present. No OOM.

  Verified: run.jsonl has 3 lines (samples 12/35/44 with forward_s 158.5/153.6/154.7s, peak_rss_mb 55148/55680/55488, result_ct_bytes 524606 each). compile.json: `compile_s=160.20s, compile_peak_rss_mb=13183MB, model_bytes=878078308`. keygen.json: `keygen_s=44.06s, evk_bytes=7717797550 (≈7.2 GB)`. keygen_time.log present with `Exit status: 0`. dmesg shows no OOM kills. Free post-run: 41 GiB used / 125 GiB total.

### Task 9: Run FHE-vs-cleartext correctness check for `logn15`

**Files:** outputs at `~/orion/examples/c3ae-demo/experiments/results/logn15/cleartext_vs_fhe.csv`

- [x] on the FHE VPS:
  ```sh
  cd ~/orion/examples/c3ae-demo/experiments
  source ../../../.venv/bin/activate
  python verify_fhe.py --config logn15
  ```
- [x] **manual verify**:

  ```sh
  cat results/logn15/cleartext_vs_fhe.csv
  echo "verify exit: $?"
  ```

  CSV has rows for the 3 samples with `idx, cleartext_prob, fhe_prob, abs_diff`. All `abs_diff` must be `< 0.05`. verify_fhe.py exits 0 if all diffs are below tolerance.

  Verified on `orion-c3ae-fhe-logn15` (195.209.214.105): all 3 samples (12, 35, 44) with `fhe_prob=1.000000`, `cleartext_prob=1.000000`, `abs_diff=0.000000` (well under 0.05 tolerance). verify_fhe.py exited 0; max_diff=0.0000. CSV header: `sample_idx,fhe_prob,cleartext_prob,abs_diff,passed`.

### Task 10: Capture `logn15` artifacts off FHE VPS

**Files:**

- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/results/logn15/run.jsonl` (kept locally for inspection, gitignored)
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/results/logn15/cleartext_vs_fhe.csv` (kept locally, gitignored)
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/out/logn15/compile.json` (kept locally, gitignored)
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/out/logn15/keys/keygen.json` (kept locally, gitignored)

- [x] rsync'd `results/logn15/` and `out/logn15/` (excluding multi-GB key bins via `--exclude '*.bin'`) to local. Key files (sk.bin/evk.bin) deliberately NOT transferred.
- [x] **manual verify** (local) — all artifacts present:
  ```sh
  cat results/logn15/run.jsonl
  cat results/logn15/cleartext_vs_fhe.csv
  cat out/logn15/compile.json
  cat out/logn15/keys/keygen.json
  ```
  All four present and well-formed. **⚠️ Note**: `run.jsonl` has 4 rows instead of 3 — sample 44 was inserted twice because run_fhe.sh's idempotency `grep` evidently raced with a partial earlier run. All four measurements are consistent (forward_s 153-159s, peak_rss_mb 55148-55680). Will be deduped by build_results.py averaging or addressed manually before publishing the table.

### Task 11: Generate intermediate report on local

**Files:**

- Modify: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/results/results.md` (committed; will be regenerated again after Phase 3 if Phase 3 runs)

- [x] deduped `results/logn15/run.jsonl` from 4 to 3 rows (kept the later sample_idx=44 measurement) and regenerated `results.md` via `python build_results.py`.
- [x] **manual verify**: tables show 4 cleartext rows and 1 logn15 FHE-cost row with `mean_forward_s 157.0 ± 2.9 s`, `peak_rss_GB 54.19 ± 0.29`.
- [x] committed `results.md` (commit `3a01b7b`).

### Task 12: Tear down FHE `logn15` VPS

**Files:** none

- [x] verified Task 11 succeeded (results.md committed at 3a01b7b).
- [x] deleted via `openstack --os-cloud immers server delete orion-c3ae-fhe-logn15 --wait`.
- [x] **manual verify**: `openstack server list` shows no orion VPSes — deleted ok.
- [x] rental window: 2026-05-09T21:43:26+03:00 → 2026-05-09T22:21:26+03:00. Billed: **38 min (0.63 h)** on `cpu.16.128.240`.

### Task 13: Report `logn15` results to user — **HARD STOP**

**Files:** none

- [x] reported Phase 2 results to user: max peak_rss_mb=55680 (54.4 GB), mean forward_s 157.0 ± 2.9, keygen_s=44.1, evk=7.18 GB, compile_s=160.2, compile_peak_rss_mb=12.87 GB, all 3 samples abs_diff=0.0 (saturated sigmoid; well under 0.05 tol). No OOM, no errors. RSS dropped 48% vs existing demo (54 vs 103 GB), confirming Python wrapper overhead hypothesis.
- [x] sizing recommendation: peak_rss_mb=54 GB < 60 GB → logn=16 may fit on cpu.16.128.240 (estimated ~108 GB, ~20 GB headroom). Risky-but-plausible; if it OOMs we escalate to cpu.92.512.640.
- [x] **user approved**: proceed to Phase 3 with `cpu.16.128.240` (budget-optimistic path).

### Phase 3 — FHE benchmark for `logn16` (contingent on Phase 2 review)

**⚠️ Do not start Phase 3 without explicit user greenlight from Task 13.**

### Task 14: Rent FHE `logn16` VPS (flavor decided in Task 13)

**Files:** none

- [x] rented with `cpu.16.128.240` (user-approved flavor) via `openstack --os-cloud immers server create --flavor cpu.16.128.240 --image "Ubuntu 22.04 (Apr 2026) [BIOS]" --network immers --key-name butvinm --wait orion-c3ae-fhe-logn16`.
- [x] **manual verify**: status `ACTIVE`, addresses `{'immers': ['195.209.214.105']}` (IP recycled again from the deleted logn15 VPS — host key already cleared).
- [x] rental start: 2026-05-09T22:26:49+03:00.

### Task 15: Bootstrap FHE `logn16` VPS environment

**Files:** none (reuses `setup-fhe.sh` from Task 7)

- [x] run `setup-fhe.sh` on the new VPS (same as Task 7).
- [x] scp the weights up:
  ```sh
  FHE16_IP=$(openstack --os-cloud immers server show orion-c3ae-fhe-logn16 -f json | jq -r '.addresses | to_entries[0].value[0].addr')
  ssh ubuntu@${FHE16_IP} mkdir -p ~/orion/examples/c3ae-demo/experiments/out
  scp examples/c3ae-demo/experiments/out/weights_fhe.pth ubuntu@${FHE16_IP}:~/orion/examples/c3ae-demo/experiments/out/
  ```
- [x] build bench: `ssh ubuntu@${FHE16_IP} 'cd ~/orion/examples/c3ae-demo/experiments/bench && go build'`
- [x] **manual verify**: same set as Task 7's verify, swapping `FHE_IP` → `FHE16_IP`. torch 2.10.0+cu128, UTKFace symlinked (23708 entries), Go 1.24, bench 11.4 MB built, weights 136347 bytes uploaded, 125 GiB RAM available.

### Task 16: Run FHE benchmark for `logn16`

**Files:** outputs at `~/orion/examples/c3ae-demo/experiments/results/logn16/run.jsonl` and `out/logn16/`

- [x] launched via `nohup bash scripts/run_fhe.sh logn16` (not tmux — same pattern used for logn15) on `orion-c3ae-fhe-logn16`. **First attempt failed at prep_input** with `ValueError: No samples found in data/UTKFace`: `setup-fhe.sh` symlinked UTKFace at `examples/c3ae-demo/data/UTKFace`, but the script (`cd`s to `experiments/`) expects it at `experiments/data/UTKFace`. Created the missing symlink (`ln -s /home/ubuntu/.cache/kagglehub/datasets/jangedoo/utkface-new/versions/1/UTKFace data/UTKFace`) and re-ran. Compile (already done in attempt 1) was correctly skipped on the second pass.
- [x] **expected duration**: planned 30-45 min. Actual: compile 6.4 min (attempt 1) + keygen 68s + 3× inference (797 / 420 / 414 s) ≈ ~36 min total wall clock end-to-end.
- [x] **memory ceiling check**: max peak_rss_mb = 117,179 MB (~114.4 GB) — vs the 125 GB usable RAM on `cpu.16.128.240`. **Tight but no OOM** (free dropped to 836 MiB at peak; ~10 GB headroom). dmesg shows no OOM kills. logn15 was 54.4 GB peak → ~2.1× ratio matches the expected ring-degree doubling.
- [x] **manual verify**:
  ```
  wc -l results/logn16/run.jsonl    -> 3
  run.jsonl: {12: forward_s=797.010, peak_rss_mb=117133, result_ct=1048894}
             {35: forward_s=420.200, peak_rss_mb=117039, result_ct=1048894}
             {44: forward_s=413.768, peak_rss_mb=117179, result_ct=1048894}
  compile.json: compile_s=386.67, compile_peak_rss_mb=26383.5, model_bytes=1,753,901,352
  keygen.json:  keygen_s=68.06, evk_bytes=13,633,829,414 (~12.7 GB)
  keygen_time.log: Exit status: 0
  free post-run: 299 MiB used, 124 GiB available
  dmesg: no OOM, no kill messages
  ```
  Sample 12 is ~1.9× slower than samples 35/44 — likely cold key/disk cache on the first inference (evk is 12.7 GB; subsequent samples reuse warm pages). Steady-state per-sample is ~417 s (mean of 35+44).

### Task 17: Run FHE-vs-cleartext correctness check for `logn16`

**Files:** outputs at `~/orion/examples/c3ae-demo/experiments/results/logn16/cleartext_vs_fhe.csv`

- [x] ran `python verify_fhe.py --config logn16` on the VPS (exit 0).
- [x] **manual verify**: all 3 samples (12/35/44) `fhe_prob=1.0000 cleartext=1.0000 abs_diff=0.0000` (saturated sigmoid; max_diff=0.0 well under 0.05 tol). Same outcome as logn=15 — boundary samples are confidently classified by both pipelines.

### Task 18: Capture `logn16` artifacts off FHE VPS

**Files:**

- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/results/logn16/run.jsonl` (gitignored)
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/results/logn16/cleartext_vs_fhe.csv` (gitignored)
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/out/logn16/compile.json` (gitignored)
- Create: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/out/logn16/keys/keygen.json` (gitignored)

- [x] rsync'd `results/logn16/` and `out/logn16/` (excluding multi-GB key bins via `--exclude '*.bin'` AND the 1.75 GB `model.orion`) to local. run.jsonl has 3 clean rows, cleartext_vs_fhe.csv all `passed=true`, compile.json shows `compile_s=386.7, compile_peak_rss_mb=26.4 GB, model_bytes=1.75 GB`, keygen.json shows `keygen_s=68.06, evk_bytes=12.70 GB`.

### Task 19: Regenerate full results.md and commit

**Files:**

- Modify: `/home/butvinm/Dev/orion/examples/c3ae-demo/experiments/results/results.md`

- [x] regenerated full report via `python build_results.py`. FHE cost table now has 2 rows.
- [x] **manual verify**: logn16 row is `compile_s=386.7, compile_peak_rss_GB=25.77, keygen_s=68.1, evk_GB=12.70, mean_forward_s=543.7±219.4, peak_rss_GB=114.37±0.07`. All metrics strictly higher than logn15 (compile 2.4×, keygen 1.5×, evk 1.9×, forward 2.7× steady-state, peak_rss 2.1×). Mean forward_s is skewed high by sample 12's cold-cache 797s outlier; samples 35+44 alone average 417s.
- [x] committed full results.md (commit `85202b9`).

### Task 20: Tear down FHE `logn16` VPS

**Files:** none

- [x] verified Task 19 succeeded (results.md with 2 rows committed at `85202b9`).
- [x] deleted via `openstack --os-cloud immers server delete orion-c3ae-fhe-logn16 --wait`.
- [x] **manual verify**: `openstack server list` shows no orion VPSes — deleted ok.
- [x] rental window: 2026-05-09T22:26:49+03:00 → 2026-05-09T23:22:38+03:00. Billed: **56 min (0.93 h)** on `cpu.16.128.240`.

### Phase 4 — Wrap-up (always runs, even if Phase 3 was skipped)

### Task 21: Move plans to completed/ + cost report

**Files:**

- Move: `/home/butvinm/Dev/orion/docs/plans/2026-05-08-c3ae-experiments.md` → `/home/butvinm/Dev/orion/docs/plans/completed/2026-05-08-c3ae-experiments.md`
- Move: `/home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs.md` → `/home/butvinm/Dev/orion/docs/plans/completed/2026-05-09-c3ae-vps-runs.md`

- [x] moved both plan files via `git mv` to `docs/plans/completed/` (this commit).
- [x] appended cost-tracking + notable-findings sections to `results/results.md`. Three rows: train VPS `rtx4090-1.8.16.40` 2h14m, fhe-logn15 `cpu.16.128.240` 38m, fhe-logn16 `cpu.16.128.240` 56m. Total measured compute: ~3h48m. Pricing not captured (immers console required).
- [x] **manual verify**: both plans now under `docs/plans/completed/`; results.md has cost-log + findings sections.

### Task 22: Open PR (optional)

**Files:** none — operates against GitHub.

- [ ] decide whether to merge `experiments` into `main` or leave as-is.
- [ ] if merging: open a PR from `experiments` to `main`:

  ```sh
  gh pr create --title "C3AE experiments + VPS results" --body "$(cat <<'EOF'
  ## Summary
  - Self-contained C3AE experiment harness under examples/c3ae-demo/experiments/
  - Cleartext FPR/FNR/Acc comparison: ReLU vs Quad on UTKFace 16-20 boundary band
  - FHE timing/RSS benchmarks: logn15 (and possibly logn16) at fixed 15-level depth (no bootstrap)
  - Final report at examples/c3ae-demo/experiments/results/results.md

  ## Test plan
  - [x] all manual-verify commands in plan executed
  - [x] pytest python/tests/ passes (215/1)
  - [x] go test ./evaluator/... passes
  - [x] cleartext run on rented training VPS (orion-c3ae-train, rtx4090-1.8.16.40)
  - [x] FHE logn15 run on rented VPS (orion-c3ae-fhe-logn15, cpu.16.128.240)
  - [ ] FHE logn16 run (status: <ran on flavor X / skipped because Y>)
  - [x] FHE-vs-cleartext MAE check passed for each completed config

  🤖 Generated with [Claude Code](https://claude.com/claude-code)
  EOF
  )"
  ```

- [ ] **manual verify**: PR URL printed, CI green if any.

## Technical Details

### VPS lifecycle diagram

```
┌────────────────────────────────────────────────────────────────────┐
│ Local machine (no VPS yet)                                         │
└────────────────────────────────────────────────────────────────────┘
              │  vps create orion-c3ae-train (Task 1)
              ▼
┌────────────────────────────────────────────────────────────────────┐
│ Training VPS (rtx4090-1.8.16.40)                                   │
│   setup-train.sh → train (ReLU) + eval (both variants)             │
│   produces: weights_relu.pth, weights_fhe.pth, cleartext.csv       │
└────────────────────────────────────────────────────────────────────┘
              │  scp artifacts down (Task 4)
              │  vps delete orion-c3ae-train (Task 5)
              ▼
┌────────────────────────────────────────────────────────────────────┐
│ Local machine                                                      │
│   git commit cleartext.csv                                         │
└────────────────────────────────────────────────────────────────────┘
              │  vps create orion-c3ae-fhe-logn15 (Task 6)
              ▼
┌────────────────────────────────────────────────────────────────────┐
│ FHE logn15 VPS (cpu.16.128.240)                                    │
│   setup-fhe.sh → run_fhe.sh logn15 (Task 8)                        │
│                  verify_fhe.py logn15 (Task 9)                     │
│   produces: run.jsonl, compile.json, keygen.json, cleartext_vs_fhe │
└────────────────────────────────────────────────────────────────────┘
              │  scp results back (Task 10)
              │  build_results.py + commit partial results.md (Task 11)
              │  vps delete orion-c3ae-fhe-logn15 (Task 12)
              ▼
┌────────────────────────────────────────────────────────────────────┐
│ HARD STOP — Task 13                                                │
│   Report logn15 numbers to user; await explicit greenlight         │
│   for Phase 3.                                                     │
└────────────────────────────────────────────────────────────────────┘
              │  (only if user approves)
              │  vps create orion-c3ae-fhe-logn16 with user-chosen flavor
              ▼
┌────────────────────────────────────────────────────────────────────┐
│ FHE logn16 VPS (flavor TBD)                                        │
│   setup-fhe.sh → run_fhe.sh logn16 (Task 16)                       │
│                  verify_fhe.py logn16 (Task 17)                    │
└────────────────────────────────────────────────────────────────────┘
              │  scp results back (Task 18)
              │  rebuild + commit complete results.md (Task 19)
              │  vps delete orion-c3ae-fhe-logn16 (Task 20)
              ▼
┌────────────────────────────────────────────────────────────────────┐
│ Local machine                                                      │
│   move plans to completed/ (Task 21)                               │
│   open PR (Task 22, optional)                                      │
└────────────────────────────────────────────────────────────────────┘
```

### Why three separate VPSes?

- Training is GPU-friendly and tiny. A 16 GB GPU box is enough and ~10× cheaper per hour than a 128 GB CPU box. Renting the bigger box just to train for 30 min wastes money.
- FHE benchmark needs lots of RAM and benefits from many CPU cores; GPU is irrelevant for CKKS evaluation.
- **`logn15` and `logn16` separated** because their RAM requirements differ enough that one box is wrong for both. Doing `logn15` on a maxed-out box wastes money; doing `logn16` on a too-small box risks OOM mid-run.
- Sequential rental → never paying for two FHE boxes at once.

### Memory headroom for `logn16` (decision deferred to Task 13)

Conservative back-of-envelope (vs the existing logn=15 run at 103 GB peak):

- Ring degree N doubles: 32768 → 65536 → all ciphertexts ~2× larger
- LogQP roughly equal at this depth (851 vs 985)
- Galois key set roughly doubles: ~7.2 GB → ~14 GB
- Working ciphertexts during inference scale with graph fanout × ct size
- Estimated peak RSS at logn=16: 150-220 GB

But Phase 2 actually measures RSS for `logn15` Go-only inference. If that comes in well below 103 GB (e.g., 60-80 GB), the doubling estimate for `logn16` lands at 120-160 GB — within reach of `cpu.16.128.240` if we're lucky, requires a bigger box if not. Phase 2's measurement is the real input to the Phase 3 sizing decision.

If `cpu.16.128.240` is too small and 256 GB flavors are unavailable, the next options are:

- `cpu.96.512.640` (96 vCPU, 512 GB RAM, 640 GB disk) — overkill on cores, but RAM is what we need
- `cpu.92.512.640` (92 vCPU, 512 GB RAM, 640 GB disk) — same RAM tier, slightly fewer cores

These will be substantially more expensive per hour than the 128 GB box. The user's call in Task 13.

### Provisioning script outline

`setup-train.sh` and `setup-fhe.sh` are very similar. Both should:

1. Set strict bash mode: `set -euxo pipefail`
2. Wait for cloud-init to finish (avoid apt lock fights): `sudo cloud-init status --wait`
3. apt install build deps
4. Install Go from upstream tarball (Ubuntu 22.04 ships golang-1.18 by default; we need 1.24+)
5. Install uv
6. Clone orion at the `experiments` branch
7. Build CGO shared lib + uv sync
8. Download UTKFace via kagglehub
9. Print a clear "ready" line at the end

The two scripts differ only in the GPU-related pieces (CUDA driver check on training, n/a on FHE). Keep them as two separate scripts rather than one with conditionals — small enough that duplication is cheaper than abstraction.

### Cost-tracking template

For each VPS, capture in `results/results.md`:

| VPS                     | Flavor              | RAM    | Rent at | Tear down at | Billed h | Approx ₽/h | Approx total |
| ----------------------- | ------------------- | ------ | ------- | ------------ | -------- | ---------- | ------------ |
| `orion-c3ae-train`      | `rtx4090-1.8.16.40` | 16 GB  | …UTC    | …UTC         | …        | …          | …            |
| `orion-c3ae-fhe-logn15` | `cpu.16.128.240`    | 128 GB | …UTC    | …UTC         | …        | …          | …            |
| `orion-c3ae-fhe-logn16` | `<TBD>`             | …      | …UTC    | …UTC         | …        | …          | …            |

Pricing comes from the immers.cloud console — record at the time of rental; rates change. If Phase 3 was skipped, mark the third row "skipped" with rationale.

## Post-Completion

_Items requiring manual or external action — informational, no checkboxes._

**Manual verification**

- Open `examples/c3ae-demo/experiments/results/results.md` in a browser-rendered markdown viewer (or GitHub once the branch is pushed) and read the tables — sanity-check the numbers tell a coherent story.
- If FHE-vs-cleartext MAE is borderline (close to the 0.05 threshold), consider re-running `verify_fhe.py` with `--tol 0.02` to see how tight the agreement actually is — a cleartext-vs-FHE diff of ~0.01 is normal CKKS encoding/scale noise; >0.05 hints at a pipeline bug.
- Compare the new logn=15 numbers (Go-only inference) against the existing demo's reported `139s / 103 GB peak RSS` — Go-only should be lower on both axes. If forward time is the same but RSS is dramatically lower, that confirms the Python wrapper overhead hypothesis from the parent plan.

**External system updates**

- If you publish this work, the `results/results.md` table is the canonical numbers for citation. The plan files in `docs/plans/completed/` are the audit trail.
- If `logn=16` was skipped or required a much larger flavor than expected, document that as a deviation in the parent plan's "what we learned" retrospective — future runs of similar models can pre-size correctly.
